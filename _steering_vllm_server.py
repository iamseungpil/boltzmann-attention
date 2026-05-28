"""vLLM-based steering server (Phase 2a/2b).

Monkey-patches Qwen2DecoderLayer / LlamaDecoderLayer.forward at process start,
then invokes vllm.entrypoints.openai.api_server with the user's vLLM args.

Steering is BAKED INTO THE SERVER at startup — restart between α values.
This gives identical generation quality to the baseline vLLM (same engine,
batching, KV cache, tool-call parser) plus our residual-stream injection.

Usage (CLI args split into "patcher" args and "vllm passthrough" args):
  python _steering_vllm_server.py \
    --steering-vectors steering_vectors_qwen7b.pt \
    --relation validates --alpha 0.5 --layers 12,13,14 \
    -- \
    --model Qwen/Qwen2.5-7B-Instruct --port 9200 --host 0.0.0.0 \
    --served-model-name qwen7b-steer --tool-call-parser hermes \
    --enable-auto-tool-choice --dtype bfloat16 --max-model-len 32768 \
    --gpu-memory-utilization 0.85

α=0.0 (no steering) is also valid — no hook installed.

Env override:
  PHASE2A_STEERING_LOG=1 to enable verbose patching logs.
"""
import argparse
import json
import os
import sys
import re

import torch


def parse_patcher_args():
    """Split sys.argv at the first '--' into our args + vllm passthrough."""
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        our_argv = sys.argv[1:idx]
        vllm_argv = sys.argv[idx + 1:]
    else:
        our_argv, vllm_argv = sys.argv[1:], []

    ap = argparse.ArgumentParser(prog="steering_vllm_server")
    ap.add_argument("--steering-vectors", default=None)
    ap.add_argument("--relation", default=None)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--layers", default="")
    ap.add_argument("--target-layer-class", default="auto",
                    choices=["auto", "qwen2", "llama"],
                    help="Which decoder layer class to patch.")
    args = ap.parse_args(our_argv)
    return args, vllm_argv


def install_patch(args):
    """Install monkey-patch on decoder layer; return (active, summary_str)."""
    if args.alpha == 0.0 or not args.steering_vectors or not args.relation:
        print(f"[steer-vllm] steering DISABLED (alpha={args.alpha} rel={args.relation})")
        return False, "disabled"

    layer_idxs = [int(x) for x in args.layers.split(",") if x.strip()]
    if not layer_idxs:
        print("[steer-vllm] steering DISABLED (no layers)")
        return False, "no-layers"

    print(f"[steer-vllm] loading steering vectors from {args.steering_vectors}")
    blob = torch.load(args.steering_vectors, map_location="cpu", weights_only=False)
    vectors_all = blob["vectors"]
    if args.relation not in vectors_all:
        raise ValueError(f"relation '{args.relation}' not in vectors file; "
                         f"available: {list(vectors_all)[:10]}...")
    rel_vecs = vectors_all[args.relation]
    # normalise each layer vector to unit, cache on GPU bf16
    vecs_per_layer = {}
    for L in layer_idxs:
        if L not in rel_vecs:
            print(f"[steer-vllm] WARN layer {L} missing from vectors; skipping")
            continue
        v = rel_vecs[L]
        v = v / (v.norm() + 1e-8)
        vecs_per_layer[L] = v.to(dtype=torch.bfloat16, device="cuda").contiguous()
    active_layers = set(vecs_per_layer.keys())
    if not active_layers:
        print("[steer-vllm] no valid layers; steering DISABLED")
        return False, "no-valid-layers"

    alpha = float(args.alpha)

    # decide which layer class(es) to patch
    target = args.target_layer_class
    if target == "auto":
        # we patch BOTH; whichever the model loads will be used
        targets = ["qwen2", "llama"]
    else:
        targets = [target]

    for t in targets:
        if t == "qwen2":
            from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer as DL
        elif t == "llama":
            from vllm.model_executor.models.llama import LlamaDecoderLayer as DL
        else:
            continue
        _orig_init = DL.__init__
        _orig_forward = DL.forward

        def make_patched_init(orig):
            def patched_init(self, *args, **kwargs):
                orig(self, *args, **kwargs)
                prefix = kwargs.get("prefix") or (args[1] if len(args) > 1 else "")
                m = re.search(r"\.layers\.(\d+)", str(prefix))
                self._steer_layer_idx = int(m.group(1)) if m else -1
            return patched_init

        def make_patched_forward(orig, alpha_=alpha, vecs_=vecs_per_layer, layers_=active_layers):
            def patched_forward(self, positions, hidden_states, residual):
                out = orig(self, positions, hidden_states, residual)
                idx = getattr(self, "_steer_layer_idx", -1)
                if idx in layers_:
                    v = vecs_[idx]
                    if isinstance(out, tuple):
                        h = out[0] + alpha_ * v
                        return (h,) + out[1:]
                    return out + alpha_ * v
                return out
            return patched_forward

        DL.__init__ = make_patched_init(_orig_init)
        DL.forward = make_patched_forward(_orig_forward)
        print(f"[steer-vllm] patched {t} decoder layer class")

    summary = f"relation={args.relation} alpha={alpha} layers={sorted(active_layers)}"
    print(f"[steer-vllm] ACTIVE: {summary}")
    return True, summary


def main():
    args, vllm_argv = parse_patcher_args()
    active, summary = install_patch(args)
    if active:
        # also expose via env so process info / clients can see
        os.environ["STEERING_SUMMARY"] = summary

    # Now hand off to vllm openai api server.
    print(f"[steer-vllm] launching vllm OpenAI api_server with args: {vllm_argv}")
    sys.argv = ["api_server"] + vllm_argv
    # use runpy to mimic `python -m vllm.entrypoints.openai.api_server`
    import runpy
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
