"""vLLM-based steering server with CONTEXT GATING (Phase 2c / H4).

This is a SAFE COPY of _steering_vllm_server.py that adds opt-in gating modes
to patched_forward. The original file is left untouched so the live Phase 2a
grid is unaffected. Default --gate-mode=none reproduces the original raw-add.

Gating modes (--gate-mode):
  none    : h + alpha * v                       (identical to original raw add)
  decay   : h + alpha * d(pos) * v              (progress-decay; d=clamp(1-pos/P0,0,1))
            -> reduces steering late in conversation so the agent is free to
               pivot to discrete actions (e.g. transfer_to_human_agents) instead
               of being trapped in self-validation loops. Motivated by the
               -0.50 task trajectory analysis (2026-05-29).
  orth    : h + alpha * (v - (v.h_hat) h_hat)   (Gram-Schmidt; add only the
            component of v orthogonal to the current hidden direction -> curbs
            over-amplification that produces the U-shape harm at high alpha).
  softcos : h + alpha * g * v, g=clamp(1-cos(h,v),0,1)  (suppress where the
            steering direction is already expressed in the hidden state).

decode-only gating (skip prompt prefill) is left as future work; it needs the
vLLM forward_context attn_metadata which is fragile across versions.

Usage: same as _steering_vllm_server.py, plus:
  --gate-mode decay --gate-decay-pos 8000
"""
import argparse
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

    ap = argparse.ArgumentParser(prog="steering_vllm_server_gated")
    ap.add_argument("--steering-vectors", default=None)
    ap.add_argument("--relation", default=None)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--layers", default="")
    ap.add_argument("--target-layer-class", default="auto",
                    choices=["auto", "qwen2", "llama"],
                    help="Which decoder layer class to patch.")
    ap.add_argument("--gate-mode", default="none",
                    choices=["none", "decay", "orth", "softcos"],
                    help="Context-gating mode for steering injection.")
    ap.add_argument("--gate-decay-pos", type=float, default=8000.0,
                    help="For --gate-mode=decay: position at which steering "
                         "decays to 0 (linear from full at pos=0).")
    args = ap.parse_args(our_argv)
    return args, vllm_argv


def _steer(h, v, alpha, mode, positions, decay_p0):
    """Apply gated steering to hidden states h ([T,H]) with unit vector v ([H])."""
    if mode == "none":
        return h + alpha * v
    if mode == "decay":
        pos = positions.to(torch.float32)
        d = (1.0 - pos / float(decay_p0)).clamp(0.0, 1.0)  # [T]
        return h + alpha * d.unsqueeze(-1) * v
    if mode == "orth":
        hf = h.to(torch.float32)
        vf = v.to(torch.float32)
        hn = hf / (hf.norm(dim=-1, keepdim=True) + 1e-8)
        coef = (vf.unsqueeze(0) * hn).sum(dim=-1, keepdim=True)  # [T,1] = v . h_hat
        v_perp = vf.unsqueeze(0) - coef * hn                     # [T,H]
        return (hf + alpha * v_perp).to(h.dtype)
    if mode == "softcos":
        hf = h.to(torch.float32)
        vf = v.to(torch.float32)
        hn = hf / (hf.norm(dim=-1, keepdim=True) + 1e-8)
        cos = (hn * vf.unsqueeze(0)).sum(dim=-1, keepdim=True)   # [T,1]
        g = (1.0 - cos).clamp(0.0, 1.0)
        return (hf + alpha * g * vf.unsqueeze(0)).to(h.dtype)
    return h + alpha * v


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
    gate_mode = args.gate_mode
    decay_p0 = float(args.gate_decay_pos)

    target = args.target_layer_class
    if target == "auto":
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

        def make_patched_forward(orig, alpha_=alpha, vecs_=vecs_per_layer,
                                 layers_=active_layers, mode_=gate_mode,
                                 decay_p0_=decay_p0):
            def patched_forward(self, positions, hidden_states, residual):
                out = orig(self, positions, hidden_states, residual)
                idx = getattr(self, "_steer_layer_idx", -1)
                if idx not in layers_:
                    return out
                v = vecs_[idx]
                if isinstance(out, tuple):
                    h = _steer(out[0], v, alpha_, mode_, positions, decay_p0_)
                    return (h,) + out[1:]
                return _steer(out, v, alpha_, mode_, positions, decay_p0_)
            return patched_forward

        DL.__init__ = make_patched_init(_orig_init)
        DL.forward = make_patched_forward(_orig_forward)
        print(f"[steer-vllm] patched {t} decoder layer class")

    summary = (f"relation={args.relation} alpha={alpha} layers={sorted(active_layers)} "
               f"gate={gate_mode} decay_p0={decay_p0}")
    print(f"[steer-vllm] ACTIVE: {summary}")
    return True, summary


def main():
    args, vllm_argv = parse_patcher_args()
    active, summary = install_patch(args)
    if active:
        os.environ["STEERING_SUMMARY"] = summary

    print(f"[steer-vllm] launching vllm OpenAI api_server with args: {vllm_argv}")
    sys.argv = ["api_server"] + vllm_argv
    import runpy
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
