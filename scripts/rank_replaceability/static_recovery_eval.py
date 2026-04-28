#!/usr/bin/env python3
"""static_recovery_eval.py — E3 of EXPERIMENT_PLAN_v27.

Tests Theorem 1's empirical sufficiency: does a rank-k static intervention
applied at attention output recover the full-prompt next-token distribution
when the prefix is removed?

For each query q in Q:
  full_logits[q]    := model(prefix + q).last_position_logits
  noprompt_logits[q]:= model(q).last_position_logits
  inj_k_logits[q]   := model_with_hook_k(q).last_position_logits  (prompt removed,
                       injection adds V_k V_k^T phi_mean at attention output of every (ell,h))

Reports:
  - per-k KL(full || inj_k) — main "internalization" metric
  - per-k top-1 agreement (full vs inj_k)
  - L2 residual at attention output per (layer, head) — direct Theorem 1 metric
  - baseline KL(full || noprompt) for reference

Inputs:
  --e1-json  path to E1 result JSON (must have associated .npz with eigvecs/phi_mean)
  --k-list   comma-separated k values to test (default: 0,1,2,4,8,16,32)
            k=0 means no-prompt baseline (no injection)

Output: JSON with all metrics per k.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse data loaders + prompt construction from E1 script
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from measure_phi_rank import (  # type: ignore
    DEFAULT_TOOL_SYSTEM_PROMPT,
    build_messages,
    detect_model_family,
    find_user_block_start,
    load_metatool_st4,
    load_tau2,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", default="metatool_st4")
    p.add_argument("--metatool-path", default="/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json")
    p.add_argument("--e1-json", required=True, help="E1 result JSON path")
    p.add_argument("--max-samples", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--k-list", default="0,1,2,4,8,16,32")
    p.add_argument("--out", required=True)
    return p.parse_args()


# =============================================================================
# Static intervention via o_proj pre-hook
# =============================================================================

class StaticIntervention:
    """Adds a per-(layer, head) static vector to the input of o_proj at the
    last query position. The o_proj input has shape (B, T, H_q*d_h) — i.e.,
    the concatenated per-head attention outputs.
    """

    def __init__(self, model, family: str, injection: torch.Tensor):
        """injection: (L, H_q, d_h) — vector to add at last position per (ell, h)."""
        self.model = model
        self.family = family
        self.injection = injection  # on device, dtype matched to model
        self.handles = []
        self.enabled = False  # set True after install

    def _hook(self, layer_idx: int):
        def fn(module, inputs):
            if not self.enabled:
                return inputs
            x = inputs[0]  # (B, T, H_q*d_h)
            B, T, D = x.shape
            d_h = self.injection.shape[2]
            H_q = self.injection.shape[1]
            assert D == H_q * d_h, f"o_proj input D={D} != H_q*d_h={H_q*d_h}"
            # Reshape, add at last position, reshape back
            x_reshaped = x.view(B, T, H_q, d_h).clone()
            inj_layer = self.injection[layer_idx].to(x.dtype)  # (H_q, d_h)
            x_reshaped[:, -1, :, :] = x_reshaped[:, -1, :, :] + inj_layer
            return (x_reshaped.view(B, T, D),) + tuple(inputs[1:])
        return fn

    def install(self):
        """Hook the o_proj of every attention layer with a forward_pre_hook."""
        layers = self.model.model.layers
        for i, layer in enumerate(layers):
            o_proj = layer.self_attn.o_proj
            self.handles.append(o_proj.register_forward_pre_hook(self._hook(i)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def build_injection_for_k(eigvecs: np.ndarray, phi_mean: np.ndarray, k: int) -> np.ndarray:
    """Returns injection of shape (L, H_q, d_h) for rank-k truncation.

    inj[ell, h] = V_k[ell, h] @ V_k[ell, h]^T @ phi_mean[ell, h]
                = sum_{i<k} <v_i, phi_mean> * v_i
    """
    L, H_q, K_save, d_h = eigvecs.shape
    k = min(k, K_save)
    if k <= 0:
        return np.zeros((L, H_q, d_h), dtype=np.float32)
    inj = np.zeros((L, H_q, d_h), dtype=np.float32)
    for ell in range(L):
        for h in range(H_q):
            V = eigvecs[ell, h, :k]  # (k, d_h)
            mean = phi_mean[ell, h]   # (d_h,)
            # Project mean onto V's row span: V^T @ V @ mean
            inj[ell, h] = V.T @ (V @ mean)
    return inj


# =============================================================================
# Per-query evaluation
# =============================================================================

@torch.no_grad()
def get_last_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Returns logits at the last position only. Shape (B, vocab)."""
    out = model(input_ids=input_ids, use_cache=False, return_dict=True)
    return out.logits[:, -1, :]


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load E1 results (eigvecs, phi_mean)
    e1_json = Path(args.e1_json)
    with open(e1_json) as f:
        e1 = json.load(f)
    npz_name = e1.get("npz_path")
    if npz_name is None:
        print("ERROR: E1 JSON does not have 'npz_path' field. Re-run measure_phi_rank.py with the version that saves npz.", file=sys.stderr)
        return 2
    npz_path = e1_json.parent / npz_name
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found", file=sys.stderr)
        return 2
    npz = np.load(npz_path)
    eigvecs = npz["eigvecs"]   # (L, H_q, K_save, d_h)
    phi_mean = npz["phi_mean"]  # (L, H_q, d_h)
    L = eigvecs.shape[0]
    H_q = eigvecs.shape[1]
    d_h = eigvecs.shape[3]
    K_save = eigvecs.shape[2]
    print(f"[e1] L={L} H_q={H_q} d_h={d_h} K_save={K_save}", flush=True)

    # Load data
    if args.task == "metatool_st4":
        items = load_metatool_st4(args.metatool_path, args.max_samples)
    elif args.task.startswith("tau2_"):
        items = load_tau2(args.task.split("_", 1)[1], args.max_samples)
    else:
        raise ValueError(args.task)
    print(f"[data] N={len(items)}", flush=True)

    # Load model
    print(f"[model] loading {args.model}", flush=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device,
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    family = detect_model_family(args.model)

    k_list = sorted(set(int(x) for x in args.k_list.split(",")))

    # Pre-compute all injections
    injections_np = {k: build_injection_for_k(eigvecs, phi_mean, k) for k in k_list}
    injections_t = {
        k: torch.from_numpy(v).to(args.device, dtype=dtype) for k, v in injections_np.items()
    }
    print(f"[k_list] {k_list}", flush=True)

    # Setup intervention (we'll swap injection tensor per k)
    inj_state = StaticIntervention(model, family, injections_t[max(k_list)])
    inj_state.install()

    # =====================================================================
    # Loop over queries: collect (full, noprompt, inj_k for each k) logits
    # =====================================================================

    N = len(items)
    vocab = model.config.vocab_size
    metrics = {f"k_{k}": {"kl_full_inj": [], "agree_full_inj": [], "abs_logit_resid": []} for k in k_list}
    metrics["baseline"] = {"kl_full_noprompt": [], "agree_full_noprompt": []}

    t0 = time.time()
    for i, item in enumerate(items):
        # ---- full-prompt forward ----
        msgs_full = build_messages(item, DEFAULT_TOOL_SYSTEM_PROMPT, prefix_mode="real")
        try:
            txt_full = tokenizer.apply_chat_template(msgs_full, add_generation_prompt=True, tokenize=False)
        except Exception as e:
            print(f"[skip {i}] {e}", flush=True)
            continue
        ids_full = tokenizer(txt_full, return_tensors="pt").input_ids.to(args.device)
        inj_state.enabled = False
        logits_full = get_last_logits(model, ids_full)[0]  # (vocab,)

        # ---- no-prompt forward (user-only message) ----
        msgs_user = [{"role": "user", "content": item["query"]}]
        try:
            txt_user = tokenizer.apply_chat_template(msgs_user, add_generation_prompt=True, tokenize=False)
        except Exception:
            continue
        ids_user = tokenizer(txt_user, return_tensors="pt").input_ids.to(args.device)
        inj_state.enabled = False
        logits_noprompt = get_last_logits(model, ids_user)[0]

        # Baseline metrics: full vs noprompt
        kl_np = F.kl_div(
            F.log_softmax(logits_noprompt.float(), dim=-1),
            F.softmax(logits_full.float(), dim=-1),
            reduction="sum",
        ).item()
        ag_np = int(logits_full.argmax().item() == logits_noprompt.argmax().item())
        metrics["baseline"]["kl_full_noprompt"].append(kl_np)
        metrics["baseline"]["agree_full_noprompt"].append(ag_np)

        # ---- intervention forward, per k ----
        for k in k_list:
            if k == 0:
                # k=0 == no injection — use noprompt logits
                logits_inj = logits_noprompt
            else:
                inj_state.injection = injections_t[k]
                inj_state.enabled = True
                logits_inj = get_last_logits(model, ids_user)[0]
                inj_state.enabled = False
            kl = F.kl_div(
                F.log_softmax(logits_inj.float(), dim=-1),
                F.softmax(logits_full.float(), dim=-1),
                reduction="sum",
            ).item()
            ag = int(logits_full.argmax().item() == logits_inj.argmax().item())
            resid = (logits_full.float() - logits_inj.float()).abs().mean().item()
            metrics[f"k_{k}"]["kl_full_inj"].append(kl)
            metrics[f"k_{k}"]["agree_full_inj"].append(ag)
            metrics[f"k_{k}"]["abs_logit_resid"].append(resid)

        if (i + 1) % 8 == 0 or i == N - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            print(
                f"[{i+1}/{N}] elapsed={elapsed:.1f}s rate={rate:.2f}/s",
                flush=True,
            )

    inj_state.remove()

    # Reduce
    summary = {"baseline": {}}
    bl = metrics["baseline"]
    summary["baseline"]["kl_full_noprompt_mean"] = float(np.mean(bl["kl_full_noprompt"]))
    summary["baseline"]["agree_full_noprompt"] = float(np.mean(bl["agree_full_noprompt"]))
    for k in k_list:
        m = metrics[f"k_{k}"]
        summary[f"k_{k}"] = {
            "kl_mean": float(np.mean(m["kl_full_inj"])),
            "kl_median": float(np.median(m["kl_full_inj"])),
            "agree": float(np.mean(m["agree_full_inj"])),
            "logit_resid_mean": float(np.mean(m["abs_logit_resid"])),
        }

    out = {
        "model": args.model,
        "task": args.task,
        "n_samples": N,
        "k_list": k_list,
        "e1_json": str(e1_json),
        "summary": summary,
        "details": metrics,
        "wall_seconds": time.time() - t0,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] saved -> {out_path}", flush=True)
    print()
    print(f"Baseline (no prompt): KL={summary['baseline']['kl_full_noprompt_mean']:.4f} top1={summary['baseline']['agree_full_noprompt']:.3f}")
    print(f"{'k':>4}  {'KL':>8}  {'top1':>6}  {'logit_resid':>12}")
    for k in k_list:
        s = summary[f"k_{k}"]
        print(f"{k:>4}  {s['kl_mean']:>8.4f}  {s['agree']:>6.3f}  {s['logit_resid_mean']:>12.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
