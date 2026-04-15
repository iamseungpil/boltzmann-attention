#!/usr/bin/env python3
"""measure_epsilon_q_plan_predictor.py — Thm 6.20 plan-time predictor smoke.

For each Subtask4 query (multi-tool target), measure per-step ε_q at every
generated token's q vector (layer L=13, all q heads). Aggregate:
  - min_t ε_q (across generation positions)
  - mean_t ε_q
  - ε_q at first <tool_call> emission position
  - ε_q at second <tool_call> emission position (if any)

Then correlate with task success (F1 ≥ 0.5 → "success", binary). Compute
AUROC of (min ε_q) and (mean ε_q) as predictors.

This is a cheap proxy for the τ²-bench multi-turn evaluation: instead of
multi-turn agent execution, we observe the model's *internal* per-step
stability during single-turn multi-tool generation.

Usage:
  python scripts/ocq/measure_epsilon_q_plan_predictor.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt \
    --layer 13 \
    --max-samples 100 \
    --out reports/thm620_smoke/eps_q_predictor.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ocq"))

from eval_metatool_subtask4 import (  # noqa: E402
    build_fc_prompt, extract_tool_names, compute_metrics,
)
from eval_metatool_subtask1 import parse_candidates  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--b-ont", required=True)
    p.add_argument("--layer", type=int, default=13,
                   help="Layer to measure ε_q at (Qwen2.5-7B mid layer).")
    p.add_argument("--dataset",
                   default="/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json")
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {args.model}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).eval()
    print(f"[load] {time.time() - t0:.1f}s", flush=True)

    # Load B_ont and build per-q-head broadcast (n_q from per-kv n_kv)
    bdict = torch.load(args.b_ont, map_location="cpu", weights_only=False)
    B_ont = bdict["B_ont"] if isinstance(bdict, dict) else bdict  # (L, H_kv, d, r)
    L_total, H_kv, d, r = B_ont.shape
    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_q)
    g = n_q // n_kv
    B_layer = B_ont[args.layer].clone()  # (H_kv, d, r)
    B_q = B_layer.unsqueeze(1).expand(-1, g, -1, -1).reshape(n_q, d, r)
    B_q_dev = B_q.to(args.device, dtype=torch.float32)
    print(f"[bont] layer {args.layer}: B shape {(H_kv, d, r)} → q-broadcast {tuple(B_q.shape)}",
          flush=True)

    # Hook on q_proj layer L=args.layer to capture q at every generation step
    captured_q_per_step = []  # list of (T, n_q, d) tensors
    def q_hook(_m, _inp, output):
        # output: (B, T, n_q*d)
        B, T, D = output.shape
        Q = output.view(B, T, n_q, head_dim).float()
        captured_q_per_step.append(Q[0].detach())  # (T, n_q, d)
    h = model.model.layers[args.layer].self_attn.q_proj.register_forward_hook(q_hook)

    data = json.load(open(args.dataset))
    if args.max_samples > 0:
        data = data[:args.max_samples]
    print(f"[data] N={len(data)}", flush=True)

    results = []
    t_eval = time.time()
    for i, entry in enumerate(data):
        captured_q_per_step.clear()
        action_prompt = entry["action_prompt"]
        gt = entry["tool"] if isinstance(entry["tool"], list) else [entry["tool"]]
        cands = parse_candidates(action_prompt)
        fc_prompt = build_fc_prompt(tok, action_prompt, cands)
        ids = tok(fc_prompt, return_tensors="pt").to(args.device)
        prompt_len = ids["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tok.eos_token_id,
            )
        new_ids = out[0, prompt_len:]
        gen_text = tok.decode(new_ids, skip_special_tokens=True)
        pred = extract_tool_names(gen_text, cands)
        metrics = compute_metrics(pred, gt)

        # ε_q per generation step: only count tokens at gen positions (post-prompt)
        # captured_q_per_step has one entry per forward call. With KV cache, each
        # decoding step calls model with T=1 (just last token). Batch first call
        # has T=prompt_len (full prompt forward).
        eps_q_traj = []
        for q_chunk in captured_q_per_step:
            T_chunk = q_chunk.shape[0]
            # eps per token, per head, then average over heads
            q_norm_sq = (q_chunk ** 2).sum(dim=-1) + 1e-8  # (T, n_q)
            coeffs = torch.einsum("thd,hdr->thr", q_chunk, B_q_dev)
            proj_norm_sq = (coeffs ** 2).sum(dim=-1)  # (T, n_q)
            eps = (proj_norm_sq / q_norm_sq).mean(dim=-1)  # (T,) avg over heads
            eps_q_traj.extend(eps.cpu().tolist())

        # The first captured entry is the prompt forward (T=prompt_len);
        # subsequent are decoding steps (T=1). We restrict to decoding steps.
        gen_eps_q = eps_q_traj[prompt_len:] if len(eps_q_traj) > prompt_len else eps_q_traj
        if len(gen_eps_q) == 0:
            gen_eps_q = [float("nan")]

        eps_arr = np.array([e for e in gen_eps_q if not np.isnan(e)])
        eps_summary = {
            "min_eps_q": float(eps_arr.min()) if len(eps_arr) else None,
            "mean_eps_q": float(eps_arr.mean()) if len(eps_arr) else None,
            "max_eps_q": float(eps_arr.max()) if len(eps_arr) else None,
            "median_eps_q": float(np.median(eps_arr)) if len(eps_arr) else None,
            "n_gen_steps": len(eps_arr),
        }

        results.append({
            "index": entry.get("index", i),
            "gt": gt, "pred": pred,
            "F1": metrics["F1"], "Exact": metrics["Exact"],
            "recall": metrics["recall"], "precision": metrics["precision"],
            "success_F1_ge_05": int(metrics["F1"] >= 0.5),
            "exact_success": int(metrics["Exact"] >= 0.5),
            **eps_summary,
        })
        if i % 10 == 0:
            print(f"  [{i}/{len(data)}] F1={metrics['F1']:.3f} "
                  f"min_eps={eps_summary['min_eps_q']:.4f} "
                  f"mean_eps={eps_summary['mean_eps_q']:.4f}", flush=True)

    h.remove()
    print(f"[eval] {time.time() - t_eval:.1f}s", flush=True)

    # AUROC computation
    valid = [r for r in results if r["min_eps_q"] is not None]
    y = np.array([r["success_F1_ge_05"] for r in valid])
    y_exact = np.array([r["exact_success"] for r in valid])
    feats = {
        "min_eps_q": np.array([r["min_eps_q"] for r in valid]),
        "mean_eps_q": np.array([r["mean_eps_q"] for r in valid]),
        "median_eps_q": np.array([r["median_eps_q"] for r in valid]),
    }
    try:
        from sklearn.metrics import roc_auc_score
        auroc = {k: float(roc_auc_score(y, v)) for k, v in feats.items()}
        auroc_exact = {k: float(roc_auc_score(y_exact, v)) for k, v in feats.items()}
    except Exception as e:
        print(f"[auroc] sklearn missing or error: {e}", flush=True)
        auroc = auroc_exact = None

    summary = {
        "n_samples": len(results),
        "layer": args.layer,
        "auroc_F1_success": auroc,
        "auroc_Exact_success": auroc_exact,
        "base_F1_mean": float(np.mean([r["F1"] for r in results])),
        "base_Exact_mean": float(np.mean([r["Exact"] for r in results])),
    }

    blob = {"summary": summary, "per_sample": results}
    with open(out_path, "w") as f:
        json.dump(blob, f, indent=2)

    print("\n=== SUMMARY ===", flush=True)
    print(f"  N samples: {summary['n_samples']}", flush=True)
    print(f"  base F1 mean: {summary['base_F1_mean']:.3f}", flush=True)
    print(f"  base Exact mean: {summary['base_Exact_mean']:.3f}", flush=True)
    print(f"  AUROC (F1 ≥ 0.5 success):", flush=True)
    if auroc:
        for k, v in auroc.items():
            print(f"    {k:15s}: {v:.4f}", flush=True)
    print(f"  AUROC (Exact success):", flush=True)
    if auroc_exact:
        for k, v in auroc_exact.items():
            print(f"    {k:15s}: {v:.4f}", flush=True)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
