#!/usr/bin/env python3
"""
W_Q^T W_K SVD architectural baseline for Cor 6.7 decision matrix.

Computes the architectural upper bound on query-key coupling directions
from weight matrices alone (no data, no forward pass). If even the
weight-SVD directions cannot separate MMLU from MetaTool in epsilon_q,
then the ceiling is architectural and Cor 6.7 should be dropped.

Method
------
For each (layer, q_head):
    M = W_Q[L, q_head] @ W_K[L, q_head // 7]^T   (128 x 128)
    U, S, Vh = SVD(M)
    Right singular vectors (rows of Vh = columns of V) are K-directions
    ranked by query-key coupling strength.

Aggregation per KV group (Option B from task spec):
    Stack the 7 Q-head M matrices -> (7*128, 128), SVD -> shared K-dirs.

Output: B_ont tensor compatible with diag_epsilon_q_distribution.py.

Decision matrix:
    sep > +0.05  -> Ontology/data bottleneck; Cor 6.7 alive
    sep +0.02..+0.05 -> Partial signal; drop formal claim
    sep < +0.02  -> Architectural ceiling; drop Cor 6.7 entirely

CLI:
    source /home/woori/workspace_common/CDP/poc/set.env && \
    python scripts/ocq/diag_wqwk_svd_baseline.py \
        --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 \
        --rank 20 --n-mmlu 200 --n-metatool 200 --seed 42 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from eval_metatool_subtask1 import build_facet_masks, resolve_dtype  # noqa: E402
from eval_mmlu_subset import build_prompt, load_mmlu  # noqa: E402
from diag_gate_distribution import metatool_prompts  # noqa: E402


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W_Q W_K SVD baseline for Cor 6.7")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--rank", type=int, default=20,
                   help="Number of top singular directions to keep per KV head")
    p.add_argument("--metatool-dataset",
                   default="/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json")
    p.add_argument("--mmlu-config", default="all")
    p.add_argument("--n-mmlu", type=int, default=200)
    p.add_argument("--n-metatool", type=int, default=200)
    p.add_argument("--n-shot", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-ctx", type=int, default=3072)
    p.add_argument("--n-bins", type=int, default=50)
    p.add_argument("--hist-max", type=float, default=1.0001)
    p.add_argument("--out-bont",
                   default=str(Path(__file__).resolve().parents[2]
                               / "external" / "SEKA" / "seka_projections"
                               / "wqwk-svd-qwen25-7b" / "B_ont.pt"))
    p.add_argument("--out-report",
                   default=str(Path(__file__).resolve().parents[2]
                               / "reports" / "axis2_theoretical_verification"
                               / "diag_wqwk_svd_baseline_qwen25_7b.json"))
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------
# Step 1: Extract W_Q, W_K and build B_ont from weight SVD
# ---------------------------------------------------------------------

@torch.no_grad()
def build_wqwk_svd_basis(
    model,
    n_layers: int,
    n_q: int,
    n_kv: int,
    head_dim: int,
    rank: int,
) -> torch.Tensor:
    """Build B_ont (L, n_kv, d_head, rank) from W_Q^T W_K SVD.

    For each KV group, stack the coupling matrices M from all Q heads
    sharing that KV head, then take the top-r right singular vectors
    as the dominant K-directions.
    """
    kv_group = n_q // n_kv
    B_ont = torch.zeros(n_layers, n_kv, head_dim, rank, dtype=torch.float32)
    singular_values = {}  # for diagnostics

    for li in range(n_layers):
        layer = model.model.layers[li]
        W_Q = layer.self_attn.q_proj.weight.float()  # (n_q*d, hidden)
        W_K = layer.self_attn.k_proj.weight.float()  # (n_kv*d, hidden)

        for kv_h in range(n_kv):
            # Extract per-KV-head K weight: (d_head, hidden)
            W_K_h = W_K[kv_h * head_dim : (kv_h + 1) * head_dim, :]

            # Stack coupling matrices from all Q heads in this KV group
            # M_qh = W_Q[qh] @ W_K[kv_h]^T, shape (d_head, d_head)
            # Stacked: (kv_group * d_head, d_head)
            M_stack = []
            for q_offset in range(kv_group):
                qh = kv_h * kv_group + q_offset
                W_Q_h = W_Q[qh * head_dim : (qh + 1) * head_dim, :]  # (d, hidden)
                M_qh = W_Q_h @ W_K_h.T  # (d_head, d_head)
                M_stack.append(M_qh)

            M_stacked = torch.cat(M_stack, dim=0)  # (7*128, 128)

            # SVD: right singular vectors (rows of Vh) are K-directions
            # ranked by coupling strength across all Q heads
            _, S, Vh = torch.linalg.svd(M_stacked, full_matrices=False)
            # Vh shape: (min(7*128, 128), 128) = (128, 128)
            # Take top-rank rows of Vh, transpose to get columns
            B_kv = Vh[:rank, :].T  # (d_head, rank)

            # Verify orthonormality (Vh rows are orthonormal by SVD)
            B_ont[li, kv_h] = B_kv
            singular_values[f"L{li}_H{kv_h}"] = S[:rank].tolist()

    return B_ont, singular_values


# ---------------------------------------------------------------------
# Step 2: epsilon_q accumulator (simplified — single facet)
# ---------------------------------------------------------------------

class EpsilonQAccSimple:
    """Simplified ε_q accumulator for single-facet B_ont (no facet mask).

    Computes ε_q = ||B^T q||^2 / ||q||^2 per token, aggregates over
    all query heads and tokens.
    """

    BUCKETS = ("bos", "rest")

    def __init__(self, L: int, n_bins: int, hist_max: float):
        self.L = L
        self.n_bins = n_bins
        self.hist_max = hist_max
        self._s = {
            b: {
                "sum": torch.zeros(L, dtype=torch.float64),
                "sqsum": torch.zeros(L, dtype=torch.float64),
                "n_obs": torch.zeros(L, dtype=torch.float64),
                "hist": torch.zeros(L, n_bins, dtype=torch.float64),
            }
            for b in self.BUCKETS
        }

    def add(self, bucket: str, layer: int, eps: torch.Tensor):
        """eps: (N,) flat tensor of ε_q values."""
        if eps.numel() == 0:
            return
        s = self._s[bucket]
        s["sum"][layer] += eps.sum()
        s["sqsum"][layer] += (eps ** 2).sum()
        s["n_obs"][layer] += eps.numel()
        idx = torch.clamp(
            (eps / self.hist_max * self.n_bins).long(),
            min=0, max=self.n_bins - 1,
        )
        s["hist"][layer] += torch.bincount(idx, minlength=self.n_bins).double()

    def _bucket_dict(self, bucket: str) -> Dict:
        s = self._s[bucket]
        n = s["n_obs"].clamp(min=1.0)
        mean = s["sum"] / n
        var = (s["sqsum"] / n) - mean ** 2
        std = var.clamp(min=0.0).sqrt()
        n_tot = s["n_obs"].sum().clamp(min=1)
        return {
            "n_obs_per_layer": s["n_obs"].tolist(),
            "eps_mean_per_layer": mean.tolist(),
            "eps_std_per_layer": std.tolist(),
            "eps_mean_global": float(s["sum"].sum() / n_tot),
            "hist_per_layer": s["hist"].tolist(),
        }

    def to_dict(self) -> Dict:
        return {b: self._bucket_dict(b) for b in self.BUCKETS}


# ---------------------------------------------------------------------
# Step 3: q_proj hook (single-facet version, cleaner than full facet mask)
# ---------------------------------------------------------------------

def install_epsilon_q_hooks_simple(model, B_ont, n_q, n_kv, head_dim, acc):
    """Install forward hooks on q_proj to compute ε_q per token.

    Returns list of hook handles (caller must remove them).
    """
    L = B_ont.shape[0]
    kv_group = n_q // n_kv
    handles = []

    for layer_idx in range(L):
        layer = model.model.layers[layer_idx]
        q_proj = layer.self_attn.q_proj
        B_l = B_ont[layer_idx]  # (n_kv, d, r)

        def make_hook(li, B_l_local):
            def hook(module, inputs, output):
                B_sz, T, D = output.shape
                if D != n_q * head_dim:
                    return None
                Q = (output.view(B_sz, T, n_q, head_dim)
                           .permute(0, 2, 1, 3)
                           .contiguous()
                           .float())  # (B, n_q, T, d)

                B_dev = B_l_local.to(device=Q.device, dtype=torch.float32)
                # Expand per KV head to all Q heads: (n_kv, d, r) -> (n_q, d, r)
                B_q = B_dev.repeat_interleave(kv_group, dim=0)

                # Project: (B, n_q, T, r)
                coeffs = torch.einsum("bhtd,hdr->bhtr", Q, B_q)
                proj_norm_sq = (coeffs ** 2).sum(dim=-1)  # (B, n_q, T)
                q_norm_sq = (Q ** 2).sum(dim=-1) + 1e-6  # (B, n_q, T)
                eps = proj_norm_sq / q_norm_sq  # (B, n_q, T)

                # BOS bucket: position 0
                eps_bos = eps[:, :, 0].reshape(-1).double().cpu()
                acc.add("bos", li, eps_bos)

                # Rest bucket: positions 1+
                if T > 1:
                    eps_rest = eps[:, :, 1:].reshape(-1).double().cpu()
                    acc.add("rest", li, eps_rest)

                return None
            return hook

        handles.append(q_proj.register_forward_hook(make_hook(layer_idx, B_l)))

    return handles


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------

def mmlu_prompts(args) -> List[str]:
    samples, fewshot = load_mmlu(args.mmlu_config, args.n_mmlu, args.n_shot, args.seed)
    return [build_prompt(row, fewshot, args.n_shot) for row in samples]


@torch.no_grad()
def run_dataset(model, tokenizer, prompts, args, B_ont, n_q, n_kv, head_dim, L, tag):
    acc = EpsilonQAccSimple(L=L, n_bins=args.n_bins, hist_max=args.hist_max)
    handles = install_epsilon_q_hooks_simple(model, B_ont, n_q, n_kv, head_dim, acc)
    t0 = time.time()
    try:
        for i, prompt in enumerate(prompts):
            ids = tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=args.max_ctx,
            ).to(args.device)
            model(**ids)
            if args.verbose and (i % 25 == 0 or i == len(prompts) - 1):
                print(f"  [{tag}] {i+1}/{len(prompts)}  "
                      f"({time.time()-t0:.1f}s)", flush=True)
    finally:
        for h in handles:
            h.remove()

    out = acc.to_dict()
    out["tag"] = tag
    out["n_prompts"] = len(prompts)
    out["runtime_s"] = time.time() - t0
    return out


# ---------------------------------------------------------------------
# Also save B_ont in the format compatible with the full diagnostic
# ---------------------------------------------------------------------

def save_compatible_bont(B_ont: torch.Tensor, rank: int, n_layers: int,
                         n_kv: int, head_dim: int, model_name: str,
                         out_path: Path):
    """Save B_ont with r_per_pair metadata so it can be loaded by
    diag_epsilon_q_distribution.py (which expects facet_mask).
    """
    # Single facet: all columns belong to facet 0
    r_per_pair = {}
    for li in range(n_layers):
        for h in range(n_kv):
            r_per_pair[f"L{li}_H{h}"] = [rank]  # single facet, all cols

    payload = {
        "B_ont": B_ont,
        "model": model_name,
        "target_layers": list(range(n_layers)),
        "facet_order": ["wqwk_svd"],
        "n_layers": n_layers,
        "n_kv": n_kv,
        "head_dim": head_dim,
        "r_ont": rank,
        "r_per_pair": r_per_pair,
        "method": "W_Q^T @ W_K SVD (Option B: stacked 7 Q-heads per KV group)",
        "center_mode": "none (weight-only, no data)",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"[save] B_ont {tuple(B_ont.shape)} -> {out_path}", flush=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # --- Load model ---
    print(f"[load] {args.model} on {args.device}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = resolve_dtype(args.model, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map=args.device,
        attn_implementation="eager", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[load] done in {time.time()-t0:.1f}s", flush=True)

    cfg = model.config
    n_kv = cfg.num_key_value_heads
    n_q = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_q)
    n_layers = cfg.num_hidden_layers
    kv_group = n_q // n_kv
    print(f"[cfg] L={n_layers} n_q={n_q} n_kv={n_kv} kv_group={kv_group} "
          f"head_dim={head_dim}", flush=True)

    # --- Step 1: Build B_ont from weight SVD ---
    print(f"\n[step1] Computing W_Q^T @ W_K SVD per (layer, KV-head) ...", flush=True)
    t0 = time.time()
    B_ont, sv_diag = build_wqwk_svd_basis(
        model, n_layers, n_q, n_kv, head_dim, args.rank,
    )
    print(f"[step1] done in {time.time()-t0:.1f}s  B_ont shape={tuple(B_ont.shape)}", flush=True)

    # Quick diagnostic: singular value decay
    sv_example = sv_diag["L14_H0"]
    print(f"  Example L14_H0 top-5 singular values: "
          f"{[f'{s:.2f}' for s in sv_example[:5]]}", flush=True)
    sv_ratio = sv_example[args.rank - 1] / sv_example[0] if sv_example[0] > 0 else 0
    print(f"  S[{args.rank-1}]/S[0] = {sv_ratio:.4f}", flush=True)

    # --- Save B_ont ---
    out_bont = Path(args.out_bont)
    save_compatible_bont(B_ont, args.rank, n_layers, n_kv, head_dim,
                         args.model, out_bont)

    # --- Step 2: Run ε_q diagnostic ---
    print(f"\n[step2] Building prompts ...", flush=True)
    prompts_mmlu = mmlu_prompts(args)
    prompts_meta = metatool_prompts(args)
    print(f"  MMLU: {len(prompts_mmlu)} prompts, MetaTool: {len(prompts_meta)} prompts",
          flush=True)

    print(f"\n[step2] Running MMLU ε_q ...", flush=True)
    stats_mmlu = run_dataset(
        model, tok, prompts_mmlu, args, B_ont,
        n_q, n_kv, head_dim, n_layers, tag="mmlu",
    )
    mm = stats_mmlu["rest"]
    print(f"  MMLU rest: E[ε_q] = {mm['eps_mean_global']:.4f}", flush=True)

    print(f"\n[step2] Running MetaTool ε_q ...", flush=True)
    stats_meta = run_dataset(
        model, tok, prompts_meta, args, B_ont,
        n_q, n_kv, head_dim, n_layers, tag="metatool",
    )
    mt = stats_meta["rest"]
    print(f"  MetaTool rest: E[ε_q] = {mt['eps_mean_global']:.4f}", flush=True)

    # --- Summary ---
    sep = mt["eps_mean_global"] - mm["eps_mean_global"]
    ratio = mm["eps_mean_global"] / max(mt["eps_mean_global"], 1e-12)

    print(f"\n{'='*60}")
    print(f"  W_Q^T W_K SVD BASELINE (rank={args.rank})")
    print(f"{'='*60}")
    print(f"  E[ε_q] MMLU     = {mm['eps_mean_global']:.4f}")
    print(f"  E[ε_q] MetaTool = {mt['eps_mean_global']:.4f}")
    print(f"  Separation (MetaTool - MMLU) = {sep:+.4f}")
    print(f"  Ratio MMLU/MetaTool = {ratio:.3f}")
    print()

    # Per-layer profile
    print("  Per-layer E[ε_q] (rest bucket):")
    print(f"  {'Layer':>5}  {'MMLU':>8}  {'MetaTool':>8}  {'Sep':>8}")
    for li in range(n_layers):
        mm_li = mm["eps_mean_per_layer"][li]
        mt_li = mt["eps_mean_per_layer"][li]
        sep_li = mt_li - mm_li
        print(f"  {li:5d}  {mm_li:8.4f}  {mt_li:8.4f}  {sep_li:+8.4f}")

    # Decision
    if sep > 0.05:
        verdict = ("ABOVE +0.05: Weight-SVD separates! Ontology/data was the "
                   "bottleneck. Cor 6.7 alive — use weight-SVD B_ont for steering.")
    elif sep > 0.02:
        verdict = ("IN RANGE +0.02..+0.05: Partial architectural signal. "
                   "Drop Cor 6.7 formal claim, ship Thm 6.1/6.2 + Mode A/B/C.")
    else:
        verdict = ("BELOW +0.02: Architectural ceiling confirmed. "
                   "Drop Cor 6.7 entirely. The query-key coupling in Qwen2.5-7B "
                   "does not privilege any K-subspace for task discrimination.")

    print(f"\n  VERDICT: {verdict}")
    print(f"{'='*60}")

    # --- Save report ---
    report = {
        "model": args.model,
        "method": "W_Q^T @ W_K SVD (Option B: stacked Q-heads per KV group)",
        "rank": args.rank,
        "seed": args.seed,
        "n_mmlu": len(prompts_mmlu),
        "n_metatool": len(prompts_meta),
        "gqa": {"n_q": n_q, "n_kv": n_kv, "kv_group": kv_group},
        "b_ont_path": str(out_bont),
        "singular_value_examples": {
            k: sv_diag[k] for k in ["L0_H0", "L14_H0", "L27_H0"]
            if k in sv_diag
        },
        "mmlu": stats_mmlu,
        "metatool": stats_meta,
        "eps_mmlu_rest_global": mm["eps_mean_global"],
        "eps_metatool_rest_global": mt["eps_mean_global"],
        "separation_meta_minus_mmlu": sep,
        "ratio_mmlu_over_metatool": ratio,
        "verdict": verdict,
    }
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2))
    print(f"\n[save] report -> {out_report}", flush=True)


if __name__ == "__main__":
    main()
