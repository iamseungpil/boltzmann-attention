#!/usr/bin/env python3
"""measure_cor_6_9_nrank.py — E4: Operator-level ε-numerical rank verification of Cor 6.9.

Cor 6.9 formal claim:
  nrank_ε(P_fg(q, k_t)) = R  (for all k_t with g_f(k_t) >= ε)
  nrank_ε(P_ada(q)) <= r       (when β(q) < ε, single-expert dominant)

Our facet-gated operator:
  P_fg(q, k_t) = Σ_f g_f(k_t) · B_f B_f^⊤
  where B_f is r_f columns of B ∈ R^{d×R}, r_f = R/F, B_f^⊤ B_{f'} = δ_{ff'} I.

AdaSEKA-style operator (synthetic for comparison):
  P_ada(q) = Σ_m α_m(q) · U_m U_m^⊤
  where U_m are pairwise-orthogonal rank-r experts, α_m = softmax(logits / T),
  with max-normalization α_m / max_m α_m.

For each query sample:
  1. Compute actual B_f for Qwen/Llama/Mistral from B_ont tensor.
  2. Synthesize α_m(q) with controllable β(q) (temperature T).
  3. Form P_fg and P_ada, compute SVD.
  4. Report ε-numerical rank at ε ∈ {0.1, 0.2} across (layer, head, sample) cells.

CPU-bound, no GPU needed. Runs in ~5-10 minutes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--b-ont", required=True,
                   help="B_ont tensor path (shape L, H, d, r_ont).")
    p.add_argument("--n-facets", type=int, default=4,
                   help="Number of facets F. R = r_ont splits into F blocks.")
    p.add_argument("--n-queries", type=int, default=500,
                   help="Synthetic query count for AdaSEKA sampling.")
    p.add_argument("--ada-m", type=int, default=4,
                   help="AdaSEKA expert count M.")
    p.add_argument("--epsilons", type=float, nargs="+", default=[0.1, 0.2])
    p.add_argument("--ada-temperatures", type=float, nargs="+",
                   default=[0.1, 0.3, 0.5, 1.0])
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def nrank_eps(M: np.ndarray, eps: float) -> int:
    """ε-numerical rank: count of singular values >= eps * max_sv."""
    svs = np.linalg.svd(M, compute_uv=False)
    if len(svs) == 0 or svs[0] == 0:
        return 0
    return int(np.sum(svs >= eps * svs[0]))


def compute_facet_operator(B_fh: np.ndarray, g_vec: np.ndarray,
                           r_per_facet: np.ndarray) -> np.ndarray:
    """P_fg = Σ_f g_f · B_f B_f^⊤, with B = B_fh ∈ R^{d×r_ont}, g_vec ∈ R^F.

    r_per_facet: array of length F summing to r_ont, specifying column ranges.
    Assumes B_fh columns are orthogonal across facets (Gram-Schmidt residualized).
    """
    d, r_ont = B_fh.shape
    P = np.zeros((d, d), dtype=np.float64)
    offset = 0
    for f, r_f in enumerate(r_per_facet):
        B_f = B_fh[:, offset:offset + r_f]
        P += g_vec[f] * (B_f @ B_f.T)
        offset += r_f
    return P


def synth_adaseka_operator(U_list: List[np.ndarray], alpha: np.ndarray) -> np.ndarray:
    """P_ada = Σ_m α_m · U_m U_m^⊤."""
    d = U_list[0].shape[0]
    P = np.zeros((d, d), dtype=np.float64)
    for U_m, a_m in zip(U_list, alpha):
        P += a_m * (U_m @ U_m.T)
    return P


def sample_adaseka_alphas(M: int, T: float, rng: np.random.Generator,
                          n_samples: int) -> np.ndarray:
    """Sample AdaSEKA routing weights: logits → softmax/T → max-normalized.

    Output shape: (n_samples, M).
    At low T: sharp (β small); at high T: flat (β near 1).
    """
    logits = rng.standard_normal(size=(n_samples, M))
    weights = np.exp(logits / T)
    weights /= weights.sum(axis=1, keepdims=True)
    # max-normalize to [0, 1] with max=1
    weights /= weights.max(axis=1, keepdims=True)
    return weights


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # --- Load B_ont ---
    print(f"[load] B_ont from {args.b_ont}", flush=True)
    payload = torch.load(args.b_ont, map_location="cpu", weights_only=False)
    B_ont = payload["B_ont"] if isinstance(payload, dict) else payload
    # If payload dict has r_per_pair (per facet ranks), use that; else equal split
    B_ont_np = B_ont.numpy() if isinstance(B_ont, torch.Tensor) else B_ont
    L, H, d, r_ont = B_ont_np.shape
    print(f"[info] B_ont shape: L={L} H={H} d={d} r_ont={r_ont}", flush=True)

    # Determine facet partition
    if isinstance(payload, dict) and "r_per_pair" in payload:
        # Per (L, H) facet ranks — use layer 0 head 0 as representative, or average
        r_per_facet_default = None
    else:
        # Equal split
        r_per_facet_default = np.full(args.n_facets, r_ont // args.n_facets, dtype=int)
        remainder = r_ont - r_per_facet_default.sum()
        r_per_facet_default[:remainder] += 1

    print(f"[info] facet partition (equal): {r_per_facet_default}", flush=True)

    # --- Facet operator nrank per (L, H) ---
    print("\n[1/3] Computing P_fg nrank per (L, H)...", flush=True)
    fg_nranks = {eps: [] for eps in args.epsilons}
    all_nonzero_facet_heads = 0
    for li in range(L):
        for hi in range(H):
            B_fh = B_ont_np[li, hi]  # (d, r_ont)
            if np.linalg.norm(B_fh) < 1e-8:
                continue
            all_nonzero_facet_heads += 1
            # Assume all g_f = 1 (maximal facet activation, Cor 6.9 hypothesis)
            g_max = np.ones(args.n_facets)
            P_fg = compute_facet_operator(B_fh, g_max, r_per_facet_default)
            for eps in args.epsilons:
                fg_nranks[eps].append(nrank_eps(P_fg, eps))

    fg_summary = {}
    for eps, vals in fg_nranks.items():
        arr = np.array(vals)
        fg_summary[str(eps)] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p05": float(np.quantile(arr, 0.05)),
            "p95": float(np.quantile(arr, 0.95)),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "n_cells": int(len(arr)),
        }
    print(f"[info] facet nrank (cells={all_nonzero_facet_heads}):")
    for eps, s in fg_summary.items():
        print(f"  ε={eps}: mean={s['mean']:.2f} median={s['median']} [p05={s['p05']}, p95={s['p95']}] (target: R={r_ont})")

    # --- Synth AdaSEKA ---
    print(f"\n[2/3] Computing P_ada nrank under various temperatures...", flush=True)
    M = args.ada_m
    r_expert = r_ont // M  # match facet rank
    ada_summary = {}

    # Build M orthogonal rank-r_expert experts in R^d
    # Simpler: sample random orthonormal d×(M·r_expert), split into blocks
    random_big = rng.standard_normal(size=(d, M * r_expert))
    Q_big, _ = np.linalg.qr(random_big)
    U_list = [Q_big[:, m * r_expert:(m + 1) * r_expert] for m in range(M)]

    for T in args.ada_temperatures:
        alphas = sample_adaseka_alphas(M, T, rng, args.n_queries)
        beta_q = np.partition(alphas, -2, axis=1)[:, -2]  # 2nd-largest α per query
        ada_nranks = {eps: [] for eps in args.epsilons}
        for i in range(args.n_queries):
            alpha = alphas[i]
            P_ada = synth_adaseka_operator(U_list, alpha)
            for eps in args.epsilons:
                ada_nranks[eps].append(nrank_eps(P_ada, eps))

        ada_summary[str(T)] = {
            "beta_q_mean": float(beta_q.mean()),
            "beta_q_p95": float(np.quantile(beta_q, 0.95)),
            **{
                str(eps): {
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "max": int(np.max(vals)),
                    "p95": float(np.quantile(vals, 0.95)),
                }
                for eps, vals in ada_nranks.items()
            },
        }
        print(f"  T={T}: β_q_mean={beta_q.mean():.3f}")
        for eps, vals in ada_nranks.items():
            a = np.array(vals)
            print(f"    ε={eps}: nrank mean={a.mean():.2f} median={np.median(a)} max={a.max()} (theoretical cap: ~{r_expert} when β<ε, up to {M * r_expert} when β≥ε)")

    # --- Cor 6.9 verification verdict ---
    print(f"\n[3/3] Cor 6.9 verification verdict:")
    r_ours_mean = {eps: fg_summary[str(eps)]["mean"] for eps in args.epsilons}
    print(f"  Ours (facet-gated): mean nrank = {r_ours_mean} across all non-zero (L,H)")
    for T, stats in ada_summary.items():
        for eps in args.epsilons:
            print(f"  AdaSEKA (T={T}, β={stats['beta_q_mean']:.3f}): ε={eps} nrank mean = {stats[str(eps)]['mean']:.2f}")

    print(f"\n[predict] Cor 6.9:")
    print(f"  Ours should attain ~R={r_ont} regardless of ε.")
    print(f"  AdaSEKA should cap at r_expert={r_expert} for low-T (β small), up to {M * r_expert} for high-T.")

    # --- Write output ---
    out_payload = {
        "b_ont_path": args.b_ont,
        "shape": [L, H, d, r_ont],
        "n_facets": args.n_facets,
        "r_per_facet": r_per_facet_default.tolist() if r_per_facet_default is not None else None,
        "ada_m": M,
        "r_expert": r_expert,
        "n_queries": args.n_queries,
        "epsilons": args.epsilons,
        "ada_temperatures": args.ada_temperatures,
        "fg_summary": fg_summary,
        "ada_summary": ada_summary,
        "n_nonzero_facet_heads": all_nonzero_facet_heads,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out_payload, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
