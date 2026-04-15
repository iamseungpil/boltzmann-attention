#!/usr/bin/env python3
"""
Theorem-faithful Step A diagnostic: measure ε_q = ||B^⊤q||²/||q||² on MMLU vs MetaTool.

Theory anchor (MANDATORY pre-read, per session_start_theorem_read_first.md)
---------------------------------------------------------------------------
math/paper/lie_group/COROLLARY_6_7_FACET_PHASE_CLOSURE.md lines 51-75:

    Cor 6.7 (Exact Phase-Closure). For the facet-gated K-bias operator:
    (i)  e_t ∈ Range(B) for every t (automatic from operator definition).
    (ii) If q ⊥ Range(B), then q·e_t = 0 for every t, hence qaMSE(q;E) = 0.

The five-line proof (lines 67-71) uses ONLY `B_f^⊤ q = 0`. It never
touches g_f(k_t). Phase-closure is therefore a **query-side** property,
and the testable quantity is

    ε_q := ||B^⊤ q_t||² / ||q_t||²   ∈ [0, 1]

per token, per (layer, head), where B is the same per-(L, kv_group) facet
basis that install_facet_gated_hooks uses on K.

If MMLU ε_q ≈ 0 and MetaTool ε_q ≫ 0, Cor 6.7 holds as stated: MMLU
queries see the fourth-order remainder only, MetaTool queries get
first-order steering.

If MMLU ε_q is ≈ MetaTool ε_q, the hypothesis is false on this B_ont
and we must pivot (see cor67_root_cause_resume_2026_04_10.md UPDATED
PLAN Step C).

GQA note
--------
Qwen2.5-7B: n_q=28 query heads, n_kv=4 key/value heads, kv_group_size=7.
B_ont is stored per-(L, kv_group). Each query head `h` maps to kv_group
`h // kv_group_size`. We project each q_head's token vectors onto that
group's B_ont, then aggregate. This matches how the facet-gated hook
broadcasts B_ont across the GQA group on the K side.

CLI
---
    source /home/woori/workspace_common/CDP/poc/set.env && \\
    python scripts/ocq/diag_epsilon_q_distribution.py \\
        --model Qwen/Qwen2.5-7B --device cuda:0 \\
        --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt \\
        --n-mmlu 200 --n-metatool 200 --seed 42 --verbose \\
        --out reports/axis2_theoretical_verification/diag_epsilon_q_qwen25_7b.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from eval_metatool_subtask1 import build_facet_masks, resolve_dtype  # noqa: E402
from eval_mmlu_subset import build_prompt, load_mmlu  # noqa: E402
from diag_gate_distribution import metatool_prompts  # noqa: E402  reuse loader


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--b-ont", required=True)
    p.add_argument(
        "--metatool-dataset",
        default="/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json",
    )
    p.add_argument("--mmlu-config", default="all")
    p.add_argument("--n-mmlu", type=int, default=200)
    p.add_argument("--n-metatool", type=int, default=200)
    p.add_argument("--n-shot", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-ctx", type=int, default=3072)
    p.add_argument("--n-bins", type=int, default=50)
    p.add_argument("--hist-max", type=float, default=1.0001,
                   help="ε_q ∈ [0, 1] by construction (B^⊤q is a "
                        "projection, so ||B^⊤q|| ≤ ||q||). Set slightly "
                        ">1 to catch floating-point overshoots.")
    p.add_argument("--out", required=True)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------
# Accumulator (mirrors GateAcc but with query heads and ε_q semantics)
# ---------------------------------------------------------------------

class EpsilonQAcc:
    """Per-layer per-facet ε_q stats, split by bos vs rest token bucket.

    Dimensions:
        L         — number of decoder layers
        n_facets  — facet count (same as B_ont columns split by facet_mask)

    Each bucket keeps:
        eps_per_facet_sum/sqsum  — running sums over all (B, n_q, T) obs
        eps_total_sum/sqsum      — Σ_f ε_{q,f} aggregated the same way
        n_obs                    — count of (B, n_q, T) observations
        hist                     — histogram of Σ_f ε_{q,f} per obs

    Counters are CPU float64 to avoid precision drift.
    """

    BUCKETS = ("bos", "rest")

    def __init__(self, L: int, n_facets: int, n_bins: int, hist_max: float):
        self.L = L
        self.n_facets = n_facets
        self.n_bins = n_bins
        self.hist_max = hist_max
        self._s = {
            b: {
                "eps_sum": torch.zeros(L, n_facets, dtype=torch.float64),
                "eps_sqsum": torch.zeros(L, n_facets, dtype=torch.float64),
                "n_obs": torch.zeros(L, dtype=torch.float64),
                "total_sum": torch.zeros(L, dtype=torch.float64),
                "total_sqsum": torch.zeros(L, dtype=torch.float64),
                "hist": torch.zeros(L, n_bins, dtype=torch.float64),
            }
            for b in self.BUCKETS
        }

    def add_bucket(
        self, bucket: str, layer: int,
        eps_per_f: torch.Tensor,   # (n_facets, N)  CPU float64
        eps_total: torch.Tensor,   # (N,)           CPU float64
    ):
        if eps_total.numel() == 0:
            return
        s = self._s[bucket]
        s["eps_sum"][layer] += eps_per_f.sum(dim=1)
        s["eps_sqsum"][layer] += (eps_per_f ** 2).sum(dim=1)
        s["n_obs"][layer] += eps_total.numel()
        s["total_sum"][layer] += eps_total.sum()
        s["total_sqsum"][layer] += (eps_total ** 2).sum()
        idx = torch.clamp(
            (eps_total / self.hist_max * self.n_bins).long(),
            min=0, max=self.n_bins - 1,
        )
        s["hist"][layer] += torch.bincount(idx, minlength=self.n_bins).double()

    def _bucket_dict(self, bucket: str) -> Dict:
        s = self._s[bucket]
        n = s["n_obs"].clamp(min=1.0)
        mean = s["eps_sum"] / n.unsqueeze(1)
        var = (s["eps_sqsum"] / n.unsqueeze(1)) - mean ** 2
        std = var.clamp(min=0.0).sqrt()
        tot_mean = s["total_sum"] / n
        tot_var = (s["total_sqsum"] / n) - tot_mean ** 2
        tot_std = tot_var.clamp(min=0.0).sqrt()
        n_tot = s["n_obs"].sum().clamp(min=1)
        return {
            "n_obs_per_layer": s["n_obs"].tolist(),
            "eps_f_mean_per_layer": mean.tolist(),
            "eps_f_std_per_layer": std.tolist(),
            "eps_total_mean_per_layer": tot_mean.tolist(),
            "eps_total_std_per_layer": tot_std.tolist(),
            "hist_per_layer": s["hist"].tolist(),
            "eps_f_mean_global": (s["eps_sum"].sum(dim=0) / n_tot).tolist(),
            "eps_total_mean_global": float(s["total_sum"].sum() / n_tot),
        }

    def to_dict(self) -> Dict:
        return {
            "L": self.L,
            "n_facets": self.n_facets,
            "n_bins": self.n_bins,
            "hist_max": self.hist_max,
            "buckets": {b: self._bucket_dict(b) for b in self.BUCKETS},
        }


# ---------------------------------------------------------------------
# Read-only q_proj hook
# ---------------------------------------------------------------------

@contextmanager
def install_epsilon_q_hooks(
    model,
    B_ont: torch.Tensor,       # (L, n_kv, d, r)
    facet_mask: torch.Tensor,  # (L, n_kv, F, r)
    n_q: int,
    n_kv: int,
    head_dim: int,
    acc: EpsilonQAcc,
    eps_norm: float = 1e-6,
):
    """Hook `layer.self_attn.q_proj` forward. Computes per-head per-token
    ε_{q,f} := ||B_f^⊤ q_{l,h,t}||² / (||q_{l,h,t}||² + eps_norm). Does NOT
    modify q_proj output (returns None).

    GQA: each q_head h is mapped to kv_group `h // kv_group_size`, and
    the corresponding B_ont[l, kv_group] is used. The resulting ε_q is
    aggregated across all query heads of the layer.
    """
    handles = []
    L, _, d, r = B_ont.shape
    n_facets = facet_mask.shape[2]
    kv_group = n_q // n_kv
    assert n_q == n_kv * kv_group, f"n_q {n_q} not divisible by n_kv {n_kv}"
    assert d == head_dim, f"B_ont d {d} != head_dim {head_dim}"

    try:
        for layer_idx in range(L):
            layer = model.model.layers[layer_idx]
            q_proj = layer.self_attn.q_proj
            B_l = B_ont[layer_idx]        # (n_kv, d, r)
            M_l = facet_mask[layer_idx]   # (n_kv, F, r)

            def make_hook(li, B_l_local, M_l_local):
                def hook(module, inputs, output):
                    B_sz, T, D = output.shape
                    if D != n_q * head_dim:
                        return None
                    # (B, T, n_q*d) -> (B, n_q, T, d)
                    Q = (output.view(B_sz, T, n_q, head_dim)
                               .permute(0, 2, 1, 3).contiguous())
                    Q_f = Q.float()
                    # Broadcast per-kv-group basis to n_q heads.
                    # B_l_local: (n_kv, d, r), repeat_interleave along dim 0
                    # → (n_q, d, r)
                    B_dev = B_l_local.to(device=Q.device, dtype=torch.float32)
                    B_q = B_dev.repeat_interleave(kv_group, dim=0)  # (n_q, d, r)
                    M_dev = M_l_local.to(device=Q.device, dtype=torch.float32)
                    M_q = M_dev.repeat_interleave(kv_group, dim=0)  # (n_q, F, r)

                    # Project Q onto B: coeffs (B, n_q, T, r)
                    coeffs = torch.einsum("bhtd,hdr->bhtr", Q_f, B_q)
                    # Per-query-token norm squared
                    Q_norm_sq = (Q_f ** 2).sum(dim=-1) + eps_norm  # (B, n_q, T)

                    # Per-facet ε_{q,f} tensor (F, B, n_q, T)
                    eps_bhtf = torch.empty(
                        n_facets, B_sz, n_q, T,
                        device=Q.device, dtype=torch.float32,
                    )
                    for f in range(n_facets):
                        mask_f = M_q[:, f, :]  # (n_q, r)
                        mc = coeffs * mask_f.unsqueeze(0).unsqueeze(2)
                        num = (mc ** 2).sum(dim=-1)  # (B, n_q, T)
                        eps_bhtf[f] = num / Q_norm_sq

                    # Bucket split
                    eps_bos = eps_bhtf[:, :, :, 0:1]             # (F, B, n_q, 1)
                    eps_bos_flat = eps_bos.reshape(n_facets, -1).double().cpu()
                    tot_bos_flat = eps_bos_flat.sum(dim=0)
                    acc.add_bucket("bos", li, eps_bos_flat, tot_bos_flat)

                    if T > 1:
                        eps_rest = eps_bhtf[:, :, :, 1:]          # (F, B, n_q, T-1)
                        eps_rest_flat = eps_rest.reshape(n_facets, -1).double().cpu()
                        tot_rest_flat = eps_rest_flat.sum(dim=0)
                        acc.add_bucket("rest", li, eps_rest_flat, tot_rest_flat)
                    return None
                return hook

            handles.append(q_proj.register_forward_hook(
                make_hook(layer_idx, B_l, M_l)
            ))
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------
# Data builders — reuse MMLU loader, write a thin MetaTool loader
# ---------------------------------------------------------------------

def mmlu_prompts(args) -> List[str]:
    samples, fewshot = load_mmlu(
        args.mmlu_config, args.n_mmlu, args.n_shot, args.seed,
    )
    return [build_prompt(row, fewshot, args.n_shot) for row in samples]


@torch.no_grad()
def run_dataset(
    model, tokenizer, prompts: List[str], args,
    B_ont, facet_mask, n_q, n_kv, head_dim, L, n_facets, tag: str,
) -> Dict:
    acc = EpsilonQAcc(L=L, n_facets=n_facets,
                      n_bins=args.n_bins, hist_max=args.hist_max)
    t0 = time.time()
    with install_epsilon_q_hooks(
        model, B_ont, facet_mask,
        n_q=n_q, n_kv=n_kv, head_dim=head_dim, acc=acc,
    ):
        for i, prompt in enumerate(prompts):
            ids = tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=args.max_ctx,
            ).to(args.device)
            model(**ids)
            if args.verbose and (i % 25 == 0 or i == len(prompts) - 1):
                print(f"  [{tag}] {i+1}/{len(prompts)}  "
                      f"({time.time()-t0:.1f}s)", flush=True)
    out = acc.to_dict()
    out["tag"] = tag
    out["n_prompts"] = len(prompts)
    out["runtime_s"] = time.time() - t0
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

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
    L = cfg.num_hidden_layers
    kv_group = n_q // n_kv
    print(f"[cfg] L={L} n_q={n_q} n_kv={n_kv} kv_group={kv_group} "
          f"head_dim={head_dim}", flush=True)

    print(f"[ocq] loading B_ont from {args.b_ont}", flush=True)
    payload = torch.load(args.b_ont, map_location="cpu", weights_only=False)
    B_ont = payload["B_ont"] if isinstance(payload, dict) else payload
    if B_ont.shape[:2] != (L, n_kv):
        raise ValueError(
            f"B_ont (L,n_kv)={tuple(B_ont.shape[:2])} != model ({L},{n_kv})")
    if B_ont.shape[2] != head_dim:
        raise ValueError(
            f"B_ont head_dim={B_ont.shape[2]} != {head_dim}")
    if not (isinstance(payload, dict) and "r_per_pair" in payload):
        raise ValueError("B_ont payload missing r_per_pair")
    n_facets = len(payload.get("facet_order", ["f0", "f1", "f2", "f3"]))
    facet_mask = build_facet_masks(
        payload["r_per_pair"], L=L, H=n_kv,
        r_ont=B_ont.shape[-1], n_facets=n_facets,
    )
    print(f"[ocq] B_ont {tuple(B_ont.shape)}  "
          f"facet_mask {tuple(facet_mask.shape)}  "
          f"cols/facet={facet_mask.sum(dim=(0,1,3)).tolist()}", flush=True)

    # --- Build prompts ---
    print(f"[prompts] MMLU n={args.n_mmlu}", flush=True)
    prompts_mmlu = mmlu_prompts(args)
    print(f"[prompts] MetaTool n={args.n_metatool}", flush=True)
    prompts_meta = metatool_prompts(args)

    # --- Run both datasets ---
    print("\n[run] MMLU q_proj forward pass ...", flush=True)
    stats_mmlu = run_dataset(
        model, tok, prompts_mmlu, args,
        B_ont, facet_mask, n_q, n_kv, head_dim, L, n_facets, tag="mmlu",
    )
    mm_rest = stats_mmlu["buckets"]["rest"]
    mm_bos = stats_mmlu["buckets"]["bos"]
    print(f"  MMLU done: runtime={stats_mmlu['runtime_s']:.1f}s", flush=True)
    print(f"    rest: E[Σε_q]={mm_rest['eps_total_mean_global']:.4f}  "
          f"E[ε_q_f]={[f'{x:.3f}' for x in mm_rest['eps_f_mean_global']]}",
          flush=True)
    print(f"    bos : E[Σε_q]={mm_bos['eps_total_mean_global']:.4f}  "
          f"E[ε_q_f]={[f'{x:.3f}' for x in mm_bos['eps_f_mean_global']]}",
          flush=True)

    print("\n[run] MetaTool q_proj forward pass ...", flush=True)
    stats_meta = run_dataset(
        model, tok, prompts_meta, args,
        B_ont, facet_mask, n_q, n_kv, head_dim, L, n_facets, tag="metatool",
    )
    mt_rest = stats_meta["buckets"]["rest"]
    mt_bos = stats_meta["buckets"]["bos"]
    print(f"  MetaTool done: runtime={stats_meta['runtime_s']:.1f}s", flush=True)
    print(f"    rest: E[Σε_q]={mt_rest['eps_total_mean_global']:.4f}  "
          f"E[ε_q_f]={[f'{x:.3f}' for x in mt_rest['eps_f_mean_global']]}",
          flush=True)
    print(f"    bos : E[Σε_q]={mt_bos['eps_total_mean_global']:.4f}  "
          f"E[ε_q_f]={[f'{x:.3f}' for x in mt_bos['eps_f_mean_global']]}",
          flush=True)

    # --- Summary + Cor 6.7 decision-matrix hint (on REST bucket) ---
    ratio = [
        (mm_rest["eps_f_mean_global"][f]
         / max(mt_rest["eps_f_mean_global"][f], 1e-12))
        for f in range(n_facets)
    ]
    total_ratio = (mm_rest["eps_total_mean_global"]
                   / max(mt_rest["eps_total_mean_global"], 1e-12))
    print("\n=== SUMMARY (rest bucket, pos>=1) ===")
    print(f"E[Σ_f ε_q]  MMLU={mm_rest['eps_total_mean_global']:.4f}  "
          f"MetaTool={mt_rest['eps_total_mean_global']:.4f}  "
          f"ratio(MMLU/MetaTool)={total_ratio:.3f}")
    print(f"Per-facet MMLU/MetaTool ratio: {[f'{r:.3f}' for r in ratio]}")

    eps_mmlu = mm_rest["eps_total_mean_global"]
    eps_meta = mt_rest["eps_total_mean_global"]
    sep = eps_meta - eps_mmlu
    print(f"MetaTool − MMLU separation = {sep:+.4f}")

    if eps_mmlu < 0.1 and eps_meta > 0.3:
        verdict = ("ROW 1: Cor 6.7 holds as-is. MMLU queries ≈ ⊥ Range(B), "
                   "MetaTool queries carry first-order steering. "
                   "MMLU degradation must come from Cor 6.6 trace cost "
                   "at rank-1 layers → Step B (rank-aware hook) should fix.")
    elif sep > 0.1 and eps_meta > 0.2:
        verdict = ("ROW 2 (weakened): partial phase separation. "
                   "Pareto-frontier framing likely. Run Step B to see if "
                   "rank-skip pushes the frontier into main-venue territory.")
    else:
        verdict = ("ROW 3: no meaningful query-side separation on this B_ont. "
                   "Cor 6.7 hypothesis empirically false here. Pivot options: "
                   "(a) query null-space basis, (b) weight-only W_Q·W_K^T SVD, "
                   "(c) drop Cor 6.7 entirely, ship Theorem 6.1/6.2 + Mode A/B/C.")
    print(f"\nVerdict: {verdict}")

    payload_out = {
        "model": args.model,
        "b_ont_path": args.b_ont,
        "seed": args.seed,
        "n_mmlu": args.n_mmlu,
        "n_metatool": args.n_metatool,
        "max_ctx": args.max_ctx,
        "gqa": {"n_q": n_q, "n_kv": n_kv, "kv_group": kv_group},
        "facet_order": payload.get("facet_order"),
        "cols_per_facet": facet_mask.sum(dim=(0, 1, 3)).tolist(),
        "mmlu": stats_mmlu,
        "metatool": stats_meta,
        "per_facet_mmlu_over_metatool_ratio": ratio,
        "total_mmlu_over_metatool_ratio": total_ratio,
        "separation_meta_minus_mmlu": sep,
        "verdict": verdict,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload_out, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
