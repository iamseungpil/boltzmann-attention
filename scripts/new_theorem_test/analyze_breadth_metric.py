#!/usr/bin/env python3
"""F4 H-J breadth metric — subspace-alignment between source/target B_onts.

Per consolidation decision F4 (see
`memory/consolidation_framing_decisions_2026_04_19.md`):

    breadth(B_source → target) := mean singular value of
                                    B_source^T · B_target
    per-(L, h), averaged across (L, h).

Because `B_target` is itself built by Gram-Schmidt orthonormalization of
K activations at tool-related token positions in the target benchmark,
`B_target`'s column span is the **target-K-tool-subspace**.  Therefore the
principal-angle cosines between B_source and B_target columns are exactly
the (normalized) singular values requested by F4, computable without any
GPU from the stored `B_ont.pt` artifacts.

Output: `reports/new_theorem_test/phase_b3_breadth_scores.json`

For each B1 cross-benchmark pair we also pull the stored A/D attn_fro
ratio and KL ratio, producing a breadth-vs-transferability table and a
Pearson correlation coefficient for §5.4 ICLR draft.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch


def load_bont(path: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "B_ont" in obj:
        return obj["B_ont"].float(), obj
    return obj.float(), {}


def breadth_score(B_src: torch.Tensor, B_tgt: torch.Tensor):
    """Per-(L, h) mean principal-angle cosine = mean singular value of
    B_src[L, h]^T @ B_tgt[L, h].  Returns (breadth_mean, breadth_median,
    per_lh list of (L, h, breadth_lh, min_sv, max_sv, r_min))."""
    assert B_src.shape[0] == B_tgt.shape[0], "layer count mismatch"
    assert B_src.shape[1] == B_tgt.shape[1], "head count mismatch"
    L, H, d, r_s = B_src.shape
    _, _, _, r_t = B_tgt.shape
    per_lh = []
    scores = []
    for li in range(L):
        for h in range(H):
            Bs = B_src[li, h]  # (d, r_s)
            Bt = B_tgt[li, h]  # (d, r_t)
            # Guard against zero-norm columns
            if Bs.norm() < 1e-6 or Bt.norm() < 1e-6:
                continue
            M = Bs.T @ Bt  # (r_s, r_t)
            svs = torch.linalg.svdvals(M)  # min(r_s, r_t) values in [0, 1]
            b = float(svs.mean().item())
            scores.append(b)
            per_lh.append({
                "layer": li, "head": h,
                "breadth": b,
                "min_sv": float(svs.min().item()),
                "max_sv": float(svs.max().item()),
                "r_min": int(min(r_s, r_t)),
            })
    if not scores:
        return None, None, []
    return statistics.mean(scores), statistics.median(scores), per_lh


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def spearman(xs, ys):
    """Rank correlation."""
    def rank(a):
        s = sorted(range(len(a)), key=lambda i: a[i])
        r = [0] * len(a)
        for idx, pos in enumerate(s):
            r[pos] = idx + 1
        return r
    return pearson(rank(xs), rank(ys))


def main():
    REPO = Path(__file__).resolve().parents[2]
    proj = REPO / "external/SEKA/seka_projections"
    B_paths = {
        "telecom":  str(proj / "ontology-qwen25-7b-tau2-telecom/B_ont.pt"),
        "retail":   str(proj / "ontology-qwen25-7b-tau2-retail/B_ont.pt"),
        "airline":  str(proj / "ontology-qwen25-7b-tau2-airline/B_ont.pt"),
        "banking":  str(proj / "ontology-qwen25-7b-tau2-banking/B_ont.pt"),
        "metatool": str(proj / "ontology-qwen25-7b-metatool/B_ont.pt"),
    }
    # Phase C permuted B_onts (source comparisons for direction-specificity-origin)
    perm_paths = {
        "telecom_perm_facet_values": str(proj / "ontology-qwen25-7b-tau2-telecom-perm-facet_values-s42/B_ont.pt"),
        "telecom_perm_tool_names":   str(proj / "ontology-qwen25-7b-tau2-telecom-perm-tool_names-s42/B_ont.pt"),
        "telecom_perm_full_random":  str(proj / "ontology-qwen25-7b-tau2-telecom-perm-full_random-s42/B_ont.pt"),
    }

    B_cache = {}
    for k, p in {**B_paths, **perm_paths}.items():
        if os.path.exists(p):
            B, meta = load_bont(p)
            B_cache[k] = B
            print(f"[load] {k:30s} shape={tuple(B.shape)}")
        else:
            print(f"[miss] {k}: {p}")

    # B1 cross-benchmark pairs used in Phase B1. target "st4" uses metatool B_ont.
    B1_pairs = [
        # (label, source_key, target_key, target_eval_bench_name, b1_json)
        ("telecom→retail",  "telecom", "retail",   "tau2_retail",   "b1_telecom_on_retail.json"),
        ("telecom→airline", "telecom", "airline",  "tau2_airline",  "b1_telecom_on_airline.json"),
        ("telecom→banking", "telecom", "banking",  "tau2_banking",  "b1_telecom_on_banking.json"),
        ("telecom→st4",     "telecom", "metatool", "metatool_st4",  "b1_telecom_on_st4.json"),
        ("retail→telecom",  "retail",  "telecom",  "tau2_telecom",  "b1_retail_on_telecom.json"),
        ("metatool→telecom","metatool","telecom",  "tau2_telecom",  "b1_metatool_on_telecom.json"),
        ("retail→retail",   "retail",  "retail",   "tau2_retail",   "b1_retail_on_retail.json"),
    ]

    reports_dir = REPO / "reports/new_theorem_test"

    rows = []
    print()
    print(f"{'Pair':22s}  {'breadth μ':>9s}  {'breadth med':>11s}  {'A/D KL':>7s}  {'A/D attn_fro':>12s}  {'r_src':>5s}  {'r_tgt':>5s}")
    print("-" * 95)
    for label, src, tgt, bench, b1_json in B1_pairs:
        if src not in B_cache or tgt not in B_cache:
            print(f"[skip] {label}: missing B_ont")
            continue
        B_src = B_cache[src]
        B_tgt = B_cache[tgt]
        if B_src.shape[:3] != B_tgt.shape[:3]:
            print(f"[skip] {label}: shape mismatch {B_src.shape} vs {B_tgt.shape}")
            continue
        b_mean, b_med, per_lh = breadth_score(B_src, B_tgt)
        ad_kl = ad_attn = None
        b1_path = reports_dir / b1_json
        if b1_path.exists():
            bj = json.load(open(b1_path))
            gap = bj.get("gap", {})
            ad_kl = gap.get("kl_A_over_D_ratio")
            sA = bj.get("summary_A", {}); sD = bj.get("summary_D", {})
            if sA.get("attn_fro_mean") and sD.get("attn_fro_mean"):
                ad_attn = sA["attn_fro_mean"] / sD["attn_fro_mean"]
        r_src = B_src.shape[-1]; r_tgt = B_tgt.shape[-1]
        print(f"{label:22s}  {b_mean:>9.4f}  {b_med:>11.4f}  "
              f"{(ad_kl if ad_kl is not None else -1):>7.3f}  "
              f"{(ad_attn if ad_attn is not None else -1):>12.3f}  "
              f"{r_src:>5d}  {r_tgt:>5d}")
        rows.append({
            "label": label, "source": src, "target": tgt, "bench": bench,
            "breadth_mean": b_mean, "breadth_median": b_med,
            "ad_kl": ad_kl, "ad_attn_fro": ad_attn,
            "r_src": r_src, "r_tgt": r_tgt,
            "n_lh": len(per_lh),
            "per_lh": per_lh,
        })

    # Correlations
    br_kl_xs = [r["breadth_mean"] for r in rows if r["ad_kl"] is not None]
    br_kl_ys = [r["ad_kl"] for r in rows if r["ad_kl"] is not None]
    br_attn_xs = [r["breadth_mean"] for r in rows if r["ad_attn_fro"] is not None]
    br_attn_ys = [r["ad_attn_fro"] for r in rows if r["ad_attn_fro"] is not None]

    corr = {
        "pearson_breadth_vs_AD_KL":       pearson(br_kl_xs, br_kl_ys),
        "spearman_breadth_vs_AD_KL":      spearman(br_kl_xs, br_kl_ys),
        "pearson_breadth_vs_AD_attn_fro": pearson(br_attn_xs, br_attn_ys),
        "spearman_breadth_vs_AD_attn_fro":spearman(br_attn_xs, br_attn_ys),
    }

    print()
    print("Correlation of breadth vs transferability metrics (B1 7 pairs):")
    for k, v in corr.items():
        print(f"  {k:>34s} = {v:+.4f}")

    # Threshold check per F4: breadth ≥ 0.3 → broad, < 0.1 → narrow
    print("\nF4 threshold check (breadth ≥ 0.3 = broad, < 0.1 = narrow):")
    for r in rows:
        category = "broad" if r["breadth_mean"] >= 0.3 else ("narrow" if r["breadth_mean"] < 0.1 else "mid")
        matches = None
        if r["ad_attn_fro"] is not None:
            # Broad should have A/D attn_fro > 1.5 per F2
            if category == "broad":
                matches = r["ad_attn_fro"] > 1.5
            elif category == "narrow":
                matches = r["ad_attn_fro"] < 1.5
            else:
                matches = None
        print(f"  {r['label']:22s} breadth={r['breadth_mean']:.3f} → {category:6s}  "
              f"A/D attn_fro={r['ad_attn_fro'] if r['ad_attn_fro'] else 'N/A':.3f}"
              + (f"  [F4 prediction {'OK' if matches else 'MISS'}]" if matches is not None else ""))

    # Phase C permuted B_onts: are they in the same subspace as real Telecom B_ont?
    print("\nPhase C permutation→Telecom breadth scores:")
    perm_rows = []
    for perm_key in ["telecom_perm_facet_values", "telecom_perm_tool_names", "telecom_perm_full_random"]:
        if perm_key not in B_cache:
            continue
        b_mean, b_med, _ = breadth_score(B_cache[perm_key], B_cache["telecom"])
        print(f"  {perm_key:32s}  breadth vs real Telecom = {b_mean:.4f}  (median {b_med:.4f})")
        perm_rows.append({"label": perm_key, "breadth_mean": b_mean, "breadth_median": b_med})

    out = {
        "B1_pairs": rows,
        "perm_to_telecom": perm_rows,
        "correlations": corr,
    }
    out_path = reports_dir / "phase_b3_breadth_scores.json"
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
