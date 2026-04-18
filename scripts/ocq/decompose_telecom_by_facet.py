"""Decompose Telecom N=200 F1 results by domain-facet subset.

Reads:
  - Facet analysis JSON (telecom_gt_facet_analysis_v2.json)
  - Per-sample JSONs for each method (no_steer/ladapt/Q+/canonical_adaseka)

Computes per-subset F1 + per-sample paired bootstrap CI.

Usage:
  python decompose_telecom_by_facet.py \
    --facet reports/tau2_2026_04_18/telecom_gt_facet_analysis_v2.json \
    --ladapt-file reports/tau2_2026_04_18/qwen25_telecom_ladapt_qpos_paper4.json \
    --canonical-file reports/tau2_2026_04_18/telecom_canonical_amp03_persample_N200.json \
    --out reports/tau2_2026_04_18/telecom_facet_decomposition.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List


def load_persample(fp: str, wanted: List[str]) -> Dict[str, Dict[str, float]]:
    """Load per-sample results from an eval_tau2_bench JSON. Return
    method -> {task_id -> F1}."""
    d = json.load(open(fp))
    out = {}
    for r in d.get("results", []):
        m = r["method"]
        if wanted and m not in wanted:
            continue
        by_tid = {}
        for s in r.get("per_sample", []):
            tid = s.get("task_id")
            f1 = float(s.get("metrics", {}).get("F1", 0.0))
            by_tid[tid] = f1
        out[m] = by_tid
    return out


def subset_stats(method2f1: Dict[str, Dict[str, float]], subset_ids: set, method: str) -> Dict:
    vals = [method2f1[method][t] for t in subset_ids if t in method2f1[method]]
    return {
        "n": len(vals),
        "mean_F1": sum(vals) / max(1, len(vals)),
        "values": vals,
    }


def paired_bootstrap_ci(
    per_sample_pairs: List[tuple], n_boot: int = 2000, ci: float = 0.95
) -> Dict:
    """per_sample_pairs = list of (f1_method, f1_baseline).
    Returns mean diff + CI on mean(method - baseline)."""
    rng = random.Random(0)
    n = len(per_sample_pairs)
    if n == 0:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    diffs = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        s = sum((per_sample_pairs[i][0] - per_sample_pairs[i][1]) for i in idxs) / n
        diffs.append(s)
    diffs.sort()
    lo_idx = int((1 - ci) / 2 * n_boot)
    hi_idx = int((1 - (1 - ci) / 2) * n_boot) - 1
    mean_diff = sum((a - b) for a, b in per_sample_pairs) / n
    return {
        "mean_diff": mean_diff,
        "ci_low": diffs[lo_idx],
        "ci_high": diffs[hi_idx],
        "n": n,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--facet", required=True)
    p.add_argument("--ladapt-file", required=True)
    p.add_argument("--canonical-file", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # Load facet analysis
    facet = json.load(open(args.facet))
    per_task_facet = {x["id"]: x for x in facet["per_task"]}

    # Classify tasks by domain diversity
    single_domain_ids = {tid for tid, x in per_task_facet.items() if x["diversity"]["domain"] <= 1}
    multi_domain_ids = {tid for tid, x in per_task_facet.items() if x["diversity"]["domain"] >= 2}
    print(f"[subsets] single-domain (≤1 unique domain in GT): {len(single_domain_ids)}")
    print(f"[subsets] multi-domain (≥2 unique domains in GT): {len(multi_domain_ids)}")

    # Load per-sample F1 results
    ladapt_methods = ["no_steer", "ocq_ladapt_k0.05_q-0.03", "ocq_qbias_b0.05", "ocq_qbias_b0.10"]
    ladapt_ps = load_persample(args.ladapt_file, ladapt_methods)

    canonical_methods = ["no_steer", "canonical_adaseka_amp0.3_topk3_T1.0"]
    canon_ps = load_persample(args.canonical_file, canonical_methods)

    # Merge — use ladapt_file no_steer as canonical baseline reference
    methods = {
        "no_steer": ladapt_ps.get("no_steer", {}),
        "ocq_qbias_b0.05": ladapt_ps.get("ocq_qbias_b0.05", {}),
        "ocq_qbias_b0.10": ladapt_ps.get("ocq_qbias_b0.10", {}),
        "ocq_ladapt_k0.05_q-0.03": ladapt_ps.get("ocq_ladapt_k0.05_q-0.03", {}),
        "canonical_adaseka_amp0.3_topk3_T1.0": canon_ps.get("canonical_adaseka_amp0.3_topk3_T1.0", {}),
    }

    print(f"\n[per-method task coverage]")
    for m, d in methods.items():
        print(f"  {m:40s}: {len(d)} tasks")

    # Sanity: canonical and ladapt no_steer F1 should match
    cn = canon_ps.get("no_steer", {})
    ln = ladapt_ps.get("no_steer", {})
    overlap_ids = set(cn.keys()) & set(ln.keys())
    if overlap_ids:
        diffs = [abs(cn[t] - ln[t]) for t in overlap_ids]
        print(f"\n[sanity] no_steer overlap: {len(overlap_ids)} tasks, max |F1 diff| = {max(diffs):.4f}")

    # Per-subset F1 analysis
    subsets = {
        "all": set(per_task_facet.keys()),
        "single_domain": single_domain_ids,
        "multi_domain": multi_domain_ids,
    }

    print(f"\n{'='*100}")
    print(f"{'Subset':<15} {'Method':<40} {'N':<5} {'Mean F1':<10} {'ΔF1':<10} {'ΔF1 95% CI':<22}")
    print("="*100)

    results = {}
    for sname, ids in subsets.items():
        results[sname] = {}
        baseline_tid2f1 = {t: methods["no_steer"][t] for t in ids if t in methods["no_steer"]}
        base_mean = sum(baseline_tid2f1.values()) / max(1, len(baseline_tid2f1))
        results[sname]["no_steer"] = {"n": len(baseline_tid2f1), "mean_F1": base_mean}
        print(f"{sname:<15} {'no_steer':<40} {len(baseline_tid2f1):<5} {base_mean:<10.4f} {'-':<10} {'-':<22}")
        for m in ["ocq_qbias_b0.05", "ocq_qbias_b0.10", "ocq_ladapt_k0.05_q-0.03", "canonical_adaseka_amp0.3_topk3_T1.0"]:
            m_tid2f1 = methods.get(m, {})
            pairs = []
            for t in ids:
                if t in m_tid2f1 and t in baseline_tid2f1:
                    pairs.append((m_tid2f1[t], baseline_tid2f1[t]))
            if not pairs:
                continue
            n = len(pairs)
            mean_m = sum(p[0] for p in pairs) / n
            boot = paired_bootstrap_ci(pairs)
            ci_str = f"[{boot['ci_low']*100:+.2f}, {boot['ci_high']*100:+.2f}]"
            print(f"{sname:<15} {m:<40} {n:<5} {mean_m:<10.4f} {boot['mean_diff']*100:<+10.2f} {ci_str:<22}")
            results[sname][m] = {
                "n": n,
                "mean_F1": mean_m,
                "mean_diff_pp": boot["mean_diff"] * 100,
                "ci_low_pp": boot["ci_low"] * 100,
                "ci_high_pp": boot["ci_high"] * 100,
            }
        print("-"*100)

    # Additional: single vs multi differential for canonical AdaSEKA
    print(f"\n{'='*80}")
    print("LAYER 1/2 DIAGNOSTIC: Does canonical AdaSEKA degrade on multi-domain?")
    print("="*80)
    if "canonical_adaseka_amp0.3_topk3_T1.0" in methods:
        c_single = results["single_domain"].get("canonical_adaseka_amp0.3_topk3_T1.0", {})
        c_multi = results["multi_domain"].get("canonical_adaseka_amp0.3_topk3_T1.0", {})
        print(f"  canonical_adaseka amp=0.3 on single-domain: ΔF1 = {c_single.get('mean_diff_pp',0):+.2f}pp [{c_single.get('ci_low_pp',0):+.2f}, {c_single.get('ci_high_pp',0):+.2f}]")
        print(f"  canonical_adaseka amp=0.3 on multi-domain : ΔF1 = {c_multi.get('mean_diff_pp',0):+.2f}pp [{c_multi.get('ci_low_pp',0):+.2f}, {c_multi.get('ci_high_pp',0):+.2f}]")
    if "ocq_ladapt_k0.05_q-0.03" in methods:
        l_single = results["single_domain"].get("ocq_ladapt_k0.05_q-0.03", {})
        l_multi = results["multi_domain"].get("ocq_ladapt_k0.05_q-0.03", {})
        print(f"  ladapt on single-domain: ΔF1 = {l_single.get('mean_diff_pp',0):+.2f}pp [{l_single.get('ci_low_pp',0):+.2f}, {l_single.get('ci_high_pp',0):+.2f}]")
        print(f"  ladapt on multi-domain : ΔF1 = {l_multi.get('mean_diff_pp',0):+.2f}pp [{l_multi.get('ci_low_pp',0):+.2f}, {l_multi.get('ci_high_pp',0):+.2f}]")
    if "ocq_qbias_b0.05" in methods:
        q_single = results["single_domain"].get("ocq_qbias_b0.05", {})
        q_multi = results["multi_domain"].get("ocq_qbias_b0.05", {})
        print(f"  Q+0.05 on single-domain: ΔF1 = {q_single.get('mean_diff_pp',0):+.2f}pp [{q_single.get('ci_low_pp',0):+.2f}, {q_single.get('ci_high_pp',0):+.2f}]")
        print(f"  Q+0.05 on multi-domain : ΔF1 = {q_multi.get('mean_diff_pp',0):+.2f}pp [{q_multi.get('ci_low_pp',0):+.2f}, {q_multi.get('ci_high_pp',0):+.2f}]")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "subsets_size": {k: len(v) for k, v in subsets.items()},
            "results": results,
        }, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
