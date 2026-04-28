#!/usr/bin/env python3
"""analyze_rank_results.py — Summarize E1 rank measurement JSONs.

For each result file: produces histograms, bimodality test, layer profile,
and a markdown table for the gate memo.

Usage:
  python analyze_rank_results.py --in-glob 'reports/rank_replaceability_2026_04/*_n*.json' \
      --out reports/rank_replaceability_2026_04/analysis_summary.md
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np


def hartigan_dip_proxy(arr: np.ndarray) -> float:
    """Crude bimodality proxy: ratio of (max - median) / (median - min + 1).

    Returns large value when distribution has heavy tail (bimodal candidate).
    Not a real Hartigan dip — just a quick screening metric.
    """
    if len(arr) == 0:
        return 0.0
    med = np.median(arr)
    return float((arr.max() - med) / max(med - arr.min() + 1, 1))


def summarize_one(path: Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    L = d["n_layers"]
    H_q = d["n_heads_q"]
    N = d["n_samples"]
    out = {
        "file": path.name,
        "model": d["model"],
        "task": d["task"],
        "N": N,
        "L": L,
        "H_q": H_q,
        "head_dim": d["head_dim"],
        "prefix_len_mean": float(np.mean(d["prefix_lens"])),
        "wall_seconds": d.get("wall_seconds", 0),
    }
    for tau_str, rs in d["r_star"].items():
        rr = np.array(rs)  # (L, H_q)
        flat = rr.flatten()
        out[f"r_mean_{tau_str}"] = float(flat.mean())
        out[f"r_median_{tau_str}"] = float(np.median(flat))
        out[f"r_max_{tau_str}"] = int(flat.max())
        out[f"r_p90_{tau_str}"] = float(np.percentile(flat, 90))
        out[f"r_p95_{tau_str}"] = float(np.percentile(flat, 95))
        out[f"r_p99_{tau_str}"] = float(np.percentile(flat, 99))
        out[f"bimodality_{tau_str}"] = hartigan_dip_proxy(flat)
        # Layer profile: mean r* per layer
        out[f"layer_mean_{tau_str}"] = rr.mean(axis=1).tolist()
        # Head bimodality: how many heads have r* >= 8 (high-rank "mixed" heads)
        out[f"high_rank_heads_{tau_str}"] = int((rr.flatten() >= 8).sum())
        out[f"high_rank_heads_pct_{tau_str}"] = float((rr.flatten() >= 8).mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-glob", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(args.in_glob))
    files = [f for f in files if "smoke" not in f]
    if not files:
        print(f"no files match {args.in_glob}", file=sys.stderr)
        return 2

    summaries = []
    for fp in files:
        try:
            summaries.append(summarize_one(Path(fp)))
        except Exception as e:
            print(f"[skip] {fp}: {e}", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("# E1 Rank Measurement — Analysis Summary\n\n")
        f.write(f"Auto-generated from: `{args.in_glob}`  \n")
        f.write(f"Files analyzed: {len(summaries)}\n\n")

        f.write("## Headline table (τ=0.95)\n\n")
        f.write("| File | Model | Task | N | r*_mean | r*_med | r*_max | r*_p95 | high-rank heads (≥8) | bimodality | prefix_len |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for s in summaries:
            f.write(
                f"| `{s['file']}` | {s['model'].split('/')[-1]} | {s['task']} | "
                f"{s['N']} | {s['r_mean_0.95']:.2f} | "
                f"{s['r_median_0.95']:.0f} | {s['r_max_0.95']} | "
                f"{s['r_p95_0.95']:.1f} | "
                f"{s['high_rank_heads_0.95']}/{s['L']*s['H_q']} ({100*s['high_rank_heads_pct_0.95']:.1f}%) | "
                f"{s['bimodality_0.95']:.2f} | {s['prefix_len_mean']:.0f} |\n"
            )

        f.write("\n## Per-τ summary\n\n")
        for tau in ["0.90", "0.95", "0.99"]:
            f.write(f"### τ = {tau}\n\n")
            f.write("| File | r*_mean | r*_med | r*_max | r*_p99 | high-rank heads |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for s in summaries:
                f.write(
                    f"| `{s['file']}` | {s[f'r_mean_{tau}']:.2f} | "
                    f"{s[f'r_median_{tau}']:.0f} | {s[f'r_max_{tau}']} | "
                    f"{s[f'r_p99_{tau}']:.1f} | "
                    f"{s[f'high_rank_heads_{tau}']}/{s['L']*s['H_q']} |\n"
                )
            f.write("\n")

        f.write("## Layer profiles (τ=0.95, mean r* per layer)\n\n")
        for s in summaries:
            f.write(f"### {s['file']}\n\n")
            lp = s["layer_mean_0.95"]
            f.write("```\n")
            for ell, v in enumerate(lp):
                bar = "█" * int(v * 2)
                f.write(f"L{ell:02d}: {v:5.2f}  {bar}\n")
            f.write("```\n\n")

        f.write("## Diagnostic notes\n\n")
        f.write(
            "- **Theorem 1 prediction**: r*(τ) determines the rank-bound for static prompt replaceability. "
            "Mean r*(0.95) ≤ 16 ⇒ corollary 1.1 sufficient condition; > 64 ⇒ corollary 1.2 (need query-conditional).\n"
            "- **Bimodality**: high `max` with low `median` indicates head specialization. Look for high-rank heads (r* ≥ 8) clustering in particular layers.\n"
            "- **Caveat (τ² runs)**: this loader uses a *generic* tool-selection system prompt without per-domain tool catalogs. Real τ² evaluation feeds the full RETAIL_TOOLS/TELECOM_TOOLS/etc. catalog. Numbers below should be interpreted as lower bounds; full-catalog measurement is a follow-up.\n"
        )

    print(f"wrote {out_path}")
    print()
    # Quick stdout summary
    for s in summaries:
        print(
            f"{s['model'].split('/')[-1]} {s['task']:<14} N={s['N']:>3}  "
            f"r*(.95) mean={s['r_mean_0.95']:.2f} med={s['r_median_0.95']:.0f} "
            f"max={s['r_max_0.95']:>3}  bimod={s['bimodality_0.95']:.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
