#!/usr/bin/env python3
"""Aggregate the K-sweep into a diversity-transfer table/curve (EXPRESSION_DIVERSITY_TRANSFER_DESIGN §5).
Reads /home/woori/scratch/depth/c8/ksweep/{train_*.jsonl.D.json, tau2_*.json, heldout_*.json}.
For each (K, method) cell: D (mean per-op effective rank), tau2 surface-collapse rate (op-slot =
NL verb), tau2 accuracy, held-out (OOD full-diverse) recognition. Prints Y-vs-D ordered by D so the
knee D* and the random-vs-kcenter efficiency gap are visible.
"""
import json, glob, os, re, argparse
from collections import Counter

COLLAPSE = {"exchange", "replace", "update", "modify", "set", "change", "cancel", "swap", "return", None}


def load(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def cell(dir_, K, method):
    tag = f"K{K}_{method}"
    D = load(f"{dir_}/train_{tag}.jsonl.D.json")
    t2 = load(f"{dir_}/tau2_{tag}.json")
    ho = load(f"{dir_}/heldout_{tag}.json")
    row = {"K": K, "method": method,
           "D": (D or {}).get("mean_eff_rank"), "mean_dist": ((D or {}).get("D_overall") or {}).get("mean_dist")}
    if t2:
        ops = Counter(r["op"] for r in t2["rows"])
        coll = sum(v for k, v in ops.items() if k in COLLAPSE)
        row["surf_collapse"] = f"{coll}/{t2['overall'][1]}"
        row["tau2_acc"] = f"{t2['overall'][0]}/{t2['overall'][1]}"
        row["tau2_acc_f"] = t2["overall"][0] / max(t2["overall"][1], 1)
        row["collapse_f"] = coll / max(t2["overall"][1], 1)
    if ho:
        row["heldout_recog"] = ho.get("recognition")
        row["heldout_recog_f"] = (ho.get("recognition") or [0, 1])[0] / max((ho.get("recognition") or [0, 1])[1], 1)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/woori/scratch/depth/c8/ksweep")
    ap.add_argument("--Ks", default="1 2 4 8 16 32")
    args = ap.parse_args()
    Ks = [int(x) for x in args.Ks.split()]
    rows = []
    for m in ("random", "kcenter", "axis"):
        for K in Ks:
            if os.path.exists(f"{args.dir}/train_K{K}_{m}.jsonl.D.json") or os.path.exists(f"{args.dir}/tau2_K{K}_{m}.json"):
                rows.append(cell(args.dir, K, m))
    hdr = f"{'method':>8} {'K':>3} {'D(effR)':>8} {'mdist':>6} | {'surf_collapse':>13} {'tau2_acc':>9} {'heldout_recog':>14}"
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (x["method"], x["K"])):
        D = f"{r['D']:.2f}" if r.get("D") is not None else "  -"
        md = f"{r['mean_dist']:.2f}" if r.get("mean_dist") is not None else " -"
        print(f"{r['method']:>8} {r['K']:>3} {D:>8} {md:>6} | "
              f"{r.get('surf_collapse','-'):>13} {r.get('tau2_acc','-'):>9} {str(r.get('heldout_recog','-')):>14}")
    # curve view: Y vs D
    print("\n=== Y vs D (sorted by D) — knee + random/kcenter efficiency ===")
    have = [r for r in rows if r.get("D") is not None and "collapse_f" in r]
    for r in sorted(have, key=lambda x: x["D"]):
        print(f"  D={r['D']:6.2f} [{r['method']:>7} K={r['K']:>2}]  collapse={r['collapse_f']:.2f}  "
              f"tau2_acc={r.get('tau2_acc_f',0):.2f}  heldout={r.get('heldout_recog_f',0):.2f}")
