#!/usr/bin/env python3
"""metric_mining.py — data-driven discovery of discriminative goal->tool metrics.

Computes a battery of candidate per-trajectory metrics and RANKS them by AUC
(P(success scores higher than failure)) so we pick discriminators empirically
rather than by guess. Includes F1 / seq_F1 (harmonic of recall&precision) per request.

AUC 0.5 = no signal; >0.5 higher-is-better discriminates; <0.5 lower-is-better
(worse-is-better metrics like extra_actions/n_calls). |AUC-0.5| = strength.

Usage:
  python scripts/distill/metric_mining.py --domain telecom
  python scripts/distill/metric_mining.py --domain telecom --results '<glob>'
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from procedure_scorecard import (  # noqa: E402
    gt_agent_actions, agent_calls, arg_binding, _lcs_len, is_read,
    DEFAULT_TAU2,
)


def _f1(p, r):
    if p is None or r is None or (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r)


def battery(sim, gt_map, entity_keys, action_params):
    tid = sim.get("task_id", "")
    gt_names = [a["name"] for a in gt_map.get(tid, [])]
    if not gt_names:
        return None
    gt_set = set(gt_names)
    calls = agent_calls(sim.get("messages") or [])
    actions = [n for n, _ in calls if not is_read(n)]
    reads = [n for n, _ in calls if is_read(n)]
    aset = set(actions)
    matched = gt_set & aset
    recall = len(matched) / len(gt_set)
    precision = (len(matched) / len(aset)) if aset else 0.0
    lcs = _lcs_len(actions, gt_names)
    seq_match = lcs / len(gt_names)
    seq_prec = (lcs / len(actions)) if actions else 0.0
    ab = arg_binding(sim.get("messages") or [], entity_keys, action_params)
    extra = len(aset - gt_set)
    first_ok = 1.0 if actions and actions[0] in gt_set else 0.0
    cnt = Counter(actions)
    repeat = sum(c - 1 for c in cnt.values() if c > 1)
    return {
        "recall": recall,
        "precision": precision,
        "F1": _f1(precision, recall),
        "seq_match": seq_match,
        "seq_prec": seq_prec,
        "seq_F1": _f1(seq_prec, seq_match),
        "arg_bind": ab,
        "recallxargbind": (recall * ab) if ab is not None else None,
        "first_action_correct": first_ok,
        "exact_set": 1.0 if aset == gt_set else 0.0,
        "superset(req⊆called)": 1.0 if gt_set <= aset else 0.0,
        # worse-is-better:
        "extra_actions(↓)": float(extra),
        "n_action_calls(↓)": float(len(actions)),
        "n_read_calls(↓)": float(len(reads)),
        "n_total_calls(↓)": float(len(calls)),
        "repeat(↓)": float(repeat),
    }


def _auc(pos, neg):
    """P(random pos > random neg), ties=0.5, via average-rank (Mann-Whitney)."""
    pos = [v for v in pos if v is not None]
    neg = [v for v in neg if v is not None]
    if not pos or not neg:
        return None, len(pos), len(neg)
    allv = sorted(pos + neg)
    # average ranks
    ranks = {}
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based average rank
        ranks[allv[i]] = avg
        i = j + 1
    rank_sum_pos = sum(ranks[v] for v in pos)
    u = rank_sum_pos - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg)), len(pos), len(neg)


def analyze(sims, gt_map, entity_keys, action_params, label):
    rows = {"succ": [], "fail": []}
    for s in sims:
        b = battery(s, gt_map, entity_keys, action_params)
        if b is None:
            continue
        cls = "succ" if (s.get("reward_info") or {}).get("reward", 0) >= 0.999 else "fail"
        rows[cls].append(b)
    if not rows["succ"] or not rows["fail"]:
        print(f"\n[{label}] insufficient (succ={len(rows['succ'])} fail={len(rows['fail'])})")
        return
    keys = list(rows["succ"][0].keys())
    res = []
    for k in keys:
        auc, ns, nf = _auc([d[k] for d in rows["succ"]], [d[k] for d in rows["fail"]])
        if auc is None:
            continue
        ms = sum(d[k] for d in rows["succ"] if d[k] is not None) / max(ns, 1)
        mf = sum(d[k] for d in rows["fail"] if d[k] is not None) / max(nf, 1)
        res.append((abs(auc - 0.5), auc, k, ms, mf))
    res.sort(reverse=True)
    print(f"\n[{label}] n: succ={len(rows['succ'])} fail={len(rows['fail'])} — ranked by |AUC-0.5|")
    print(f"  {'metric':22} {'AUC':>6} {'|d|':>6} {'succ':>8} {'fail':>8}")
    for strength, auc, k, ms, mf in res:
        print(f"  {k:22} {auc:>6.3f} {strength:>6.3f} {ms:>8.3f} {mf:>8.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["telecom", "retail", "airline"])
    ap.add_argument("--tau2-root", default=DEFAULT_TAU2)
    ap.add_argument("--shipped-dir", default=DEFAULT_TAU2 + "/data/tau2/results/final")
    ap.add_argument("--results", nargs="*", default=None)
    ap.add_argument("--induced-dir", default="reports/facet_rft_2026/phase4_distill/induced")
    args = ap.parse_args()

    tasks = json.load(open(os.path.join(args.tau2_root, "data", "tau2", "domains",
                                        args.domain, "tasks.json")))
    gt_map = {t.get("id", ""): gt_agent_actions(t) for t in tasks}
    df_path = os.path.join(args.induced_dir, f"param_dataflow_{args.domain}.json")
    ek, ap_map = set(), {}
    if os.path.exists(df_path):
        df = json.load(open(df_path))
        ek = set(df.get("entity_keys", [])); ap_map = df.get("action_params", {})

    if not args.results:
        sims = []
        for f in (glob.glob(os.path.join(args.shipped_dir, f"*_{args.domain}_default_*4trials.json")) +
                  glob.glob(os.path.join(args.shipped_dir, f"*_{args.domain}_base_*4trials.json"))):
            sims += json.load(open(f)).get("simulations", [])
        analyze(sims, gt_map, ek, ap_map, f"shipped {args.domain}")
        return 0
    for r in args.results:
        for p in glob.glob(r):
            try:
                sims = json.load(open(p)).get("simulations", [])
            except Exception as e:
                print(f"[skip] {p}: {e}"); continue
            analyze(sims, gt_map, ek, ap_map, os.path.basename(os.path.dirname(p)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
