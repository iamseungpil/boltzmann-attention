#!/usr/bin/env python3
"""fault_fix_induce.py (item 4) — refined goal->tool ground-truth from tasks.json.

The v1 heuristic in score_fix_coverage.induce_fault_fix counts trajectory tool
co-occurrence per fault, which is contaminated in MULTI-FAULT tasks (telecom has NO
single-fault tasks; min 2). This module derives a CLEAN map from the benchmark's own
ground truth `evaluation_criteria.actions` (the canonical required write-actions per
task), which is independent of trajectory noise:

  telecom : DIFFERENTIAL attribution — fault F's fix = the GT action tool whose
            presence-probability is much higher in tasks WITH F than WITHOUT F
            (lift), removing co-occurrence artifacts (e.g. fixes data_mode_off ->
            toggle_data instead of the spurious refuel_data).
  retail/airline : task_id is an integer (no fault decomposition); emit a
            task_id -> required-tools map (goal->tool at task granularity).

Emits canonical maps consumed by score_fix_coverage.py (--fault-fix-map) and
wiseflow_baseline.py:
  fault_fix_map.json        {telecom: {fault: fix_tool}}
  task_required_tools.json  {domain: {task_id: [required action tools]}}

Usage:
  python scripts/distill/fault_fix_induce.py --domains telecom retail airline \
     --out-dir reports/facet_rft_2026/phase4_distill/induced
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_fix_coverage import parse_task, is_read, induce_fault_fix  # noqa: E402

DEFAULT_TAU2 = "/home/woori/workspace_common/boltzmann-attention/external/tau2-bench"


def load_tasks(domain, tau2_root):
    p = os.path.join(tau2_root, "data", "tau2", "domains", domain, "tasks.json")
    return json.load(open(p))


def task_required_tools(task):
    """GT action tool names from evaluation_criteria.actions (exclude read-only)."""
    ec = task.get("evaluation_criteria") or {}
    tools = []
    for a in ec.get("actions") or []:
        name = a.get("name") or a.get("func_name")
        if name and not is_read(name):
            tools.append(name)
    return sorted(set(tools))


def differential_fault_fix(tasks, min_present=5, min_present_prob=0.5, min_lift=0.30):
    """telecom: fault -> tool with high P(tool in GT actions | fault) and high lift."""
    # task_id -> required tools, fault set
    rows = []
    for t in tasks:
        tid = t.get("id", "")
        _, faults = parse_task(tid)
        rows.append((set(faults), set(task_required_tools(t))))
    all_faults = set().union(*[f for f, _ in rows]) if rows else set()
    all_tools = set().union(*[g for _, g in rows]) if rows else set()
    fix = {}
    detail = {}
    for flt in all_faults:
        with_f = [g for f, g in rows if flt in f]
        without_f = [g for f, g in rows if flt not in f]
        if len(with_f) < min_present:
            continue
        best, best_lift, best_p = None, -1, 0
        for tool in all_tools:
            p_w = sum(1 for g in with_f if tool in g) / len(with_f)
            p_wo = (sum(1 for g in without_f if tool in g) / len(without_f)) if without_f else 0.0
            lift = p_w - p_wo
            if p_w >= min_present_prob and lift > best_lift:
                best, best_lift, best_p = tool, lift, p_w
        if best and best_lift >= min_lift:
            fix[flt] = best
            detail[flt] = {"tool": best, "p_present": round(best_p, 2),
                           "lift": round(best_lift, 2), "n_with": len(with_f)}
    return fix, detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["telecom", "retail", "airline"])
    ap.add_argument("--tau2-root", default=DEFAULT_TAU2)
    ap.add_argument("--shipped-dir",
                    default=DEFAULT_TAU2 + "/data/tau2/results/final")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    fault_fix = {}
    task_required = {}
    for dom in args.domains:
        tasks = load_tasks(dom, args.tau2_root)
        task_required[dom] = {t.get("id", ""): task_required_tools(t) for t in tasks}
        print(f"\n=== {dom} === {len(tasks)} tasks; "
              f"mean GT action tools/task = "
              f"{sum(len(v) for v in task_required[dom].values())/max(len(tasks),1):.2f}")
        if dom == "telecom":
            fix, detail = differential_fault_fix(tasks)
            fault_fix[dom] = fix
            v1 = induce_fault_fix(dom, args.shipped_dir)
            print(f"  refined fault->fix: {len(fix)} faults")
            changed = 0
            for flt in sorted(fix):
                tag = ""
                if flt in v1 and v1[flt] != fix[flt]:
                    tag = f"  [v1: {v1[flt]} -> refined: {fix[flt]} ★CHANGED]"; changed += 1
                elif flt not in v1:
                    tag = "  [new]"
                d = detail[flt]
                print(f"    {flt:32} -> {fix[flt]:24} (p={d['p_present']} lift={d['lift']} n={d['n_with']}){tag}")
            print(f"  changed vs v1 heuristic: {changed}")
        else:
            # sample task->tools
            ex = list(task_required[dom].items())[:4]
            for tid, tools in ex:
                print(f"    task {tid}: {tools}")

    json.dump(fault_fix, open(os.path.join(args.out_dir, "fault_fix_map.json"), "w"),
              indent=2, ensure_ascii=False)
    json.dump(task_required, open(os.path.join(args.out_dir, "task_required_tools.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote fault_fix_map.json + task_required_tools.json -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
