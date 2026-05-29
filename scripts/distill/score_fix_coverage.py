#!/usr/bin/env python3
"""score_fix_coverage.py — eval metric for the goal->tool distillation campaign.

Computes, for a tau2 results.json (phase1_runner output) or shipped teacher file:
  - pass^1                (mean reward >= 0.999)
  - fix-coverage          per trajectory: fraction of the task's faults whose induced
                          fix-tool the agent actually called; mean over trajectories
                          (split by success/failure)
  - FULL-coverage rate    fraction of trajectories with coverage == 1.0
  - error decomposition   (optional) A wrong/missing-tool / B long-horizon / C format

`fix-coverage` is the core lever metric (§13): student baseline ≈ 0.06, teacher
success ≈ 0.75. Track-A provides this scorer; Track-B (B2) runs it on eval outputs.

Reusable API (imported by wiseflow_baseline.py / fault_fix_induce.py):
  parse_task(task_id) -> (issue, [faults])
  agent_tool_seq(messages) -> [tool names]   (requestor==assistant only)
  induce_fault_fix(domain, shipped_dir, ...) -> {fault: fix_tool}

The induced map here is the v1 heuristic (top action-tool per fault in successes);
fault_fix_induce.py (item 4) produces a refined canonical map loadable via
--fault-fix-map.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
import sys
from collections import Counter, defaultdict

READ_PREFIXES = ("get_", "list_", "find_", "search_", "lookup")


def is_read(tool: str) -> bool:
    return tool.startswith(READ_PREFIXES)


def parse_task(task_id: str):
    m = re.match(r"\[(.*?)\](.*?)\[PERSONA", task_id or "") or \
        re.match(r"\[(.*?)\](.*)", task_id or "")
    if not m:
        return "?", []
    return m.group(1), [f for f in m.group(2).split("|") if f]


def agent_tool_seq(messages):
    seq = []
    for m in messages or []:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if tc.get("requestor", "assistant") == "assistant" and tc.get("name"):
                    seq.append(tc["name"])
    return seq


def _tool_errors(messages):
    return sum(1 for m in (messages or [])
               if m.get("role") == "tool" and m.get("requestor") == "assistant" and m.get("error"))


def _shipped_files(domain, shipped_dir):
    return (glob.glob(os.path.join(shipped_dir, f"*_{domain}_default_*4trials.json")) +
            glob.glob(os.path.join(shipped_dir, f"*_{domain}_base_*4trials.json")))


def induce_fault_fix(domain, shipped_dir, min_support=10, min_frac=0.5):
    """v1 heuristic: top ACTION tool per fault among shipped teacher successes."""
    fault_tool = defaultdict(Counter)
    fault_n = Counter()
    for f in _shipped_files(domain, shipped_dir):
        for s in json.load(open(f)).get("simulations", []):
            if (s.get("reward_info") or {}).get("reward", 0) < 0.999:
                continue
            _, faults = parse_task(s.get("task_id", ""))
            actions = [t for t in set(agent_tool_seq(s.get("messages") or [])) if not is_read(t)]
            for flt in faults:
                fault_n[flt] += 1
                for t in actions:
                    fault_tool[flt][t] += 1
    fix = {}
    for flt, n in fault_n.items():
        if n < min_support or not fault_tool[flt]:
            continue
        tool, c = fault_tool[flt].most_common(1)[0]
        if c / n >= min_frac:
            fix[flt] = tool
    return fix


def score_sims(sims, fix_map, with_decomp=False, task_req=None):
    """Coverage metric. If task_req ({task_id: [required tools]}) given, use the
    benchmark GT action set (reward-aligned; recommended). Else fault->fix based."""
    cov = {"succ": [], "fail": []}
    full = {"succ": 0, "fail": 0}
    n = {"succ": 0, "fail": 0}
    rewards = []
    decomp = Counter()
    for s in sims:
        reward = (s.get("reward_info") or {}).get("reward", 0)
        rewards.append(1.0 if reward >= 0.999 else 0.0)
        cls = "succ" if reward >= 0.999 else "fail"
        seq = agent_tool_seq(s.get("messages") or [])
        called = set(seq)
        tid = s.get("task_id", "")
        if task_req is not None:
            required = set(task_req.get(tid, []))
        else:
            _, faults = parse_task(tid)
            required = {fix_map[f] for f in faults if f in fix_map}
        if required:
            c = len(required & called) / len(required)
            cov[cls].append(c)
            n[cls] += 1
            if c >= 0.999:
                full[cls] += 1
        else:
            c = None
        if with_decomp and cls == "fail":
            err = _tool_errors(s.get("messages") or [])
            term = str(s.get("termination_reason", "")).lower()
            if err >= 2:
                decomp["C_format_param"] += 1
            elif c is not None and c < 0.5:
                decomp["A_wrong_or_missing_tool"] += 1
            elif "max" in term or len(seq) >= 20:
                decomp["B_longhorizon"] += 1
            else:
                decomp["other"] += 1
    res = {
        "n": len(sims),
        "pass^1": round(st.mean(rewards), 4) if rewards else 0.0,
        "fix_cov_success": round(st.mean(cov["succ"]), 4) if cov["succ"] else None,
        "fix_cov_failure": round(st.mean(cov["fail"]), 4) if cov["fail"] else None,
        "fix_cov_all": round(st.mean(cov["succ"] + cov["fail"]), 4) if (cov["succ"] or cov["fail"]) else None,
        "full_cov_success": round(full["succ"] / n["succ"], 4) if n["succ"] else None,
        "full_cov_failure": round(full["fail"] / n["fail"], 4) if n["fail"] else None,
        "n_scored": n["succ"] + n["fail"],
    }
    if with_decomp:
        tot = sum(decomp.values())
        res["fail_decomp"] = {k: f"{v} ({v/max(tot,1):.0%})" for k, v in decomp.most_common()}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["telecom", "retail", "airline"])
    ap.add_argument("--shipped-dir",
                    default="/home/woori/workspace_common/boltzmann-attention/"
                            "external/tau2-bench/data/tau2/results/final")
    ap.add_argument("--results", nargs="*", default=None,
                    help="results.json path(s)/glob to score; if omitted, scores shipped data")
    ap.add_argument("--fault-fix-map", default=None,
                    help="JSON {fault: fix_tool}; if omitted, induce v1 from shipped")
    ap.add_argument("--task-required-map", default=None,
                    help="JSON {domain:{task_id:[tools]}} (from fault_fix_induce.py); "
                         "REWARD-ALIGNED GT-action coverage. Overrides fault-fix.")
    ap.add_argument("--emit-map", default=None, help="write induced map to this path")
    ap.add_argument("--decomp", action="store_true", help="failure error-type decomposition")
    args = ap.parse_args()

    if args.fault_fix_map:
        fix_map = json.load(open(args.fault_fix_map))
        # map may be {domain: {fault: tool}} or {fault: tool}
        if args.domain in fix_map:
            fix_map = fix_map[args.domain]
        print(f"[map] loaded {len(fix_map)} fault->fix from {args.fault_fix_map}")
    else:
        fix_map = induce_fault_fix(args.domain, args.shipped_dir)
        print(f"[map] induced v1 {len(fix_map)} fault->fix from shipped {args.domain}")
    if args.emit_map:
        json.dump(fix_map, open(args.emit_map, "w"), indent=2, ensure_ascii=False)
        print(f"[map] wrote -> {args.emit_map}")

    task_req = None
    if args.task_required_map:
        tr = json.load(open(args.task_required_map))
        task_req = tr.get(args.domain, tr)
        print(f"[map] REWARD-ALIGNED task->required-tools: {len(task_req)} tasks "
              f"(GT-action coverage mode)")

    if not args.results:
        files = _shipped_files(args.domain, args.shipped_dir)
        sims = []
        for f in files:
            sims += json.load(open(f)).get("simulations", [])
        print(f"[score:shipped {args.domain}] {json.dumps(score_sims(sims, fix_map, args.decomp, task_req), ensure_ascii=False)}")
        return 0

    paths = []
    for r in args.results:
        paths += glob.glob(r)
    for p in paths:
        try:
            sims = json.load(open(p)).get("simulations", [])
        except Exception as e:
            print(f"[skip] {p}: {e}"); continue
        print(f"[score] {p}")
        print(f"        {json.dumps(score_sims(sims, fix_map, args.decomp, task_req), ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
