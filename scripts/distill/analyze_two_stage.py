#!/usr/bin/env python3
"""analyze_two_stage.py — deep per-cause analysis of two_stage_agent runs.

Separates the TWO axes the headline coverage% conflated:
  (A) TBox-step track    : does the planner emit valid, well-placed abstract steps?
  (B) ABox/resolver track: when the resolver fires, is the tool CORRECT (vs GT)?
  (C) transfer track     : how do (A),(B) change in-domain vs LODO?

Replays each saved trajectory OFFLINE through the resolver (no LLM, no eval cost):
for every assistant tool-call turn we recover the planner's Plan step + concrete
call, run the deterministic resolver on the accumulated read-state, and compare
BOTH the planner's tool and the resolver's tool against the task's GT action set.

GT actions: tau2 task.evaluation_criteria.actions[].name (the expected write calls).

Usage:
  python scripts/distill/analyze_two_stage.py --results <dir>/results.json \
      --domain telecom --ontology-domain telecom
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ontology_resolver import OntologyResolver, ObservedState
from score_fix_coverage import is_read

PLAN_PREFIX = "Plan:"

def parse_step(content):
    if not content:
        return None
    import re
    m = re.search(r"Plan:\s*([a-zA-Z_]+)", content)
    return m.group(1) if m else None

def gt_actions_for(task):
    """Return set of GT action tool names from a tau2 task dict."""
    out = set()
    ec = task.get("evaluation_criteria") or {}
    for a in (ec.get("actions") or []):
        n = a.get("name") or a.get("tool") or a.get("action_id")
        if n:
            out.add(n)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--ontology-domain", default=None)
    ap.add_argument("--induced-dir", default="reports/facet_rft_2026/phase4_distill/induced")
    ap.add_argument("--min-score", type=float, default=1.0)
    args = ap.parse_args()

    j = json.load(open(args.results))
    sims = j["simulations"]
    tasks = {t["id"]: t for t in j.get("tasks", [])}
    res = OntologyResolver(args.ontology_domain or args.domain,
                           induced_dir=args.induced_dir, min_score=args.min_score)

    # accumulators
    n_sims = len(sims)
    rewards = [s.get("reward_info", {}).get("reward", 0.0) or 0.0 for s in sims]
    pass1 = sum(1 for r in rewards if r and r >= 0.999) / max(n_sims, 1)

    step_emit = 0; tool_turns = 0; no_step = 0
    step_dist = defaultdict(int)
    # resolver replay
    det = 0; det_miss = 0
    by_step_det = defaultdict(int); by_step_miss = defaultdict(int)
    # correctness (only meaningful for write/action turns where GT is defined)
    planner_tool_in_gt = 0; planner_tool_turns_w = 0
    resolver_tool_in_gt = 0; resolver_decisive_turns = 0
    resolver_vs_planner_agree = 0; resolver_vs_planner_turns = 0
    # per-step det rate among ACTION steps only
    ACTION_STEPS = {"apply_targeted_fix", "apply_policy_action", "escalate_or_document"}

    for s in sims:
        task = tasks.get(s.get("task_id"), {})
        gt = gt_actions_for(task)
        obs = ObservedState()
        id2read = {}
        for m in s.get("messages", []):
            role = m.get("role")
            if role == "assistant":
                tcs = m.get("tool_calls") or []
                # record reads for state replay
                for tc in tcs:
                    nm = tc.get("name")
                    if nm and is_read(nm):
                        id2read[tc.get("id")] = nm
                if not tcs:
                    continue
                tool_turns += 1
                step = parse_step(m.get("content"))
                if step is None:
                    no_step += 1
                    continue
                step_emit += 1
                step_dist[step] += 1
                planner_tool = tcs[0].get("name")
                is_write_turn = planner_tool and not is_read(planner_tool)
                # planner correctness on write turns
                if is_write_turn and gt:
                    planner_tool_turns_w += 1
                    if planner_tool in gt:
                        planner_tool_in_gt += 1
                # resolver replay on current obs state
                rr = res.resolve(step, obs)
                decisive = rr is not None and rr.decisive and not rr.missing_args
                if decisive:
                    det += 1; by_step_det[step] += 1
                    resolver_decisive_turns += 1
                    if gt and rr.tool in gt:
                        resolver_tool_in_gt += 1
                    if planner_tool:
                        resolver_vs_planner_turns += 1
                        if rr.tool == planner_tool:
                            resolver_vs_planner_agree += 1
                else:
                    det_miss += 1; by_step_miss[step] += 1
            elif role == "tool":
                rd = id2read.get(m.get("id"))
                if rd is not None:
                    obs.update(rd, m.get("content"))

    def rate(a, b):
        return f"{a}/{b} = {100.0*a/max(b,1):.1f}%"

    print(f"=== {args.results} (ontology={args.ontology_domain or args.domain}) ===")
    print(f"n_sims={n_sims}  Pass^1={pass1:.3f}  mean_reward={sum(rewards)/max(n_sims,1):.3f}")
    print(f"--- TBox track (planner step emission) ---")
    print(f"  tool_turns={tool_turns}  step_emitted={rate(step_emit,tool_turns)}  no_step={no_step}")
    print(f"  step_dist={dict(step_dist)}")
    print(f"--- resolver coverage (ALL steps) ---")
    print(f"  deterministic={rate(det, det+det_miss)}")
    action_det = sum(by_step_det[s] for s in ACTION_STEPS)
    action_miss = sum(by_step_miss[s] for s in ACTION_STEPS)
    print(f"--- resolver coverage (ACTION steps only: {sorted(ACTION_STEPS)}) ---")
    print(f"  deterministic={rate(action_det, action_det+action_miss)}")
    for s in ACTION_STEPS:
        if by_step_det[s] or by_step_miss[s]:
            print(f"    {s:22} det={rate(by_step_det[s], by_step_det[s]+by_step_miss[s])}")
    print(f"--- correctness (vs GT actions) ---")
    print(f"  planner tool in GT  (write turns) = {rate(planner_tool_in_gt, planner_tool_turns_w)}")
    print(f"  resolver tool in GT (decisive)    = {rate(resolver_tool_in_gt, resolver_decisive_turns)}")
    print(f"  resolver==planner   (decisive)    = {rate(resolver_vs_planner_agree, resolver_vs_planner_turns)}")
    print()

if __name__ == "__main__":
    raise SystemExit(main())
