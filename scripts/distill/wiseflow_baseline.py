#!/usr/bin/env python3
"""wiseflow_baseline.py — WISE-Flow-style PROMPT-SIDE baseline (comparator for §13).

Induces, from shipped contrastive trajectories, a per-issue-type workflow:
  - global "identify" reads (precede everything in successes)
  - ordered milestones (action tools by mean first-occurrence)
  - fault -> fix-tool map (goal->tool, from induce_fault_fix)
  - per-action scenario prerequisites (action tools that precede an action in successes)
and builds a prompt-side injection block for a given task (goal-conditioned: only the
present faults' fixes). This is the PROMPT-SIDE comparator that our internalization
(weights) must beat on pass^1 AND on token/KV cost (it pays the workflow tokens every
turn). cf. WISE-Flow (arXiv 2601.08158): prereq-augmented workflow, retrieval+inject,
no weight change.

NOTE: this reproduces WISE-Flow's *mechanism* (induce → retrieve → inject) at the
issue-type granularity from our shipped multi-teacher successes. Full agent eval =
augment phase1_runner's agent system prompt with build_workflow_prompt(task_id) per
episode (see INTEGRATION below) — that step needs GPU + OpenRouter (execution stage).

Usage:
  python scripts/distill/wiseflow_baseline.py --domain telecom \
     --emit workflows_telecom.json --preview-issue mms_issue
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_fix_coverage import (  # noqa: E402
    parse_task, agent_tool_seq, is_read, induce_fault_fix, _shipped_files,
)


def induce_workflows(domain, shipped_dir, min_issue_support=10):
    fix_map = induce_fault_fix(domain, shipped_dir)
    by_issue_seqs = defaultdict(list)        # issue -> [tool seq]
    for f in _shipped_files(domain, shipped_dir):
        for s in json.load(open(f)).get("simulations", []):
            if (s.get("reward_info") or {}).get("reward", 0) < 0.999:
                continue
            issue, _ = parse_task(s.get("task_id", ""))
            by_issue_seqs[issue].append(agent_tool_seq(s.get("messages") or []))

    workflows = {}
    for issue, seqs in by_issue_seqs.items():
        n = len(seqs)
        if n < min_issue_support:
            continue
        # first-occurrence index per tool per seq
        firsts = defaultdict(list)
        present = Counter()
        for seq in seqs:
            seen = {}
            for i, t in enumerate(seq):
                seen.setdefault(t, i)
            for t, idx in seen.items():
                firsts[t].append(idx)
                present[t] += 1
        # global reads: read tools present in >=60% successes
        reads = sorted([t for t in present if is_read(t) and present[t] / n >= 0.6],
                       key=lambda t: sum(firsts[t]) / len(firsts[t]))
        # milestones: action tools present in >=30%, ordered by mean first-occurrence
        actions = [t for t in present if not is_read(t) and present[t] / n >= 0.30]
        actions.sort(key=lambda t: sum(firsts[t]) / len(firsts[t]))
        # per-action scenario prerequisites: action tools preceding A in >=70% of
        # successes where both appear
        prereq = {}
        for a in actions:
            pres_a = [seq for seq in seqs if a in seq]
            cand = Counter()
            for seq in pres_a:
                ia = seq.index(a)
                for b in set(seq[:ia]):
                    if not is_read(b):
                        cand[b] += 1
            prereq[a] = [b for b, c in cand.items() if c / max(len(pres_a), 1) >= 0.70 and b != a]
        workflows[issue] = {
            "support": n, "reads": reads, "milestones": actions,
            "prerequisites": {a: prereq[a] for a in actions if prereq[a]},
        }
    return workflows, fix_map


def build_workflow_prompt(task_id, workflows, fix_map):
    """Goal-conditioned prompt-side block for one task (only its present faults)."""
    issue, faults = parse_task(task_id)
    wf = workflows.get(issue)
    lines = [f"## Suggested resolution workflow ({issue})"]
    if wf and wf["reads"]:
        lines.append(f"1. Identify the account first: {', '.join(wf['reads'])}.")
    lines.append("2. Resolve each detected issue with its tool (satisfy prerequisites first):")
    any_fix = False
    for flt in faults:
        fx = fix_map.get(flt)
        if not fx:
            continue
        any_fix = True
        pre = (wf or {}).get("prerequisites", {}).get(fx, [])
        pre_s = f"  [after: {', '.join(pre)}]" if pre else ""
        lines.append(f"   - {flt}: call `{fx}`{pre_s}")
    if not any_fix:
        lines.append("   - (no known fix tool for the detected faults; diagnose, then escalate if unresolved)")
    lines.append("3. Verify the fix, then confirm or escalate. Always meet a tool's prerequisites before calling it.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["telecom", "retail", "airline"])
    ap.add_argument("--shipped-dir",
                    default="/home/woori/workspace_common/boltzmann-attention/"
                            "external/tau2-bench/data/tau2/results/final")
    ap.add_argument("--emit", default=None, help="write workflows JSON")
    ap.add_argument("--preview-issue", default=None, help="print workflow for an issue type")
    ap.add_argument("--preview-task", default=None, help="print injected prompt for a task_id")
    args = ap.parse_args()

    workflows, fix_map = induce_workflows(args.domain, args.shipped_dir)
    print(f"[wiseflow] {args.domain}: {len(workflows)} issue-type workflows, "
          f"{len(fix_map)} fault->fix")
    if args.emit:
        json.dump({"domain": args.domain, "fault_fix": fix_map, "workflows": workflows},
                  open(args.emit, "w"), indent=2, ensure_ascii=False)
        print(f"[wiseflow] wrote -> {args.emit}")
    for issue, wf in sorted(workflows.items(), key=lambda kv: -kv[1]["support"]):
        print(f"  [{issue}] n={wf['support']} reads={wf['reads']} milestones={wf['milestones']}")
    if args.preview_issue and args.preview_issue in workflows:
        print(f"\n=== workflow {args.preview_issue} ===")
        print(json.dumps(workflows[args.preview_issue], indent=2, ensure_ascii=False))
    if args.preview_task:
        print(f"\n=== injected prompt for task ===\n{build_workflow_prompt(args.preview_task, workflows, fix_map)}")
    return 0


# INTEGRATION (full agent eval, execution stage):
#   In phase1_runner.py, prepend build_workflow_prompt(task_id, workflows, fix_map) to
#   the agent SYSTEM prompt (after the policy) per episode. Agent = base model (no SFT).
#   Compare pass^1/fix-coverage + per-turn token cost vs the internalized (none-mode SFT)
#   student. This is the prompt-side vs weights efficiency contrast (§13.7).

if __name__ == "__main__":
    sys.exit(main())
