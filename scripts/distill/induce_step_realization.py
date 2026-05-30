#!/usr/bin/env python3
"""induce_step_realization.py — domain-general ABox map: concrete tool -> abstract plan step.

TBox-isolation method B (EXPERIMENT_DESIGN §15.10). The abstract PLAN_STEP_VOCAB
(identify_root_cause / apply_targeted_fix / verify_resolution / escalate_or_document ...)
is domain-INVARIANT; the per-domain ABox = which concrete tool realizes which step
(`step_realizes_tool`, Group I). This induces that map from a domain's tools by ROLE
(read vs write, GT-fix vs other, escalation) — NO domain hardcoding. Output:
induced/step_realization_<domain>.json = {tool: plan_step}. Consumed by
build_abstract_sft.py to inject "Plan: <step>" prefixes so the model learns the
domain-general plan-step policy while the concrete realization stays swappable.

Usage:
  python scripts/distill/induce_step_realization.py --domain telecom
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_fix_coverage import is_read, _shipped_files  # noqa: E402
from procedure_scorecard import gt_agent_actions  # noqa: E402

DEFAULT_TAU2 = "/home/woori/workspace_common/boltzmann-attention/external/tau2-bench"
ESCALATION_RE = re.compile(r"transfer|escalat|human|handoff|hand_off", re.I)
STATUS_RE = re.compile(r"status|network|signal|diagnos|check|service_state|connection", re.I)

# Abstract plan-step vocabulary (PLAN_STEP_VOCAB, design §3.2.2) — domain-invariant.
STEP_GATHER = "gather_account_context"
STEP_CHECK = "check_system_status"
STEP_FIX = "apply_targeted_fix"
STEP_POLICY = "apply_policy_action"
STEP_ESCALATE = "escalate_or_document"


def reward_of(s):
    return (s.get("reward_info") or {}).get("reward", 0) or 0


def collect_tools(domain, shipped_dir):
    """Union of all agent-side tool names observed in shipped trajectories."""
    tools = set()
    for f in _shipped_files(domain, shipped_dir):
        try:
            sims = json.load(open(f)).get("simulations", [])
        except Exception:
            continue
        for s in sims:
            for m in s.get("messages") or []:
                if m.get("role") == "assistant":
                    for tc in m.get("tool_calls") or []:
                        if tc.get("name") and tc.get("requestor", "assistant") == "assistant":
                            tools.add(tc["name"])
    return tools


def step_for(tool, gt_writes):
    """Domain-general role->step assignment."""
    if ESCALATION_RE.search(tool):
        return STEP_ESCALATE
    if is_read(tool):
        return STEP_CHECK if STATUS_RE.search(tool) else STEP_GATHER
    # write
    return STEP_FIX if tool in gt_writes else STEP_POLICY


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["telecom", "retail", "airline"])
    ap.add_argument("--tau2-root", default=DEFAULT_TAU2)
    ap.add_argument("--shipped-dir", default=DEFAULT_TAU2 + "/data/tau2/results/final")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tasks = json.load(open(os.path.join(args.tau2_root, "data", "tau2", "domains",
                                        args.domain, "tasks.json")))
    gt_writes = set()
    for t in tasks:
        for a in gt_agent_actions(t):
            gt_writes.add(a["name"])

    tools = collect_tools(args.domain, args.shipped_dir) | gt_writes
    realize = {t: step_for(t, gt_writes) for t in sorted(tools)}
    dist = Counter(realize.values())

    out = args.out or os.path.join(
        "reports/facet_rft_2026/phase4_distill/induced",
        f"step_realization_{args.domain}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"domain": args.domain, "n_tools": len(realize),
               "step_dist": dict(dist), "step_realizes_tool": realize},
              open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[{args.domain}] {len(realize)} tools -> steps {dict(dist)}")
    for t, s in list(realize.items())[:12]:
        print(f"    {t:34} -> {s}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
