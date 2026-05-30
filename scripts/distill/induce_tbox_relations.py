#!/usr/bin/env python3
"""induce_tbox_relations.py — domain-general ABox inducer for Group J TBox relations.

Group J (EXPERIMENT_DESIGN §15.4) are domain-INVARIANT relation TYPES grounded in the
data-driven failure analysis (§15.2). This script induces their per-domain ABox
INSTANCE maps from shipped teacher successes + benchmark GT + (optional) weak-model
failures. No domain (telecom/retail/airline) is hardcoded: read/write are split by
name prefix, "goal" is the task's GT-write signature, the escalation tool is
auto-detected. Output: induced/tbox_relations_<domain>.json.

  repairs_state(read, fix_tool)          read that diagnoses a blocking state -> the
                                         write whose effect repairs it (read immediately
                                         preceding a GT-fix write in teacher successes)
  diagnosis_sufficient_for(goal, reads)  minimal read set seen before the first write in
                                         teacher successes (commit boundary, anti-loop)
  distractor_for(goal, wrong_tool)       GT-absent write called in FAILED trajectories
                                         (contrastive negative goal->tool)
  escalate_when(escalation_tool, conds)  conditions (faults / co-required writes) of tasks
                                         whose GT requires the auto-detected handoff tool

Usage:
  python scripts/distill/induce_tbox_relations.py --domain telecom \
      --failure-results '<student results.json glob>' \
      --out induced/tbox_relations_telecom.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_fix_coverage import is_read, _shipped_files, parse_task  # noqa: E402
from procedure_scorecard import gt_agent_actions  # noqa: E402

DEFAULT_TAU2 = "/home/woori/workspace_common/boltzmann-attention/external/tau2-bench"
ESCALATION_RE = re.compile(r"transfer|escalat|human|handoff|hand_off|agent", re.I)


def ordered_calls(messages):
    """Ordered agent tool calls [(name, is_read)] for requestor==assistant."""
    out = []
    for m in messages or []:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                n = tc.get("name")
                if n and tc.get("requestor", "assistant") == "assistant":
                    out.append((n, is_read(n)))
    return out


def reward_of(s):
    return (s.get("reward_info") or {}).get("reward", 0) or 0


def load_sims(files):
    sims = []
    for f in files:
        try:
            sims += json.load(open(f)).get("simulations", [])
        except Exception as e:
            print(f"[warn] skip {f}: {e}")
    return sims


def goal_sig(gt_writes):
    """Domain-general 'goal' key = sorted GT write-tool signature."""
    return "+".join(sorted(set(gt_writes))) if gt_writes else "(none)"


def induce(domain, tau2_root, shipped_dir, failure_globs,
           min_support=5, min_frac=0.4):
    tasks = json.load(open(os.path.join(tau2_root, "data", "tau2", "domains",
                                        domain, "tasks.json")))
    gt_map = {t.get("id", ""): [a["name"] for a in gt_agent_actions(t)] for t in tasks}

    teacher_files = _shipped_files(domain, shipped_dir)
    t_sims = load_sims(teacher_files)
    t_succ = [s for s in t_sims if reward_of(s) >= 0.999]

    # ---- A) repairs_state: read immediately preceding a GT-fix write ----
    rs_pair = Counter()        # (read, fix) -> count
    fix_total = Counter()      # fix -> times appeared (with some preceding read)
    # ---- B) diagnosis_sufficient_for: reads before first write, per goal ----
    diag_reads = defaultdict(Counter)   # goal -> Counter(read)
    diag_first = defaultdict(Counter)   # goal -> Counter(first_write)
    diag_n = Counter()                  # goal -> n successes
    for s in t_succ:
        tid = s.get("task_id", "")
        gt = set(gt_map.get(tid, []))
        if not gt:
            continue
        calls = ordered_calls(s.get("messages") or [])
        g = goal_sig(gt)
        # B: reads before first write
        seen_reads, first_write = set(), None
        for n, rd in calls:
            if rd:
                if first_write is None:
                    seen_reads.add(n)
            else:
                if first_write is None:
                    first_write = n
        if first_write is not None:
            diag_n[g] += 1
            diag_first[g][first_write] += 1
            for r in seen_reads:
                diag_reads[g][r] += 1
        # A: last read before each GT-fix write occurrence
        last_read = None
        for n, rd in calls:
            if rd:
                last_read = n
            else:
                if n in gt:  # a real fix write
                    fix_total[n] += 1
                    if last_read is not None:
                        rs_pair[(last_read, n)] += 1
                last_read = None  # reset: only immediately-preceding read counts

    repairs_state = []
    for (r, w), c in rs_pair.items():
        if c >= min_support and fix_total[w] and c / fix_total[w] >= min_frac:
            repairs_state.append({"read": r, "fix_tool": w,
                                  "support": c, "frac": round(c / fix_total[w], 3)})
    repairs_state.sort(key=lambda d: -d["support"])

    diagnosis_sufficient_for = []
    for g, n in diag_n.items():
        if n < min_support:
            continue
        reads = [r for r, c in diag_reads[g].items() if c / n >= min_frac]
        commit = diag_first[g].most_common(1)[0][0] if diag_first[g] else None
        diagnosis_sufficient_for.append({"goal": g, "n": n,
                                         "sufficient_reads": sorted(reads),
                                         "then_commit": commit})
    diagnosis_sufficient_for.sort(key=lambda d: -d["n"])

    # ---- C) distractor_for: GT-absent writes in FAILED trajectories ----
    fail_files = []
    for gpat in failure_globs or []:
        fail_files += glob.glob(gpat)
    fail_sims = load_sims(fail_files) + [s for s in t_sims if reward_of(s) < 0.999]
    distr = defaultdict(Counter)   # goal -> Counter(wrong_tool)
    for s in fail_sims:
        tid = s.get("task_id", "")
        gt = set(gt_map.get(tid, []))
        if not gt:
            continue
        calls = ordered_calls(s.get("messages") or [])
        wrong = {n for n, rd in calls if not rd and n not in gt}
        g = goal_sig(gt)
        for w in wrong:
            distr[g][w] += 1
    distractor_for = []
    for g, ctr in distr.items():
        items = [{"wrong_tool": w, "count": c} for w, c in ctr.most_common()]
        if items:
            distractor_for.append({"goal": g, "distractors": items})
    distractor_for.sort(key=lambda d: -sum(i["count"] for i in d["distractors"]))

    # ---- D) escalate_when: conditions of tasks requiring the handoff tool ----
    all_gt_writes = Counter()
    for w in gt_map.values():
        all_gt_writes.update(set(w))
    esc_candidates = [t for t in all_gt_writes if ESCALATION_RE.search(t)]
    escalation_tool = max(esc_candidates, key=lambda t: all_gt_writes[t]) if esc_candidates else None
    esc_conditions = Counter()
    esc_cowrites = Counter()
    n_esc_tasks = 0
    if escalation_tool:
        for tid, gt in gt_map.items():
            if escalation_tool in set(gt):
                n_esc_tasks += 1
                _, faults = parse_task(tid)
                for f in faults:
                    esc_conditions[f] += 1
                for w in set(gt):
                    if w != escalation_tool:
                        esc_cowrites[w] += 1
    escalate_when = {
        "escalation_tool": escalation_tool,
        "n_tasks_requiring": n_esc_tasks,
        "fault_conditions": [{"fault": f, "count": c} for f, c in esc_conditions.most_common(20)],
        "co_required_writes": [{"tool": w, "count": c} for w, c in esc_cowrites.most_common(10)],
    }

    return {
        "domain": domain,
        "meta": {
            "n_teacher_sims": len(t_sims), "n_teacher_success": len(t_succ),
            "n_fail_sims": len(fail_sims), "n_tasks": len(gt_map),
            "min_support": min_support, "min_frac": min_frac,
            "fail_sources": fail_files,
        },
        "repairs_state": repairs_state,
        "diagnosis_sufficient_for": diagnosis_sufficient_for,
        "distractor_for": distractor_for,
        "escalate_when": escalate_when,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["telecom", "retail", "airline"])
    ap.add_argument("--tau2-root", default=DEFAULT_TAU2)
    ap.add_argument("--shipped-dir", default=DEFAULT_TAU2 + "/data/tau2/results/final")
    ap.add_argument("--failure-results", nargs="*", default=None,
                    help="glob(s) of weak/student results.json for distractor_for")
    ap.add_argument("--min-support", type=int, default=5)
    ap.add_argument("--min-frac", type=float, default=0.4)
    ap.add_argument("--out", default=None,
                    help="output json (default: induced/tbox_relations_<domain>.json)")
    args = ap.parse_args()

    res = induce(args.domain, args.tau2_root, args.shipped_dir,
                 args.failure_results, args.min_support, args.min_frac)

    out = args.out or os.path.join(
        "reports/facet_rft_2026/phase4_distill/induced",
        f"tbox_relations_{args.domain}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)

    m = res["meta"]
    print(f"[{args.domain}] teacher_sims={m['n_teacher_sims']} succ={m['n_teacher_success']} "
          f"fail_sims={m['n_fail_sims']} tasks={m['n_tasks']}")
    print(f"  repairs_state            : {len(res['repairs_state'])} pairs"
          + (f"  e.g. {res['repairs_state'][0]['read']}->{res['repairs_state'][0]['fix_tool']}"
             if res['repairs_state'] else ""))
    print(f"  diagnosis_sufficient_for : {len(res['diagnosis_sufficient_for'])} goals")
    print(f"  distractor_for           : {len(res['distractor_for'])} goals"
          + (f"  top wrong={res['distractor_for'][0]['distractors'][0]['wrong_tool']}"
             if res['distractor_for'] and res['distractor_for'][0]['distractors'] else ""))
    ew = res["escalate_when"]
    print(f"  escalate_when            : tool={ew['escalation_tool']} "
          f"n_tasks={ew['n_tasks_requiring']} conds={len(ew['fault_conditions'])}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
