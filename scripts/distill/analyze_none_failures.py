#!/usr/bin/env python3
"""analyze_none_failures.py — per-failure breakdown of a distill-arm eval vs teacher.

Given a student results.json (phase1_runner output) on a tau2 domain, this:
  1. classifies each sim pass/fail (reward >= 0.999),
  2. per sim computes the GT agent WRITE-actions, the student's write calls,
     MISSING (recall gap), EXTRA (precision loss / db pollution), termination
     reason, tool-error count, and #assistant steps,
  3. tallies failure modes,
  4. ranks the most-frequently MISSED GT write-tools across failures,
  5. diffs against the shipped TEACHERS (multi-teacher final/, reward==1.0) on the
     SAME task_ids: which tools teachers call that the student omits, and mean
     #writes teacher-success vs student-failure.

The goal: find the concrete levers to close the gap to teacher level.

Usage:
  python scripts/distill/analyze_none_failures.py --domain telecom \
      --results '<...>/NONE_telecom_test.json/results.json'
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_fix_coverage import is_read, _shipped_files, parse_task  # noqa: E402
from procedure_scorecard import gt_agent_actions  # noqa: E402

DEFAULT_TAU2 = "/home/woori/workspace_common/boltzmann-attention/external/tau2-bench"


def agent_write_seq(messages):
    """Ordered agent WRITE tool calls (requestor==assistant, non-read)."""
    out = []
    for m in messages or []:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                n = tc.get("name")
                if n and tc.get("requestor", "assistant") == "assistant" and not is_read(n):
                    out.append(n)
    return out


def tool_error_count(messages):
    return sum(1 for m in (messages or [])
               if m.get("role") == "tool" and m.get("error"))


def n_assistant_steps(messages):
    return sum(1 for m in (messages or []) if m.get("role") == "assistant")


def reward_of(s):
    return (s.get("reward_info") or {}).get("reward", 0) or 0


def load_teacher_succ(domain, shipped_dir):
    """task_id -> list of (teacher_label, [write tool seq]) for reward==1.0 shipped sims."""
    succ = defaultdict(list)
    n_files = 0
    for f in _shipped_files(domain, shipped_dir):
        n_files += 1
        teacher = os.path.basename(f).split(f"_{domain}_")[0]
        try:
            sims = json.load(open(f)).get("simulations", [])
        except Exception:
            continue
        for s in sims:
            if reward_of(s) < 0.999:
                continue
            tid = s.get("task_id", "")
            succ[tid].append((teacher, agent_write_seq(s.get("messages") or [])))
    return succ, n_files


def classify_failure(gt_set, missing, extra, errs, term, steps):
    t = str(term or "").lower()
    if errs >= 2:
        return "C_tool_errors"
    if gt_set and len(missing) / len(gt_set) >= 0.5:
        return "A_recall_miss_major"
    if missing:
        return "A_recall_miss_minor"
    if "max" in t or steps >= 40:
        return "B_long_horizon"
    if extra:
        return "D_over_action"
    return "E_args_order_state"  # all GT tools called, no extras -> wrong args/order/state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["telecom", "retail", "airline"])
    ap.add_argument("--results", required=True, help="student results.json path/glob")
    ap.add_argument("--tau2-root", default=DEFAULT_TAU2)
    ap.add_argument("--shipped-dir", default=DEFAULT_TAU2 + "/data/tau2/results/final")
    ap.add_argument("--max-detail", type=int, default=40,
                    help="max per-failure detail lines to print")
    args = ap.parse_args()

    tasks = json.load(open(os.path.join(args.tau2_root, "data", "tau2", "domains",
                                         args.domain, "tasks.json")))
    gt_map = {t.get("id", ""): [a["name"] for a in gt_agent_actions(t)] for t in tasks}

    teacher_succ, n_tfiles = load_teacher_succ(args.domain, args.shipped_dir)
    print(f"[teacher] {n_tfiles} shipped files, {len(teacher_succ)} task_ids with >=1 success")

    paths = sorted(glob.glob(args.results))
    if not paths:
        print(f"[err] no results matched {args.results}")
        return 1
    sims = []
    for p in paths:
        sims += json.load(open(p)).get("simulations", [])
    print(f"[student] {len(sims)} sims from {len(paths)} file(s)\n")

    rows = []
    for s in sims:
        tid = s.get("task_id", "")
        gt = gt_map.get(tid, [])
        gt_set = set(gt)
        sseq = agent_write_seq(s.get("messages") or [])
        scalled = set(sseq)
        missing = gt_set - scalled
        extra = scalled - gt_set
        errs = tool_error_count(s.get("messages") or [])
        term = s.get("termination_reason", "")
        steps = n_assistant_steps(s.get("messages") or [])
        passed = reward_of(s) >= 0.999
        rows.append(dict(tid=tid, gt=gt, gt_set=gt_set, sseq=sseq, scalled=scalled,
                         missing=missing, extra=extra, errs=errs, term=term,
                         steps=steps, passed=passed))

    n_pass = sum(1 for r in rows if r["passed"])
    scored = [r for r in rows if r["gt_set"]]  # tasks with agent-required write actions
    fails = [r for r in scored if not r["passed"]]
    print(f"=== OVERVIEW === sims={len(rows)} pass={n_pass} (pass^1={n_pass/max(len(rows),1):.3f}) "
          f"| scored(with GT writes)={len(scored)} fails_scored={len(fails)}\n")

    # failure-mode tally
    modes = Counter()
    for r in fails:
        r["mode"] = classify_failure(r["gt_set"], r["missing"], r["extra"], r["errs"],
                                     r["term"], r["steps"])
        modes[r["mode"]] += 1
    print("=== FAILURE MODES (scored fails) ===")
    for m, c in modes.most_common():
        print(f"  {m:22} {c}")

    # most-missed GT tools across fails
    miss_ctr = Counter()
    for r in fails:
        miss_ctr.update(r["missing"])
    print("\n=== MOST-MISSED GT WRITE-TOOLS (across fails) ===")
    for tool, c in miss_ctr.most_common(15):
        print(f"  {tool:32} missed in {c} fails")

    # extra (over-action) tools across fails
    extra_ctr = Counter()
    for r in fails:
        extra_ctr.update(r["extra"])
    if extra_ctr:
        print("\n=== EXTRA (over-action) WRITE-TOOLS across fails ===")
        for tool, c in extra_ctr.most_common(12):
            print(f"  {tool:32} extra in {c} fails")

    # teacher diff: tools teachers call (in their successes) that student missed on same task
    teach_gap = Counter()
    tw, sw = [], []
    no_teacher = 0
    for r in fails:
        tsucc = teacher_succ.get(r["tid"], [])
        if not tsucc:
            no_teacher += 1
            continue
        # union of teacher write-calls across their successful trajectories
        tunion = set()
        twrites = []
        for _, seq in tsucc:
            tunion |= set(seq)
            twrites.append(len([x for x in seq]))
        tw.append(st.mean(twrites) if twrites else 0)
        sw.append(len(r["sseq"]))
        teach_gap.update(tunion - r["scalled"])
    print(f"\n=== TEACHER DIFF (on the SAME failed tasks; {len(fails)-no_teacher} have teacher successes,"
          f" {no_teacher} none) ===")
    if tw:
        print(f"  mean #writes  teacher-success={st.mean(tw):.2f}  student-fail={st.mean(sw):.2f}")
    print("  tools TEACHERS call (in success) that STUDENT omitted on the same task:")
    for tool, c in teach_gap.most_common(15):
        print(f"    {tool:32} in {c} failed tasks")

    # per-failure detail
    print(f"\n=== PER-FAILURE DETAIL (up to {args.max_detail}) ===")
    for r in fails[:args.max_detail]:
        issue, faults = parse_task(r["tid"])
        tsucc = teacher_succ.get(r["tid"], [])
        tex = ""
        if tsucc:
            # pick the shortest successful teacher write-seq as the canonical reference
            best = min(tsucc, key=lambda x: len(x[1]))
            tex = f" | TEACHER({best[0]}): {best[1]}"
        print(f"\n  [{r['mode']}] {issue} faults={faults} steps={r['steps']} errs={r['errs']} term={r['term']}")
        print(f"    GT_writes : {r['gt']}")
        print(f"    student   : {r['sseq']}")
        print(f"    MISSING   : {sorted(r['missing'])}  EXTRA: {sorted(r['extra'])}{tex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
