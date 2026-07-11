#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""전수 실패 task-지도 — arm의 456 sims를 task 단위로 접어 실패 전량을 열거한다 ([[08]]).

ecomp_fail_census.py --dump 산출(per-sim jsonl) + results.gz(reward)를 받아:
  task_id | pass k/4 | 실패 trial별 bucket | 기지-표적 여부 | robust 등급
robust 등급: SYSTEMIC(0/4) > MOSTLY(1/4) > FLAKY(2-3/4). 대조 arm(--ref)이 있으면 ref pass도 병기.

usage: t5c_taskmap.py --dump comp_fail.jsonl --results <comp.gz> [--ref <floor.gz>] [--known 0,2,17,...]
"""
import argparse, gzip, json
from collections import defaultdict


def load_json(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def pass_by_task(results_path):
    sims = load_json(results_path)["simulations"]
    agg = defaultdict(lambda: [0, 0])  # tid -> [pass, total]
    for s in sims:
        r = (s.get("reward_info") or {}).get("reward")
        tid = str(s.get("task_id"))
        agg[tid][1] += 1
        if r is not None and r >= 1:
            agg[tid][0] += 1
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--ref", default=None, help="대조 arm results.gz (예: floor)")
    ap.add_argument("--known", default="0,2,17,28,40,46,47,61,69,92,95,101,103")
    a = ap.parse_args()

    known = set(a.known.split(","))
    agg = pass_by_task(a.results)
    ref = pass_by_task(a.ref) if a.ref else {}

    rows = defaultdict(list)  # tid -> [(trial, bucket, disamb)]
    for line in open(a.dump, encoding="utf-8"):
        r = json.loads(line)
        rows[str(r["task_id"])].append((r.get("trial"), r["bucket"], r.get("disamb")))

    fail_tasks = sorted(rows, key=lambda t: (agg[t][0], int(t)))
    n_sys = n_most = n_flaky = 0
    print("%-5s %-6s %-7s %-5s %-9s  %s" % ("task", "pass", "grade", "known", "ref", "fail buckets (trial:bucket[,d=disamb])"))
    for tid in fail_tasks:
        p, tot = agg[tid]
        grade = "SYS" if p == 0 else ("MOST" if p == 1 else "FLAKY")
        if p == 0:
            n_sys += 1
        elif p == 1:
            n_most += 1
        else:
            n_flaky += 1
        rp = ("%d/%d" % tuple(ref[tid])) if tid in ref else "-"
        bs = ",".join("%s:%s%s" % (tr, b, "[d]" if d else "") for tr, b, d in sorted(rows[tid]))
        print("%-5s %d/%-4d %-7s %-5s %-9s  %s" % (tid, p, tot, grade, "*" if tid in known else "", rp, bs))
    print("\ntasks failing: %d (SYSTEMIC %d · MOSTLY 1/4 %d · FLAKY %d) | known-targets among them: %d"
          % (len(fail_tasks), n_sys, n_most, n_flaky, sum(1 for t in fail_tasks if t in known)))
    new_sys = [t for t in fail_tasks if agg[t][0] == 0 and t not in known]
    print("NEW systemic (0/4, not in known list): %s" % ",".join(new_sys))
    new_most = [t for t in fail_tasks if agg[t][0] == 1 and t not in known]
    print("NEW mostly-fail (1/4, not in known):   %s" % ",".join(new_most))


if __name__ == "__main__":
    main()
