"""Per-task, per-step comparison of the tasks the front-32 arms already ran.

The full-97 sweep re-runs 32 tasks that A and B4 had run three days earlier on the same
stack, so those tasks answer a question the rest of the sweep cannot: is the drop a
harder task set, or the same tasks getting worse? Aggregates cannot say — the 2026-08-04
close-out measured flip 16/64 = 25% between two arms that scored identically, so any
single task changing hands is inside the noise. What is *not* inside the noise is whether
the failing step moved: a task that failed at the same step in all three arms is stable,
and one that fails at a new step is where something actually changed.

So for every shared task this prints, per arm and trial, the cause class and the decisive
step that x54 derives — same predicates, no new taxonomy ([[48]]) — side by side.
"""

import argparse
import collections
import glob
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x50_says_not_does import ARMS, SIM  # noqa: E402
from x54_perstep_by_cause import classify, report  # noqa: E402


def load(pattern):
    out = []
    for p in sorted(glob.glob(f"{SIM}/{pattern}.results.json.gz")):
        out.extend(json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or [])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B4,N97")
    ap.add_argument("--steps", action="store_true", help="print the decisive-step detail")
    ap.add_argument("--tasks", help="comma-separated subset")
    args = ap.parse_args()

    arms = args.arms.split(",")
    data = {}
    for a in arms:
        for s in load(ARMS[a]):
            data[(a, s["task_id"], s.get("trial"))] = s

    shared = None
    for a in arms:
        ts = {t for (arm, t, _) in data if arm == a}
        shared = ts if shared is None else shared & ts
    if args.tasks:
        shared &= set(args.tasks.split(","))
    shared = sorted(shared)

    print(f"공통 태스크 {len(shared)}개 · arm {arms}\n")
    tally = collections.Counter()
    moved, stable = [], []

    for t in shared:
        head = f"── {t} " + "─" * (66 - len(t))
        print(head)
        classes = {}
        for a in arms:
            for tr in (0, 1):
                s = data.get((a, t, tr))
                if s is None:
                    continue
                ok = (s.get("reward_info") or {}).get("reward") == 1.0
                if ok:
                    print(f"  {a:4} t{tr}  PASS")
                    classes.setdefault(a, []).append("PASS")
                    tally[(a, "PASS")] += 1
                    continue
                cls, missed = classify(s)
                classes.setdefault(a, []).append(cls)
                tally[(a, cls)] += 1
                key, lines = report(s, cls, missed)
                term = s.get("termination_reason")
                print(f"  {a:4} t{tr}  {cls}   [{term}]")
                if args.steps:
                    for ln in lines:
                        print("    " + ln.strip())
        old = set(classes.get("A", [])) | set(classes.get("B4", []))
        new = set(classes.get("N97", []))
        if new and old and not (new & old):
            moved.append((t, sorted(old), sorted(new)))
        elif new & old:
            stable.append(t)
        print()

    print("=" * 70)
    print("클래스 분포 (arm별)")
    for a in arms:
        row = [(c, v) for (arm, c), v in tally.most_common() if arm == a]
        print(f"  {a:4} " + ", ".join(f"{c}={v}" for c, v in row))
    print(f"\n같은 실패가 그대로 (클래스 겹침): {len(stable)}  {stable}")
    print(f"실패 지점이 바뀐 태스크: {len(moved)}")
    for t, o, n in moved:
        print(f"  {t}: {o}  ->  {n}")


if __name__ == "__main__":
    main()
