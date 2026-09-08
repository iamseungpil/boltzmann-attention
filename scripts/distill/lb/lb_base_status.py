#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Where the base arm stands, and what its finished set does and does not license.

    python lb_base_status.py <results dir or glob ...> [--queue q_cbase.txt] [--denominator 96]

Prints the pass distribution over finished tasks, the same figures split by task-id range, and a
projection for the tasks still queued. The split is there because the finished set is not a sample:
the lane walks the task ids in order, so "what we have" and "what is left" differ by construction.
The projection is therefore given as a range built from the ranges already finished, not as a number.
"""

import argparse
import collections
import glob
import gzip
import io
import json
import os


def load(path):
    opener = gzip.open if path.endswith(".gz") else io.open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def collect(patterns):
    """{task: (passes, sims)} from results files or simulation directories."""
    out = collections.defaultdict(lambda: [0, 0])
    for pattern in patterns:
        paths = glob.glob(pattern)
        for p in paths:
            if os.path.isdir(p):
                p = os.path.join(p, "results.json")
            if not os.path.exists(p):
                continue
            try:
                data = load(p)
            except Exception:
                continue
            for sim in data.get("simulations") or []:
                row = out[sim.get("task_id")]
                row[1] += 1
                row[0] += 1 if (sim.get("reward_info") or {}).get("reward") == 1.0 else 0
    return {k: tuple(v) for k, v in out.items() if k}


def number(task):
    tail = str(task).split("_")[-1]
    return int(tail) if tail.isdigit() else -1


def band(task):
    n = number(task)
    return "001-058" if n <= 58 else ("059-078" if n <= 78 else "079-101")


def summarise(scores, label):
    sims = sum(n for _p, n in scores.values())
    passes = sum(p for p, _n in scores.values())
    full = sum(1 for p, n in scores.values() if n and p == n)
    zero = sum(1 for p, _n in scores.values() if p == 0)
    rate = passes / float(sims) if sims else 0.0
    print("%-10s tasks %3d  sims %4d  pass %4d  = %5.1f%%   all-four %3d   none %3d"
          % (label, len(scores), sims, passes, 100 * rate, full, zero))
    return rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+")
    ap.add_argument("--queue", default=None)
    ap.add_argument("--denominator", type=int, default=96)
    ap.add_argument("--known-fail", nargs="*", dest="known_fail", default=None,
                    help="tasks the record already shows failing; counted as zero, never from a prior")
    ap.add_argument("--exclude", nargs="*", default=None, help="tasks outside the denominator")
    ap.add_argument("--prior", nargs="*", default=None,
                    help="an earlier run on the same model, used as a per-task difficulty estimate")
    a = ap.parse_args()
    scores = collect(a.patterns)
    for t in (a.exclude or ()):
        scores.pop(t, None)
    if not scores:
        print("nothing found")
        return
    print("finished")
    overall = summarise(scores, "all")
    bands = collections.defaultdict(dict)
    for t, v in scores.items():
        bands[band(t)][t] = v
    rates = {}
    for b in sorted(bands):
        rates[b] = summarise(bands[b], b)

    left = []
    if a.queue and os.path.exists(a.queue):
        left = [l.strip() for l in io.open(a.queue, encoding="utf-8") if l.strip()
                if l.strip() not in set(a.exclude or ())]
    print("\nqueued %d: %s" % (len(left), " ".join(left[:8]) + (" ..." if len(left) > 8 else "")))
    known = [r for b, r in rates.items() if bands[b]]
    lo, hi = (min(known), max(known)) if known else (overall, overall)
    done_sims = sum(n for _p, n in scores.values())
    done_pass = sum(p for p, _n in scores.values())
    rest = len(left) * 4
    print("\nprojection over %d tasks, if the rest behaves like the finished ranges (%.1f%% to %.1f%%):"
          % (a.denominator, 100 * lo, 100 * hi))
    for name, r in (("low", lo), ("mid", overall), ("high", hi)):
        print("   %-5s %5.1f%%  (%d of %d simulations)"
              % (name, 100 * (done_pass + rest * r) / float(done_sims + rest),
                 int(done_pass + rest * r), done_sims + rest))
    print("\nthe finished set is ordered, not sampled: the lane walks task ids in order, so this is a")
    print("range from the ranges already done, not a confidence interval.")

    if a.prior:
        prior = collect(a.prior)
        for t in (a.exclude or ()):
            prior.pop(t, None)
        both = sorted(set(prior) & set(scores))
        if not both:
            print("\nprior run shares no task with this one")
            return
        pn = sum(prior[t][1] for t in both)
        pp = sum(prior[t][0] for t in both)
        cn = sum(scores[t][1] for t in both)
        cp = sum(scores[t][0] for t in both)
        prate, crate = pp / float(pn), cp / float(cn)
        agree = sum(1 for t in both if (prior[t][0] > 0) == (scores[t][0] > 0))
        print("\nprior as a difficulty estimate: %d tasks shared" % len(both))
        print("   prior  %5.1f%% (%d of %d)   this run %5.1f%% (%d of %d)   ratio %.2f"
              % (100 * prate, pp, pn, 100 * crate, cp, cn, (crate / prate) if prate else 0))
        print("   the two agree on whether a task passes at all in %d of %d tasks" % (agree, len(both)))
        cal = (crate / prate) if prate else 1.0
        known_fail = set(a.known_fail or ())
        seen = [t for t in left if t in prior and t not in known_fail]
        blind = [t for t in left if t not in prior and t not in known_fail]
        expect = sum(min(1.0, cal * prior[t][0] / float(prior[t][1])) for t in seen) * 4
        if known_fail & set(left):
            print("\n   taken as failing on the record: %s" % " ".join(sorted(known_fail & set(left))))
        for label, fill_rate in (("prior only", 0.0), ("prior + this run's rate for the rest", overall)):
            total = done_pass + expect + len(blind) * 4 * fill_rate
            print("   %-38s %5.1f%%  (%d of %d simulations)"
                  % (label, 100 * total / float(done_sims + rest), int(total), done_sims + rest))
        print("   %d queued tasks are in the prior, %d are not: %s"
              % (len(seen), len(blind), " ".join(blind) or "-"))


if __name__ == "__main__":
    main()
