#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pair the LB arm against the base arm, task by task, and say what our layer did in each.

    python lb_pairs.py --base <dir-or-glob> --lb <dir-or-glob> [--sidecar <dir-or-glob>]

Only a task present in both arms is compared, and only reward counts: a task is p/n where n is the
number of simulations and p the number with reward 1.0. The noise floor is 18.8% (a base arm run
twice flips that often, n=16), so a one-simulation difference on a four-simulation task is inside it
and this script says so rather than calling it a change.

A-cell = the tasks base passed 4/4. If the LB arm drops one of those, that is our layer breaking
something the model could already do, and no statistics are needed to see it.
"""

import argparse
import collections
import glob
import gzip
import io
import json
import os

NOISE = "within the noise floor (a base arm run twice flips 18.8% of simulations)"


def expand(pattern):
    out = []
    for p in glob.glob(pattern) or ([pattern] if os.path.exists(pattern) else []):
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "*.results.json.gz")) + glob.glob(os.path.join(p, "*.results.json")))
        else:
            out.append(p)
    return out


def scores(patterns):
    """{task: (passes, sims)} over every results file given."""
    out = collections.defaultdict(lambda: [0, 0])
    for pattern in patterns:
        for path in expand(pattern):
            opener = gzip.open if path.endswith(".gz") else io.open
            try:
                with opener(path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            for sim in data.get("simulations") or []:
                row = out[sim.get("task_id")]
                row[1] += 1
                row[0] += 1 if (sim.get("reward_info") or {}).get("reward") == 1.0 else 0
    return {k: tuple(v) for k, v in out.items()}


def interventions(patterns):
    """{task: Counter(kind or engine)} from the LB sidecars."""
    out = collections.defaultdict(collections.Counter)
    for pattern in patterns:
        for path in expand(pattern) + [p for p in glob.glob(pattern) if "jsonl" in p]:
            if "jsonl" not in path:
                continue
            task = os.path.basename(path).replace("fb_lb_", "").split(".")[0]
            opener = gzip.open if path.endswith(".gz") else io.open
            try:
                with opener(path, "rt", encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        key = row.get("lb") or row.get("winner") or row.get("kind")
                        out[task][row.get("kind")] += 1
                        if row.get("kind") == "lb-deny":
                            out[task]["deny:" + str(key)] += 1
                        if row.get("kind") == "lb-conflict":
                            out[task]["conflict:" + str(key)] += 1
            except Exception:
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--lb", nargs="+", required=True)
    ap.add_argument("--sidecar", nargs="*", default=[])
    a = ap.parse_args()
    base, lb = scores(a.base), scores(a.lb)
    fb = interventions(a.sidecar) if a.sidecar else {}
    shared = sorted(set(base) & set(lb))
    if not shared:
        print("no task is present in both arms yet (base %d tasks, lb %d tasks)" % (len(base), len(lb)))
        return

    print("task      base    lb     delta  cell   our-layer interventions")
    up = down = same = 0
    acell_drop = []
    for t in shared:
        bp, bn = base[t]
        lp, ln = lb[t]
        d = lp - bp
        up += d > 0
        down += d < 0
        same += d == 0
        cell = "A" if bn and bp == bn else ("0" if bp == 0 else "-")
        if cell == "A" and lp < ln:
            acell_drop.append((t, "%d/%d" % (lp, ln)))
        marks = fb.get(t, {})
        note = " ".join("%s=%d" % (k.split(":")[-1], v) for k, v in sorted(marks.items()) if k.startswith("deny:")) \
            or ("advice=%d" % marks.get("lb-advice", 0) if marks else "")
        print("%-9s %d/%-4d %d/%-4d %+5d  %-5s  %s" % (t, bp, bn, lp, ln, d, cell, note))

    print("\npaired tasks %d: better %d, worse %d, unchanged %d, net %+d" % (len(shared), up, down, same, up - down))
    print("a one-simulation difference on a four-simulation task is %s" % NOISE)
    if acell_drop:
        print("\nA-cell drops (base passed every simulation, we did not) - our layer, not the model:")
        for t, s in acell_drop:
            print("  %s now %s" % (t, s))
    else:
        print("\nA-cell drops: none among the paired tasks so far")
    print("\nread the sidecar for any task above before naming a cause: the denies are not in the trajectory.")


if __name__ == "__main__":
    main()
