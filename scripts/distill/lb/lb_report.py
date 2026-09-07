#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fold LB sidecars into two tables: denies per engine, and every conflict (winner vs losers).

  python lb_report.py fb_rep3_task_048.jsonl [more.jsonl or .jsonl.gz ...]
"""

import collections
import gzip
import io
import json
import sys


def rows(paths):
    for p in paths:
        opener = gzip.open if p.endswith(".gz") else io.open
        with opener(p, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    yield p, json.loads(line)
                except Exception:
                    pass


def main(paths):
    denies, conflicts = collections.Counter(), collections.Counter()
    for p, r in rows(paths):
        if r.get("kind") == "lb-deny":
            denies[(p, r.get("lb"), r.get("source"))] += 1
        elif r.get("kind") == "lb-conflict":
            conflicts[(p, r.get("target"), r.get("winner"), r.get("text", "").split("losers=")[-1])] += 1
    print("denies per engine")
    for (p, lb, src), n in sorted(denies.items()):
        print("  %-40s %-4s %-24s %d" % (p[-40:], lb, src, n))
    print("conflicts (target | winner | losers | count)")
    for (p, t, w, l), n in sorted(conflicts.items()):
        print("  %-40s %-30s %-4s %-30s %d" % (p[-40:], t, w, l, n))


if __name__ == "__main__":
    main(sys.argv[1:])
