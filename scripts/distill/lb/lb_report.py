#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fold LB sidecars into three tables: denies per engine, every conflict, and the intervention
ledger - one row per simulation naming everything our layer did that base did not.

  python lb_report.py fb_rep3_task_048.jsonl [more.jsonl or .jsonl.gz ...]

The ledger is the base comparison. Base runs none of this, so its row would be zero across the
board; ours is the difference, and every column is a place where the two arms are not the same
experiment. Columns, in the order the model meets them:

  tools    verifier tools added to the model's tool list before the first message
  fold     turns where the view handed to the model was not the conversation
  ask      extra model calls we made (the claims audit)
  tool     our verifier tools the model actually called, and our answers it read
  inject   content we put into the view
  advice   sentences we added, in the customer's slot
  deny     tool results we replaced with a refusal
  regen    messages of the model's that we threw away and had it write again
  stop     regenerations we stopped because the budget ran out
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


COLUMNS = [("tools", "lb-tools"), ("fold", "lb-fold"), ("ask", "lb-ask"), ("tool", "lb-tool"),
           ("inject", "lb-inject"), ("advice", "lb-advice"), ("deny", "lb-deny"),
           ("regen", "lb-regen"), ("stop", "lb-regen-stop")]


def ledger(counts):
    """One row per simulation. Base would be zero in every column - that is the point."""
    if not counts:
        return
    print("")
    print("intervention ledger, one row per simulation (base would be zero across the board)")
    print("  %-38s %s" % ("sim", " ".join("%6s" % n for n, _ in COLUMNS)))
    for sim in sorted(counts):
        c = counts[sim]
        print("  %-38s %s" % (sim, " ".join("%6d" % c[k] for _, k in COLUMNS)))


def main(paths):
    denies, conflicts = collections.Counter(), collections.Counter()
    per_sim = collections.defaultdict(collections.Counter)
    for p, r in rows(paths):
        if r.get("sim"):
            per_sim[r["sim"]][r.get("kind")] += 1
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
    ledger(per_sim)


if __name__ == "__main__":
    main(sys.argv[1:])
