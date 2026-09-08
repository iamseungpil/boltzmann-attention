#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Watch a running LB lane and name the trouble while it is still cheap to fix.

    python lb_watch.py [--logs /home/woori/scratch/logs] [--minutes 90]

Reads the live sidecars and driver logs - the sidecar is the only record of what our layer did,
because a deny replaces the message and never reaches the trajectory. Every alert below is a
pattern that has already cost a run:

  runaway      one rule denying the same target over and over: the deny reproduces its own cause
  gold-risk    a rule other than the procedure walker blocking a declared write tool
  inversion    a conflict where the loser rests on much stronger evidence. A gap of one is
               ordinary: the procedure walker quotes policy (E2) and outranks a ledger fact
               (E1) when both name the same step, and the two merge into one sentence.
  flood        one rule advising more often than its per-simulation budget
  silent       an engine that has not fired once this far into the run
  stuck        a simulation with no new sidecar row and no new log line for a long time
  crash        a traceback, or our own engine reporting that it failed

Exit code is 1 when any alert fired, so a watchdog can act on it.
"""

import argparse
import collections
import glob
import io
import json
import os
import sys
import time

RUNAWAY = 5           # same rule, same target, same simulation
FLOOD = 3             # one rule advising in one simulation (the budget is 2)
STUCK_MIN = 12        # minutes without a new line while the driver is still running


def rows(path):
    out = []
    try:
        with io.open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
    except IOError:
        pass
    return out


def watch(logs, a2, minutes):
    alerts, seen_engines = [], set()
    writes = {w for s in (a2.get("LB4") or {}).get("sets") or [] for w in s.get("write_tools") or []}
    finals = {w for s in (a2.get("LB4") or {}).get("sets") or [] for w in s.get("finalize_writes") or []}
    guarded = writes | finals
    now = time.time()

    for fb in sorted(glob.glob(os.path.join(logs, "fb_lb_*.jsonl"))):
        task = os.path.basename(fb)[6:-6]
        drv = os.path.join(logs, "%s_drv.log" % task)
        data = rows(fb)
        if not data:
            continue
        age = (now - os.path.getmtime(fb)) / 60.0
        deny = collections.Counter()
        advice = collections.Counter()
        for r in data:
            sim = r.get("sim", "-")
            if r.get("kind") == "lb-deny":
                deny[(sim, r.get("source"), r.get("target"))] += 1
                seen_engines.add(r.get("lb"))
                if r.get("target") in guarded and not str(r.get("source") or "").startswith("procedure"):
                    alerts.append(("gold-risk", task, "%s denies the declared write %s (sim %s)"
                                   % (r.get("source"), r.get("target"), sim)))
            elif r.get("kind") == "lb-advice":
                advice[(sim, r.get("source"))] += 1
            elif r.get("kind") == "lb-conflict":
                if r.get("loser_grade") is not None and r.get("winner_grade") is not None \
                        and r["winner_grade"] - r["loser_grade"] >= 2:
                    alerts.append(("inversion", task, "%s beat %s on %s though the loser rests on "
                                   "stronger evidence" % (r.get("winner"), r.get("loser"), r.get("target"))))
        for (sim, source, target), n in deny.items():
            if n >= RUNAWAY:
                alerts.append(("runaway", task, "%s denied %s %d times in simulation %s"
                               % (source, target, n, sim)))
        for (sim, source), n in advice.items():
            if n > FLOOD and sim != "-":
                alerts.append(("flood", task, "%s advised %d times in simulation %s" % (source, n, sim)))

        if any(sim == "-" for sim, _s in advice) and data:
            alerts.append(("no-sim-id", task, "rows carry no simulation id, so per-simulation counts "
                                              "are blind: this run started before that was wired"))
        text = io.open(drv, encoding="utf-8", errors="replace").read() if os.path.exists(drv) else ""
        if "Traceback" in text:
            alerts.append(("crash", task, "traceback in the driver log"))
        for line in text.splitlines():
            if "failed (empty)" in line or "] error" in line.lower():
                alerts.append(("crash", task, line.strip()[:120]))
        running = "complete" in text and text.rstrip().endswith(")") or "running:" in text[-2000:]
        if running and age > STUCK_MIN:
            alerts.append(("stuck", task, "no sidecar row for %.0f minutes" % age))

    for lb in ("LB1", "LB2", "LB3", "LB4", "LB5", "LB7"):
        if lb not in seen_engines and minutes >= 30:
            alerts.append(("silent", "-", "%s has not fired in %d minutes of running" % (lb, minutes)))
    return alerts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="/home/woori/scratch/logs")
    ap.add_argument("--a2", default=None)
    ap.add_argument("--minutes", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import lb_a2
    a2 = lb_a2.load("banking_knowledge") or {}
    alerts = watch(a.logs, a2, a.minutes)
    seen = set()
    for kind, task, detail in alerts:
        key = (kind, task, detail[:60])
        if key in seen:
            continue
        seen.add(key)
        print("%-9s %-16s %s" % (kind, task, detail))
    print("%d alert(s)" % len(seen) if seen else "no alerts")
    sys.exit(1 if seen else 0)


if __name__ == "__main__":
    main()
