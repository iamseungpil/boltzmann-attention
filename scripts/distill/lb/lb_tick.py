#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""One line per task per sample: where each simulation is and what our layer has done so far.

    python lb_tick.py [--logs DIR] [--tasks 048 049]

Written for a per-minute loop (lb_ctl.sh tick), so the line is short and the columns never move:

    11:07 task_048 done=1/4 run=.2(320s) deny=7 adv=18 conf=2 last=procedure:credit_card_closure
"""

import argparse
import collections
import glob
import io
import json
import os
import time


def sidecar(path):
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


def tail_status(path):
    """The driver's own progress line, and how long the current simulation has been going."""
    try:
        text = io.open(path, encoding="utf-8", errors="replace").read()
    except IOError:
        return "", ""
    done, running = "", ""
    for line in text.splitlines():
        if "complete." in line and "Status:" in line:
            done = line.split("Status:")[1].split("complete")[0].strip()
            running = line.split("running:")[1].strip() if "running:" in line else ""
    return done, running[:22]


def line(logs, task, now):
    fb = os.path.join(logs, "fb_lb_%s.jsonl" % task)
    drv = os.path.join(logs, "lb_%s_drv.log" % task)      # the tag is lb_<task>, the sidecar is fb_lb_<task>
    rows = sidecar(fb)
    kinds = collections.Counter(r.get("kind") for r in rows)
    last = ""
    for r in reversed(rows):
        if r.get("kind") == "lb-deny":
            last = "%s>%s" % (str(r.get("source"))[:26], str(r.get("target"))[:20])
            break
    done, running = tail_status(drv)
    age = (now - os.path.getmtime(fb)) / 60.0 if os.path.exists(fb) else -1
    return "%s %s done=%-4s run=%-22s deny=%-3d adv=%-3d conf=%-2d quiet=%.0fm %s" % (
        time.strftime("%H:%M", time.localtime(now)), task, done or "0/?", running or "-",
        kinds.get("lb-deny", 0), kinds.get("lb-advice", 0), kinds.get("lb-conflict", 0),
        max(age, 0), last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="/home/woori/scratch/logs")
    ap.add_argument("--tasks", nargs="*", default=None)
    a = ap.parse_args()
    tasks = a.tasks or sorted(os.path.basename(p)[6:-6] for p in glob.glob(os.path.join(a.logs, "fb_lb_*.jsonl")))
    now = time.time()
    for t in tasks:
        print(line(a.logs, t, now))


if __name__ == "__main__":
    main()
