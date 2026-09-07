#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x771 — T2_PROCEDURE_LEFT 가 **어느 자리에서** 터지나 (2026-09-05). 센다. 판정 0."""
import io, os, re, sys
from pathlib import Path
from loguru import logger
logger.remove()
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass
from tau2.data_model.simulation import Results   # noqa: E402
import gate_interpreter as _GI                   # noqa: E402
import t2_gate_patch as G                        # noqa: E402
import t2_procedure as PROC                      # noqa: E402

SIMROOT = "/home/woori/scratch/tau2-bench/data/simulations"
A2 = _GI.load_domain_a2("banking_knowledge") or {}
PROCS = A2.get("procedures") or []
TARGET = set("task_032 task_033 task_035 task_043 task_044 task_045 task_047 "
             "task_048 task_049".split())

TAGRE = re.compile(r"^bank_.*_(202609(?:03|04|05))_(\d{4})$")
best = {}
for d in sorted(os.listdir(SIMROOT)):
    m = TAGRE.match(d)
    if not m:
        continue
    day, hhmm = m.group(1), int(m.group(2))
    if day == "20260903" and hhmm < 1400:
        continue
    if day == "20260905" and hhmm > 330:
        continue
    p = "%s/%s/results.json" % (SIMROOT, d)
    if not os.path.exists(p):
        continue
    try:
        res = Results.load(Path(p))
    except Exception:
        continue
    for s in res.simulations:
        tid = getattr(s, "task_id", None)
        if tid not in TARGET:
            continue
        ts = str(getattr(s, "end_time", None) or getattr(s, "start_time", None) or "")
        rw = getattr(getattr(s, "reward_info", None), "reward", None)
        if tid not in best or (ts, d) > best[tid][0]:
            best[tid] = ((ts, d), s, rw)

for tid in sorted(best):
    (_k, sim, rw) = best[tid]
    msgs = list(sim.messages or [])
    pts = []
    for i, m in enumerate(msgs):
        if str(getattr(m, "role", "")) != "assistant" or (getattr(m, "tool_calls", None) or []):
            continue
        c = getattr(m, "content", None)
        if not (isinstance(c, str) and c.strip()):
            continue
        done = G._executed_tool_counts(msgs[:i])
        rows, pids = [], []
        for p in PROC.active_procedures(PROCS, done):
            for nid, tools, ok in PROC.checklist(p, done):
                if ok is False:
                    rows.append((nid, list(tools or [])))
                    if p.get("id") not in pids:
                        pids.append(p.get("id"))
        pts.append((i, len(rows), pids, [r[0] for r in rows]))
    firing = [p for p in pts if p[1]]
    first = firing[0] if firing else None
    print("%-9s reward=%-5s msgs=%3d resign_pts=%2d firing_pts=%2d" %
          (tid, rw, len(msgs), len(pts), len(firing)))
    if first:
        ordn = [j for j, p in enumerate(pts) if p[0] == first[0]][0] + 1
        print("      FIRST fire at msg %d/%d (resign #%d/%d · %.0f%% through) proc=%s nodes=%s"
              % (first[0], len(msgs), ordn, len(pts), 100.0 * first[0] / max(1, len(msgs)),
                 first[2], first[3]))
        print("      TOOLS %s" % ([r for r in
                                   [(n, t) for n, t in
                                    [(x[0], x[1]) for x in
                                     [(nid, tools) for p in PROC.active_procedures(
                                         PROCS, G._executed_tool_counts(msgs[:first[0]]))
                                      for nid, tools, ok in PROC.checklist(p, G._executed_tool_counts(msgs[:first[0]]))
                                      if ok is False]]]],))
