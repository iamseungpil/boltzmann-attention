#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x773 - locate sims for tasks 026/039/086 across all recovered results (read-only)."""
import gzip, json, os, glob
SIMDIR = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"
TARGET = {"task_026","task_039","task_086"}
rows=[]
for fp in glob.glob(os.path.join(SIMDIR,"*.results.json.gz")):
    try:
        with gzip.open(fp,"rt",encoding="utf-8") as fh: d=json.load(fh)
    except Exception as e:
        continue
    sims = d.get("simulations") or []
    for s in sims:
        tid=s.get("task_id")
        if tid in TARGET:
            ri=s.get("reward_info") or {}
            rows.append((s.get("timestamp") or s.get("end_time") or "", tid, os.path.basename(fp), s.get("id"), ri.get("reward")))
rows.sort()
for r in rows:
    print("%s | %s | rw=%s | %s | %s" % (r[0], r[1], r[4], r[2], r[3]))
