#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io, os, sys, glob
from pathlib import Path
from loguru import logger
logger.remove()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from tau2.data_model.simulation import Results
SIMROOT = "/home/woori/scratch/tau2-bench/data/simulations"
LO, HI = "2026-09-03T14", "2026-09-05T03"
best = {}
for p in sorted(glob.glob(os.path.join(SIMROOT, "*", "*.json"))):
    try:
        res = Results.load(Path(p))
    except Exception:
        continue
    for s in (res.simulations or []):
        ts = str(getattr(s, "start_time", "") or "")
        if not (LO <= ts[:13] <= HI):
            continue
        t = s.task_id
        if t not in best or ts > best[t][0]:
            best[t] = (ts, os.path.basename(os.path.dirname(p)), s)
for t, (ts, tag, s) in sorted(best.items()):
    print("R %s %s %s %s %.1f" % (t, ts, tag, s.id,
                                  (getattr(s.reward_info, "reward", 0) or 0)))
