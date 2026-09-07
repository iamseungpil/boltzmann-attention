#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x769c - 캠페인 97 전체(pass 51 + fail 46)에 대한 두 레버의 **발화 실측**.
[[70]] 파는 것을 세려면 통과 sim 에서의 발화도 세야 한다.
"""
import io, os, sys, glob
from pathlib import Path
from loguru import logger
logger.remove()
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = "/home/woori/workspace_common/boltzmann-attention-pi"
sys.path.insert(0, REPO + "/scripts/distill/tau2")
import t2_gate_patch as G
import gate_interpreter as GI
from tau2.data_model.simulation import Results

SIMROOT = "/home/woori/scratch/tau2-bench/data/simulations"
LO, HI = "2026-09-03T14", "2026-09-05T03"
A2 = GI.load_domain_a2("banking_knowledge")
WRSET = (G._confirm_write_tools(A2)
         | set(((A2 or {}).get("eplan") or {}).get("write_tools") or []))
MIN = 8

sys.path.insert(0, "/home/woori/scratch/x768")
from x769_remote_fire import specfacts_hits, specatwrite_hits

def main():
    best = {}
    files = sorted(glob.glob(os.path.join(SIMROOT, "*", "*.json")))
    for p in files:
        st = os.path.getmtime(p)
        try:
            res = Results.load(Path(p))
        except Exception:
            continue
        for s in (res.simulations or []):
            ts = str(getattr(s, "start_time", "") or getattr(s, "timestamp", "") or "")
            if not (LO <= ts[:13] <= HI):
                continue
            t = s.task_id
            if t not in best or ts > best[t][0]:
                best[t] = (ts, os.path.basename(os.path.dirname(p)), s)
    rows = sorted(best.items())
    npass = sum(1 for _, v in rows if (getattr(v[2].reward_info, "reward", 0) or 0) >= 1.0)
    print("RECON tasks=%d pass=%d fail=%d" % (len(rows), npass, len(rows) - npass))
    agg = {"pass": [0, 0, 0, 0], "fail": [0, 0, 0, 0]}
    for t, (ts, tag, s) in rows:
        r = (getattr(s.reward_info, "reward", 0) or 0)
        band = "pass" if r >= 1.0 else "fail"
        msgs = list(s.messages or [])
        th, eh = specfacts_hits(msgs)
        sw = specatwrite_hits(msgs)
        fire = [x for x in sw if x[4] == "FIRE"]
        agg[band][0] += len(th); agg[band][1] += len(eh)
        agg[band][2] += len(sw); agg[band][3] += len(fire)
        if th or eh or fire:
            print("HIT %s %s r=%.1f type=%d enum=%d SAWfire=%d dists=%s"
                  % (band, t, r, len(th), len(eh), len(fire),
                     [(x[0], x[2], x[3]) for x in sw][:6]))
        for h in th:
            print("   TYPE %s tool=%s args=%s vals=%s" % (t, h[1], h[2], h[3]))
        for h in eh:
            print("   ENUM %s tool=%s arg=%s val=%r" % (t, h[1], h[2], h[3]))
    for b in ("pass", "fail"):
        print("AGG %s type=%d enum=%d SAWreach=%d SAWfire=%d"
              % (b, agg[b][0], agg[b][1], agg[b][2], agg[b][3]))

main()
