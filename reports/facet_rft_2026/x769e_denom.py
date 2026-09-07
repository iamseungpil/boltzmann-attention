#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x769e - SPEC_ARG_FACTS 두 술어의 **분모**를 46 핀 sim 에서 센다([[77]] «없다»의 검색 경로)."""
import io, os, sys, collections
from pathlib import Path
from loguru import logger
logger.remove()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
REPO = "/home/woori/workspace_common/boltzmann-attention-pi"
sys.path.insert(0, REPO + "/scripts/distill/tau2")
import t2_gate_patch as G
from tau2.data_model.simulation import Results
SIMROOT = "/home/woori/scratch/tau2-bench/data/simulations"
pairs = [ln.split() for ln in open(sys.argv[1]).read().strip().splitlines() if ln.strip()]
cache = {}
nb = ne = bb = be = 0
kinds = collections.Counter()
for tag, task, simid in pairs:
    if tag not in cache:
        p = Path(SIMROOT) / tag / "results.json"
        if not p.exists():
            import glob
            g = glob.glob(os.path.join(SIMROOT, tag, "*.json"))
            p = Path(g[0]) if g else None
        cache[tag] = Results.load(p) if p else None
    res = cache[tag]
    if res is None:
        print("LOADFAIL", tag); continue
    sim = next((s for s in res.simulations if str(s.id) == simid), None)
    if sim is None:
        print("NOSIM", tag, task); continue
    msgs = list(sim.messages or [])
    for i, m in enumerate(msgs):
        tcs = getattr(m, "tool_calls", None) or []
        if str(getattr(m, "role", "")) != "assistant" or not tcs:
            continue
        dpt = G._declared_params_by_tool(msgs[:i])
        for c in tcs:
            d2 = dpt.get(str(G._exact_tool_name(c) or "")) or {}
            if not d2:
                continue
            for k, v in G._prov_scan_args(c, selectors=None):
                typ, enum = (d2.get(k) or ("", []))
                if typ == "boolean":
                    nb += 1
                    if not isinstance(v, bool):
                        bb += 1; kinds["B:" + k] += 1
                elif enum:
                    ne += 1
                    if str(v).strip() not in enum:
                        be += 1; kinds["E:" + k] += 1
print("DENOM sims=%d boolean_arg_passes=%d nonbool=%d enum_arg_passes=%d out_of_enum=%d"
      % (len(pairs), nb, bb, ne, be))
print("KINDS", dict(kinds))
