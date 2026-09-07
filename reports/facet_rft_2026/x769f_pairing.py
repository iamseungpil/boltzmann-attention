#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x769f - SPEC_AT_WRITE 가 되붙이는 블록이 **그 write 의 도구 명세가 맞는가**(짝 검산)."""
import io, os, sys, glob
from pathlib import Path
from loguru import logger
logger.remove()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
REPO = "/home/woori/workspace_common/boltzmann-attention-pi"
sys.path.insert(0, REPO + "/scripts/distill/tau2")
import t2_gate_patch as G
import gate_interpreter as GI
from tau2.data_model.simulation import Results
SIMROOT = "/home/woori/scratch/tau2-bench/data/simulations"
A2 = GI.load_domain_a2("banking_knowledge")
WRSET = (G._confirm_write_tools(A2)
         | set(((A2 or {}).get("eplan") or {}).get("write_tools") or []))
pairs = [ln.split() for ln in open(sys.argv[1]).read().strip().splitlines() if ln.strip()]
cache = {}
ok = bad = 0
for tag, task, simid in pairs:
    if tag not in cache:
        g = glob.glob(os.path.join(SIMROOT, tag, "*.json"))
        cache[tag] = Results.load(Path(g[0])) if g else None
    res = cache[tag]
    if res is None:
        continue
    sim = next((s for s in res.simulations if str(s.id) == simid), None)
    if sim is None:
        continue
    msgs = list(sim.messages or [])
    seen = set()
    for i, m in enumerate(msgs):
        tcs = getattr(m, "tool_calls", None) or []
        if str(getattr(m, "role", "")) != "assistant" or not tcs:
            continue
        wc = next((c for c in tcs if G._eff_tool_name(c) in WRSET
                   or str(getattr(c, "name", "")) in WRSET), None)
        if wc is None:
            continue
        spec, si, sd = G._env_spec_for(wc, msgs[:i])
        k = str(G._exact_tool_name(wc) or "")
        if not spec or sd < 8 or k in seen:
            continue
        seen.add(k)
        tm = G._DECL_TOOL_RE.search(spec) or G._DECL_TOOL_ALT_RE.search(spec)
        got = tm.group(1) if tm else "(none)"
        hit = (got == k)
        ok += hit; bad += (not hit)
        print("FIRE %s msg=%d dist=%d len=%d want=%s delivered=%s %s"
              % (task, i, sd, len(spec), k, got, "MATCH" if hit else "MISMATCH"))
print("PAIRING match=%d mismatch=%d" % (ok, bad))
