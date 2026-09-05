#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x794 - 실패 sim 의 **원시 DB diff 줄**을 그대로 찍는다 (x768 의 diff 를 재사용). 2026-09-05.
사용:  python x794_flipB_dbdiff_raw.py <3col.txt> [maxchars]
⛔ 판정하지 않는다.
"""
import io, sys
from pathlib import Path
from loguru import logger
from tau2.registry import registry
from tau2.data_model.simulation import Results

logger.remove()
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

SIMROOT = "/home/woori/scratch/tau2-bench/data/simulations"
DOMAIN = "banking_knowledge"
env_ctor = registry.get_env_constructor(DOMAIN)
tasks = {t.id: t for t in registry.get_tasks_loader(DOMAIN)()}
PAIRS = [ln.split() for ln in open(sys.argv[1]).read().strip().splitlines() if ln.strip()]
MAXC = int(sys.argv[2]) if len(sys.argv) > 2 else 600


class _NoInitial(object):
    initialization_data = None
    initialization_actions = None
    message_history = None


def diff(g, p, path="", out=None):
    if type(g) != type(p):
        out.append("TYPE %s: %r vs %r" % (path, g, p)); return
    if isinstance(g, dict):
        for k in sorted(set(g) | set(p), key=str):
            if k not in g:
                out.append("ONLY-PRED %s.%s = %r" % (path, k, p[k]))
            elif k not in p:
                out.append("ONLY-GOLD %s.%s = %r" % (path, k, g[k]))
            else:
                diff(g[k], p[k], path + "." + str(k), out)
    elif isinstance(g, list):
        if len(g) != len(p):
            out.append("LEN %s: gold=%d pred=%d" % (path, len(g), len(p)))
        for i in range(min(len(g), len(p))):
            diff(g[i], p[i], "%s[%d]" % (path, i), out)
    elif g != p:
        out.append("DIFF %s: gold=%r pred=%r" % (path, g, p))


cache = {}
for tag, tid, simid in PAIRS:
    if tag not in cache:
        cache[tag] = Results.load(Path("%s/%s/results.json" % (SIMROOT, tag)))
    res = cache[tag]
    sim = next((s for s in res.simulations if s.id == simid), None)
    task = tasks.get(tid)
    istate = task.initial_state or _NoInitial
    gold = env_ctor(retrieval_variant="no_knowledge")
    gold.set_state(istate.initialization_data, istate.initialization_actions,
                   list(istate.message_history or []))
    for a in (task.evaluation_criteria.actions or []):
        try:
            gold.make_tool_call(tool_name=a.name, requestor=a.requestor, **a.arguments)
        except Exception:
            pass
    pred = env_ctor(retrieval_variant="no_knowledge")
    try:
        pred.set_state(istate.initialization_data, istate.initialization_actions, list(sim.messages))
    except ValueError as e:
        print("== %s REPLAY-FAIL %r" % (tid, str(e)[:120])); continue
    lines = []
    diff(gold.tools.db.model_dump(), pred.tools.db.model_dump(), "", lines)
    print("== %s %s" % (tid, tag))
    for ln in lines:
        if ln.startswith(("ONLY-PRED .agent_discoverable", "ONLY-GOLD .agent_discoverable",
                          "ONLY-PRED .user_discoverable", "ONLY-GOLD .user_discoverable")):
            continue
        print("   " + ln[:MAXC])
