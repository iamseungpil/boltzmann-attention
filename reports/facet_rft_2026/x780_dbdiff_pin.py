#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x780 - pinned (tag, task, sim) full DB diff for w-value-select forensics. READ-ONLY."""
import io, sys
from pathlib import Path
from loguru import logger
from tau2.registry import registry
from tau2.data_model.simulation import Results
logger.remove()
try: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception: pass

SIMROOT = "/home/woori/scratch/tau2-bench/data/simulations"
DOMAIN = "banking_knowledge"
env_ctor = registry.get_env_constructor(DOMAIN)
tasks = {t.id: t for t in registry.get_tasks_loader(DOMAIN)()}

class _NoInitial(object):
    initialization_data = None; initialization_actions = None; message_history = None

def diff(g, p, path="", out=None):
    if type(g) != type(p):
        out.append("TYPE %s: gold=%r pred=%r" % (path, g, p)); return
    if isinstance(g, dict):
        for k in sorted(set(g) | set(p), key=str):
            if k not in g: out.append("ONLY-PRED %s.%s = %r" % (path, k, p[k]))
            elif k not in p: out.append("ONLY-GOLD %s.%s = %r" % (path, k, g[k]))
            else: diff(g[k], p[k], path + "." + str(k), out)
    elif isinstance(g, list):
        if len(g) != len(p): out.append("LEN %s: gold=%d pred=%d" % (path, len(g), len(p)))
        for i in range(min(len(g), len(p))): diff(g[i], p[i], "%s[%d]" % (path, i), out)
    elif g != p:
        out.append("DIFF %s: gold=%r pred=%r" % (path, g, p))

for spec in sys.argv[1:]:
    tag, tid, simid = spec.split(",")
    res = Results.load(Path("%s/%s/results.json" % (SIMROOT, tag)))
    sim = next((s for s in res.simulations if s.id == simid), None)
    if sim is None:
        print("NOSIM %s %s" % (tid, simid)); continue
    task = tasks[tid]
    istate = task.initial_state or _NoInitial
    print("\n########## %s  tag=%s  sim=%s" % (tid, tag, simid))
    print("REWARD_BASIS_TASK=%r" % (getattr(task.evaluation_criteria, "reward_basis", None),))
    gold = env_ctor(retrieval_variant="no_knowledge")
    gold.set_state(istate.initialization_data, istate.initialization_actions, list(istate.message_history or []))
    for a in (task.evaluation_criteria.actions or []):
        try: gold.make_tool_call(tool_name=a.name, requestor=a.requestor, **a.arguments)
        except Exception as e: print("GOLDERR %s %r" % (a.name, str(e)[:120]))
    pred = env_ctor(retrieval_variant="no_knowledge")
    try:
        pred.set_state(istate.initialization_data, istate.initialization_actions, list(sim.messages))
    except ValueError as e:
        print("REPLAY-FAIL %r" % str(e)[:200]); continue
    lines = []
    diff(gold.tools.db.model_dump(), pred.tools.db.model_dump(), "", lines)
    if gold.user_tools:
        diff(gold.user_tools.db.model_dump(), pred.user_tools.db.model_dump(), "USERDB", lines)
    print("MATCH=%s  ndiff=%d" % (gold.tools.get_db_hash() == pred.tools.get_db_hash(), len(lines)))
    for ln in lines:
        print("  " + ln[:400])
