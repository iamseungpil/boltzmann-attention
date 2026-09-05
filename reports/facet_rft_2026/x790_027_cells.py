#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x790 — 한 (tag, task, sim) 의 **변이 칸 전수 출력** (2026-09-05, task_027 포렌식용).

x768_fail46_mutunits.py 와 같은 리플레이 로직. 다른 점 = 상위 N 요약이 아니라
**모든 diff 줄을 그대로** 찍는다. gold.tools.db 와 gold.user_tools.db 가 같은 객체라
2배로 세어지는 문제는 tools.db 쪽만 찍어서 회피(=고유칸).

사용:  python x790_027_cells.py <3열파일>
⛔ 판정하지 않는다. 찍기만 한다.
"""
import io, os, re, sys
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
        try:
            cache[tag] = Results.load(Path("%s/%s/results.json" % (SIMROOT, tag)))
        except Exception as e:
            print("LOADFAIL %s %s %r" % (tid, tag, e)); cache[tag] = None
    res = cache[tag]
    if res is None:
        continue
    sim = next((s for s in res.simulations if s.id == simid), None)
    if sim is None:
        print("NOSIM %s %s %s" % (tid, tag, simid)); continue
    task = tasks.get(tid)
    istate = task.initial_state or _NoInitial
    gold = env_ctor(retrieval_variant="no_knowledge")
    gold.set_state(istate.initialization_data, istate.initialization_actions,
                   list(istate.message_history or []))
    golderr = 0
    for a in (task.evaluation_criteria.actions or []):
        try:
            gold.make_tool_call(tool_name=a.name, requestor=a.requestor, **a.arguments)
        except Exception as e:
            golderr += 1
            print("GOLDERR %s %s %r" % (tid, a.name, str(e)[:120]))
    pred = env_ctor(retrieval_variant="no_knowledge")
    try:
        pred.set_state(istate.initialization_data, istate.initialization_actions,
                       list(sim.messages))
    except ValueError as e:
        print("UNIT %s REPLAY-FAIL %r" % (tid, str(e)[:160])); continue
    lines = []
    diff(gold.tools.db.model_dump(), pred.tools.db.model_dump(), "", lines)
    match = (gold.tools.get_db_hash() == pred.tools.get_db_hash())
    print("### UNIT %s tag=%s sim=%s match=%s golderr=%d uniqcells=%d"
          % (tid, tag, simid, match, golderr, len(lines)))
    print("### GOLD_ACTIONS %s" % ([ (a.name, a.arguments) for a in (task.evaluation_criteria.actions or []) ],))
    for ln in lines:
        print("CELL %s | %s" % (tid, ln))
    sys.stdout.flush()
