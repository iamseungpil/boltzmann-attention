#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x768 — 캠페인 실패 46건의 **변이 단위 전수 재집계** (2026-09-05).

dbdiff_task.py 의 리플레이 로직을 그대로 쓰되 **(tag, task, sim_id) 핀 지정**으로 돈다
(캠페인의 «태스크당 최신 sim» 이 태그를 가로질러 흩어져 있기 때문).

출력 = 태스크당 한 블록:
  UNIT <task> match=<db_match> E=<ONLY-PRED> M=<ONLY-GOLD> W=<DIFF> L=<LEN> T=<TYPE> RC=<read-coverage 줄>
  FIELD <task> <kind> <path>  … (write 축 상위 12줄)
⛔ 판정하지 않는다. 세기만 한다.
"""
import collections, io, os, re, sys
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


def toppath(line):
    m = re.match(r"[A-Z-]+ \.([^.\[]+)(?:\[[^\]]*\])?\.?([^.\[ :]*)", line)
    if not m:
        return "?", ""
    return m.group(1), (m.group(2) or "")


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
        except Exception:
            golderr += 1
    pred = env_ctor(retrieval_variant="no_knowledge")
    try:
        pred.set_state(istate.initialization_data, istate.initialization_actions,
                       list(sim.messages))
    except ValueError as e:
        print("UNIT %s REPLAY-FAIL %r" % (tid, str(e)[:100])); continue
    lines = []
    diff(gold.tools.db.model_dump(), pred.tools.db.model_dump(), "", lines)
    ulines = []
    if gold.user_tools:
        diff(gold.user_tools.db.model_dump(), pred.user_tools.db.model_dump(), "", ulines)
    match = (gold.tools.get_db_hash() == pred.tools.get_db_hash())
    cnt = collections.Counter()
    fields = collections.Counter()
    rc = collections.Counter()
    for ln in lines + ulines:
        kind = ln.split(" ", 1)[0]
        top, sub = toppath(ln)
        if top in ("agent_discoverable_tools", "user_discoverable_tools",
                   "agent_discoverable_tool_calls", "user_discoverable_tool_calls"):
            rc[kind + ":" + top] += 1
            continue
        cnt[kind] += 1
        fields[kind + " " + top + ("." + sub if sub else "")] += 1
    print("UNIT %s tag=%s match=%s golderr=%d E=%d M=%d W=%d L=%d T=%d RC=%d"
          % (tid, tag, match, golderr, cnt["ONLY-PRED"], cnt["ONLY-GOLD"], cnt["DIFF"],
             cnt["LEN"], cnt["TYPE"], sum(rc.values())))
    for k, v in rc.most_common(6):
        print("   RC   %s %s x%d" % (tid, k, v))
    for k, v in fields.most_common(14):
        print("   FLD  %s %s x%d" % (tid, k, v))
    sys.stdout.flush()
