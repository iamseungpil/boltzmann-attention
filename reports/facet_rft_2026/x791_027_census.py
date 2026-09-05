#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x791 — task_027 (및 임의 태스크) 의 **전 런 sim census** (2026-09-05).

귀무가설 검정용: 같은 태스크가 **개입 없이도** sim 마다 실패 축(E/M/W/T/RC)이
통째로 바뀌는가. x768 의 변이-단위 로직을 그대로 쓰되 (tag,sim) 을 열거한다.

usage:  x791_027_census.py <task_id> <listfile-of-results.json-paths>
⛔ 판정하지 않는다. 세기만 한다.
"""
import collections, gzip, io, json, os, re, sys
from pathlib import Path
from loguru import logger
from tau2.registry import registry
from tau2.data_model.simulation import Results

logger.remove()
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

DOMAIN = "banking_knowledge"
env_ctor = registry.get_env_constructor(DOMAIN)
tasks = {t.id: t for t in registry.get_tasks_loader(DOMAIN)()}

TID = sys.argv[1]
PATHS = [ln.strip() for ln in open(sys.argv[2]).read().splitlines() if ln.strip()]


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


def load(p):
    if p.endswith(".gz"):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            return Results.model_validate(json.load(fh))
    return Results.load(Path(p))


task = tasks.get(TID)
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
GOLD_MAIN = gold.tools.db.model_dump()
GOLD_USER = gold.user_tools.db.model_dump() if gold.user_tools else None
GOLD_HASH = gold.tools.get_db_hash()
print("GOLD %s golderr=%d" % (TID, golderr)); sys.stdout.flush()

for p in PATHS:
    tag = Path(p).parent.name if p.endswith("results.json") else Path(p).name
    try:
        res = load(p)
    except Exception as e:
        print("LOADFAIL %s %r" % (tag, str(e)[:120])); sys.stdout.flush(); continue
    info = res.info
    try:
        ai = info.agent_info.llm
        ui = info.user_info.llm
        gc = (info.git_commit or "")[:8]
    except Exception:
        ai = ui = gc = "?"
    sims = [s for s in res.simulations if s.task_id == TID]
    if not sims:
        print("NOTASK %s" % tag); sys.stdout.flush(); continue
    for sim in sims:
        # first user utterance
        u1 = ""
        for m in sim.messages:
            if getattr(m, "role", "") == "user" and getattr(m, "content", None):
                u1 = m.content.replace("\n", " ")[:150]
                break
        pred = env_ctor(retrieval_variant="no_knowledge")
        try:
            pred.set_state(istate.initialization_data, istate.initialization_actions,
                           list(sim.messages))
        except Exception as e:
            print("SIM %s %s tag=%s REPLAY-FAIL %r" % (TID, sim.id[:8], tag, str(e)[:90]))
            sys.stdout.flush(); continue
        lines = []
        diff(GOLD_MAIN, pred.tools.db.model_dump(), "", lines)
        ulines = []
        if GOLD_USER is not None and pred.user_tools:
            diff(GOLD_USER, pred.user_tools.db.model_dump(), "", ulines)
        match = (GOLD_HASH == pred.tools.get_db_hash())
        cnt = collections.Counter(); fields = collections.Counter(); rc = collections.Counter()
        for ln in lines + ulines:
            kind = ln.split(" ", 1)[0]
            top, sub = toppath(ln)
            if top in ("agent_discoverable_tools", "user_discoverable_tools",
                       "agent_discoverable_tool_calls", "user_discoverable_tool_calls"):
                rc[kind + ":" + top] += 1
                continue
            cnt[kind] += 1
            fields[kind + " " + top + ("." + sub if sub else "")] += 1
        rw = None
        try:
            rw = sim.reward_info.reward
        except Exception:
            pass
        print("SIM %s tag=%s sim=%s rw=%s term=%s nmsg=%d match=%s "
              "E=%d M=%d W=%d L=%d T=%d RC=%d llm=%s user=%s gc=%s"
              % (TID, tag, sim.id, rw, getattr(sim, "termination_reason", "?"),
                 len(sim.messages), match, cnt["ONLY-PRED"], cnt["ONLY-GOLD"],
                 cnt["DIFF"], cnt["LEN"], cnt["TYPE"], sum(rc.values()), ai, ui, gc))
        print("   U1   %s" % u1)
        for k, v in rc.most_common(6):
            print("   RC   %s x%d" % (k, v))
        for k, v in fields.most_common(14):
            print("   FLD  %s x%d" % (k, v))
        if os.environ.get("X791_FULL") == "1":
            for ln in lines + ulines:
                print("   RAW  %s" % ln[:240])
        sys.stdout.flush()
print("DONE")
