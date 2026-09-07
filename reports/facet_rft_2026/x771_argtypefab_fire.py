#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x769 — `T2_WRITE_ARG_TYPE` / `T2_WRITE_ARG_FAB` **발화 실측** (2026-09-05).

⛔판정하지 않는다. 세기만 한다. GPU 0 · 회수된 궤적만 읽는다.

엔진의 술어를 **재구현하지 않는다** — `t2_gate_patch` 의 함수를 그대로 import 해서
회수 궤적 위에서 돌린다([[78]] 격리→배선 · [[71]] 엔진 빌더 호출).

  T2_WRITE_ARG_FAB  = `_declared_params_by_tool` + `_prov_scan_args` + `_looks_placeholder`
                      + `_ctx_has(_ctx_from_messages(...))`  (t2_gate_patch:11932~11968 축자)
  T2_WRITE_ARG_TYPE = A2 `write_arg_enum` 의 `booleans` + `isinstance(v, bool)`
                      (t2_gate_patch:12117~12150 축자) — ⚠바깥 게이트
                      `if os.environ.get("T2_WRITE_ARG_ENUM") == "1" and _ens:` (12041) 포함해 센다.

출력:
  WIRE   ...                      바깥 게이트/선택자 집합 사실
  FIRE   <task> <lever> tool=<> arg=<> val=<> turn=<i> goldhit=<0|1>
  SUM    <task> TYPE=<n> FAB=<n> TYPEcand=<n> FABcand=<n> calls=<n>
  TOTAL  ...
"""
import collections, io, json, os, re, sys
from pathlib import Path
from loguru import logger

REPO = "/home/woori/workspace_common/boltzmann-attention-pi"
sys.path.insert(0, REPO + "/scripts/distill/tau2")

logger.remove()
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

from tau2.registry import registry
from tau2.data_model.simulation import Results

import t2_gate_patch as G
from gate_interpreter import load_domain_a2

SIMROOT = os.environ.get("SIMROOT", "/home/woori/scratch/tau2-bench/data/simulations")
DOMAIN = "banking_knowledge"

env_ctor = registry.get_env_constructor(DOMAIN)
tasks = {t.id: t for t in registry.get_tasks_loader(DOMAIN)()}
A2 = load_domain_a2(DOMAIN) or {}
ENS = A2.get("write_arg_enum") or []
BOOL_SPECS = [s for s in ENS if s.get("booleans")]

_env0 = env_ctor(retrieval_variant="no_knowledge")
try:
    SEL = G._selector_args_cached(_env0)
except Exception as e:
    SEL = G._SELECTOR_FALLBACK
    print("WIRE selector-derivation-failed %r -> fallback" % (e,))
print("WIRE selectors=%s" % (sorted(SEL),))
print("WIRE a2.write_arg_enum=%d  bool-specs=%d  bool-names=%s"
      % (len(ENS), len(BOOL_SPECS),
         sorted({b for s in BOOL_SPECS for b in s["booleans"]})))
print("WIRE outer-gate T2_WRITE_ARG_ENUM must be '1' for TYPE branch to run at all")

PAIRS = [ln.split() for ln in open(sys.argv[1]).read().strip().splitlines() if ln.strip()]
ONLY_FAILS = os.environ.get("X769_ALLSIMS", "0") != "1"


def gold_arg_values(task):
    """gold 호출들의 인자 값(문자열화) 집합 — 오차단 판정용([[70]] sells)."""
    out = set()
    for a in (getattr(task, "evaluation_criteria", None).actions or []
              if getattr(task, "evaluation_criteria", None) else []):
        for k, v in (a.arguments or {}).items():
            if isinstance(v, (str, int, float)):
                out.add(str(v).strip())
            if k == "arguments" and isinstance(v, str):
                try:
                    inner = json.loads(v)
                except Exception:
                    inner = None
                if isinstance(inner, dict):
                    for _k2, _v2 in inner.items():
                        if isinstance(_v2, (str, int, float)):
                            out.add(str(_v2).strip())
    return out


def scan_sim(tid, sim, gold_vals):
    msgs = list(sim.messages)
    fseen = set()          # T2_WRITE_ARG_FAB: (exact_tool, arg) — sim 당 1회
    tseen = set()          # T2_WRITE_ARG_TYPE: 변이 키 — sim 당 1회
    n = {"TYPE": 0, "FAB": 0, "TYPEcand": 0, "FABcand": 0, "calls": 0,
         "nospec": 0, "TYPEoverblock": 0, "FABoverblock": 0}
    rows = []
    for i, m in enumerate(msgs):
        tcs = getattr(m, "tool_calls", None) or []
        if getattr(m, "role", None) != "assistant" or not tcs:
            continue
        hist = msgs[:i]
        n["calls"] += len(tcs)
        # ── T2_WRITE_ARG_FAB (엔진 축자 순서) ──
        dpt = G._declared_params_by_tool(hist)
        if not dpt:
            n["nospec"] += 1
        else:
            fctx = G._ctx_from_messages(hist)
            fired_turn = False
            for c in tcs:
                if fired_turn:
                    break
                dp = dpt.get(str(G._exact_tool_name(c) or "")) or {}
                for fk, fv in G._prov_scan_args(c, selectors=SEL):
                    ft = dp.get(fk)
                    if not ft or ft[0] != "string" or ft[1]:
                        continue
                    fs = str(fv).strip()
                    if len(fs) < 4 or not G._looks_placeholder(fs):
                        continue
                    if G._ctx_has(fs, fctx):
                        continue
                    n["FABcand"] += 1
                    fkey = (str(G._exact_tool_name(c) or ""), fk)
                    if fkey in fseen:
                        continue
                    fseen.add(fkey)
                    n["FAB"] += 1
                    gh = 1 if fs in gold_vals else 0
                    n["FABoverblock"] += gh
                    rows.append("FIRE %s FAB tool=%s arg=%s val=%r turn=%d goldhit=%d"
                                % (tid, G._eff_tool_name(c), fk, fs[:48], i, gh))
                    fired_turn = True
                    break
        # ── T2_WRITE_ARG_TYPE (엔진 축자 순서 · 바깥 ENUM 게이트는 별도 보고) ──
        for c in tcs:
            ad = G._args_dict(c)
            for sp in BOOL_SPECS:
                if str(getattr(c, "name", "")) != str(sp.get("applies_to")):
                    continue
                aw = sp.get("applies_when") or {}
                if aw and not str(ad.get(aw.get("arg")) or "").startswith(
                        str(aw.get("prefix") or "\0")):
                    continue
                ia = ad.get("arguments")
                try:
                    ia = json.loads(ia) if isinstance(ia, str) else (ia or {})
                except Exception:
                    ia = {}
                if not isinstance(ia, dict):
                    continue
                bad = [(bk, ia.get(bk)) for bk in sp["booleans"]
                       if bk in ia and not isinstance(ia.get(bk), bool)]
                if not bad:
                    continue
                n["TYPEcand"] += len(bad)
                tk = G._mut_key_of(c) or str(G._exact_tool_name(c) or "")
                if not tk or tk in tseen:
                    continue
                tseen.add(tk)
                n["TYPE"] += 1
                rows.append("FIRE %s TYPE tool=%s bad=%s turn=%d"
                            % (tid, G._eff_tool_name(c),
                               [(k, v) for k, v in bad], i))
    return n, rows


cache = {}
tot = collections.Counter()
per = {}
for tag, tid, simid in PAIRS:
    if tag not in cache:
        try:
            cache[tag] = Results.load(Path("%s/%s/results.json" % (SIMROOT, tag)))
        except Exception as e:
            print("LOADFAIL %s %s %r" % (tid, tag, e)); cache[tag] = None
    res = cache[tag]
    if res is None:
        continue
    sims = [s for s in res.simulations if s.id == simid] if ONLY_FAILS else \
        [s for s in res.simulations if getattr(s, "task_id", None) == tid]
    if not sims:
        print("NOSIM %s %s %s" % (tid, tag, simid)); continue
    gv = gold_arg_values(tasks.get(tid)) if tasks.get(tid) else set()
    agg = collections.Counter()
    for sim in sims:
        n, rows = scan_sim(tid, sim, gv)
        for r in rows:
            print(r)
        agg.update(n)
    per[tid] = agg
    tot.update(agg)
    print("SUM %s TYPE=%d FAB=%d TYPEcand=%d FABcand=%d calls=%d nospec=%d "
          "TYPEover=%d FABover=%d sims=%d"
          % (tid, agg["TYPE"], agg["FAB"], agg["TYPEcand"], agg["FABcand"],
             agg["calls"], agg["nospec"], agg["TYPEoverblock"], agg["FABoverblock"],
             len(sims)))
    sys.stdout.flush()

print("TOTAL tasks=%d TYPE=%d FAB=%d TYPEcand=%d FABcand=%d calls=%d "
      "TYPEover=%d FABover=%d"
      % (len(per), tot["TYPE"], tot["FAB"], tot["TYPEcand"], tot["FABcand"],
         tot["calls"], tot["TYPEoverblock"], tot["FABoverblock"]))
print("TOTAL tasks-with-TYPE=%s" % sorted(t for t, a in per.items() if a["TYPE"]))
print("TOTAL tasks-with-FAB=%s" % sorted(t for t, a in per.items() if a["FAB"]))
