#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x771b — `T2_WRITE_ARG_TYPE`/`T2_WRITE_ARG_FAB` **도달 분해 + 계기 부정통제** (2026-09-05).

x771 이 46/46 에서 0 을 냈다. 0 이 «결손 없음» 인지 «계기 고장» 인지 가른다([[55]]).
  ⑴ 도달 분해 — 술어를 한 칸씩 벗겨 어디서 0 이 되는지.
  ⑵ 부정통제 — 같은 코드를 t7354(2026-08-25 · 이 레버가 20건/전건 걸렸다고 기록된 런)에
     그대로 돌린다. 거기서 울면 계기는 산 것이고 캠페인의 0 은 실측이다.
⛔ 판정하지 않는다.
"""
import collections, io, json, os, sys
from pathlib import Path
from loguru import logger

REPO = "/home/woori/workspace_common/boltzmann-attention-pi"
sys.path.insert(0, REPO + "/scripts/distill/tau2")

logger.remove()
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

from tau2.data_model.simulation import Results
import t2_gate_patch as G
from gate_interpreter import load_domain_a2

SIMROOT = "/home/woori/scratch/tau2-bench/data/simulations"
A2 = load_domain_a2("banking_knowledge") or {}
BOOL_SPECS = [s for s in (A2.get("write_arg_enum") or []) if s.get("booleans")]
SEL = frozenset(["agent_tool_name", "discoverable_tool_name", "user_tool_name"])

LABEL = sys.argv[1]
PAIRS = [ln.split() for ln in open(sys.argv[2]).read().strip().splitlines() if ln.strip()]
ALLSIMS = os.environ.get("ALLSIMS", "0") == "1"

R = collections.Counter()
boolvals = collections.Counter()
strvals = collections.Counter()
cache = {}
for row in PAIRS:
    tag, tid = row[0], row[1]
    simid = row[2] if len(row) > 2 else None
    if tag not in cache:
        try:
            cache[tag] = Results.load(Path("%s/%s/results.json" % (SIMROOT, tag)))
        except Exception as e:
            print("LOADFAIL %s %r" % (tag, e)); cache[tag] = None
    res = cache[tag]
    if res is None:
        continue
    sims = ([s for s in res.simulations if getattr(s, "task_id", None) == tid] if ALLSIMS
            else [s for s in res.simulations if s.id == simid])
    for sim in sims:
        R["sims"] += 1
        msgs = list(sim.messages)
        for i, m in enumerate(msgs):
            tcs = getattr(m, "tool_calls", None) or []
            if getattr(m, "role", None) != "assistant" or not tcs:
                continue
            hist = msgs[:i]
            dpt = G._declared_params_by_tool(hist)
            fctx = G._ctx_from_messages(hist)
            R["turns"] += 1
            if dpt:
                R["turns_with_decl"] += 1
            for c in tcs:
                R["calls"] += 1
                raw = str(getattr(c, "name", "") or "")
                exact = str(G._exact_tool_name(c) or "")
                # ── TYPE 도달 분해 ──
                for sp in BOOL_SPECS:
                    if raw != str(sp.get("applies_to")):
                        continue
                    R["TYPE_a_name_match"] += 1
                    ad = G._args_dict(c)
                    aw = sp.get("applies_when") or {}
                    if aw and not str(ad.get(aw.get("arg")) or "").startswith(
                            str(aw.get("prefix") or "\0")):
                        continue
                    R["TYPE_b_prefix_match"] += 1
                    ia = ad.get("arguments")
                    try:
                        ia = json.loads(ia) if isinstance(ia, str) else (ia or {})
                    except Exception:
                        ia = {}
                    if not isinstance(ia, dict):
                        continue
                    present = [bk for bk in sp["booleans"] if bk in ia]
                    if present:
                        R["TYPE_c_bool_arg_present"] += 1
                    for bk in present:
                        v = ia.get(bk)
                        boolvals["%s=%s(%s)" % (bk, repr(v)[:12], type(v).__name__)] += 1
                        if not isinstance(v, bool):
                            R["TYPE_d_nonbool"] += 1
                # ── FAB 도달 분해 ──
                if not dpt:
                    continue
                dp = dpt.get(exact) or {}
                if dp:
                    R["FAB_a_decl_for_tool"] += 1
                for fk, fv in G._prov_scan_args(c, selectors=SEL):
                    ft = dp.get(fk)
                    if not ft:
                        continue
                    R["FAB_b_arg_declared"] += 1
                    if ft[0] != "string" or ft[1]:
                        continue
                    R["FAB_c_string_nonenum"] += 1
                    fs = str(fv).strip()
                    if len(fs) < 4:
                        continue
                    R["FAB_d_len4"] += 1
                    strvals[fs[:40]] += 1
                    if not G._looks_placeholder(fs):
                        continue
                    R["FAB_e_placeholder"] += 1
                    if G._ctx_has(fs, fctx):
                        R["FAB_f_in_ctx_pass"] += 1
                        continue
                    R["FAB_g_WOULD_FIRE"] += 1
                    print("HIT %s %s tool=%s arg=%s val=%r" % (LABEL, tid, exact, fk, fs[:40]))

print("REACH %s %s" % (LABEL, " ".join("%s=%d" % kv for kv in sorted(R.items()))))
print("BOOLVALS %s %s" % (LABEL, boolvals.most_common(20)))
print("PLACEHOLDER-CAND %s %s" % (LABEL, [(k, v) for k, v in strvals.most_common(15)
                                          if G._looks_placeholder(k)]))
