#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x769 — 레버군 6종의 **발화 실측** (2026-09-05).

대상 = T2_EPLAN_ENUM_SUBTRACT · T2_PROCEDURE_LEFT · T2_CALL_FORM_FIX ·
       T2_DENY_HOWTO(+PARAMS) · T2_STOP_FIRST_TOOLCALL

입력 = pairs.txt (tag task sim_id · 46행) + 리모트 런 로그(/home/woori/scratch/logs/<tag>.log)
⛔ 판정하지 않는다. **센다.**  ⛔ 코드 수정 0 — 엔진 함수를 그대로 부른다([[67]] 사본 금지).
"""
import collections, io, json, os, re, sys
from pathlib import Path
from loguru import logger

logger.remove()
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

from tau2.registry import registry            # noqa: E402
from tau2.data_model.simulation import Results  # noqa: E402
import gate_interpreter as _GI                # noqa: E402
import t2_gate_patch as G                     # noqa: E402
import t2_procedure as PROC                   # noqa: E402
import t2_eplan_patch as EP                   # noqa: E402

DOMAIN = "banking_knowledge"
SIMROOT = "/home/woori/scratch/tau2-bench/data/simulations"
LOGDIR = "/home/woori/scratch/logs"

A2 = _GI.load_domain_a2(DOMAIN) or {}
PROCS = A2.get("procedures") or []
ESPEC = EP.load_eplan_spec(DOMAIN) or {}
_le = ESPEC.get("list_enumerator")
ENUMS = set(_le) if isinstance(_le, (list, tuple)) else ({_le} if _le else set())

PAIRS = [ln.split() for ln in open(sys.argv[1]).read().strip().splitlines() if ln.strip()]

# ── 로그 패턴 ────────────────────────────────────────────────────────────────
PATS = {
    "EPLAN_L1_deny":  re.compile(r"\[T2_EPLAN\] L1 deny"),
    "EPLAN_L2_deny":  re.compile(r"\[T2_EPLAN\] L2 deny"),
    "PROC_checklist": re.compile(r"\[T2_PROCEDURE\] checklist proc=(\S+) nodes=(\d+) done=(\d+) left=(\[.*\])"),
    "TOOLOBS_err":    re.compile(r"\[T2_TOOL_OBS\] id=\S* err=True -> (.*)"),
    "STACK_folded":   re.compile(r"\[T2_STACK\] window folded fb tag=(\S+)"),
    "ROUTE_compete":  re.compile(r"\[T2_ROUTE\] (\S+) 경합"),
    "GIVEREQ":        re.compile(r"\[T2_GIVE_REQUIRED\]"),
}
# 우리 층 deny 문면의 선두 태그 (env 오류와 가르는 유일한 표지)
OURS_RE = re.compile(r"^Error: \[([A-Z0-9_\- ]+)\]")
FB_GENERIC = "Error: resolve the flagged call(s) first; do not call this tool yet."


def sim_log_lines(tag, task):
    p = os.path.join(LOGDIR, "%s.log" % tag)
    if not os.path.exists(p):
        return None
    key = "[sim=%s#" % task
    out = []
    with open(p, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if key in ln:
                out.append(ln.rstrip("\n"))
    return out


class _NoInitial(object):
    initialization_data = None
    initialization_actions = None
    message_history = None


def calls_of(m):
    return list(getattr(m, "tool_calls", None) or [])


cache = {}
tasks = {t.id: t for t in registry.get_tasks_loader(DOMAIN)()}

# 전역 집계
AGG = collections.Counter()
HOWTO_CACHE = {}


def howto(name, params_on):
    k = (name, params_on)
    if k in HOWTO_CACHE:
        return HOWTO_CACHE[k]
    old = os.environ.get("T2_DENY_HOWTO_PARAMS")
    os.environ["T2_DENY_HOWTO_PARAMS"] = "1" if params_on else "0"
    try:
        v = G._decl_howto(name, A2)
    except Exception as e:
        v = "ERR:%r" % (e,)
    if old is None:
        os.environ.pop("T2_DENY_HOWTO_PARAMS", None)
    else:
        os.environ["T2_DENY_HOWTO_PARAMS"] = old
    HOWTO_CACHE[k] = v
    return v


print("== A2 선언 ==")
print("procedures=%d · eplan.list_enumerator=%s · relations.by_tool=%d"
      % (len(PROCS), sorted(ENUMS),
         len(((A2.get("relations") or {}).get("by_tool")) or {})))
print("procedure ids=%s" % [p.get("id") for p in PROCS])
print()

for tag, tid, simid in PAIRS:
    if tag not in cache:
        try:
            cache[tag] = Results.load(Path("%s/%s/results.json" % (SIMROOT, tag)))
        except Exception as e:
            print("LOADFAIL %s %s %r" % (tid, tag, e))
            cache[tag] = None
    res = cache[tag]
    if res is None:
        continue
    sim = next((s for s in res.simulations if s.id == simid), None)
    if sim is None:
        print("NOSIM %s %s %s" % (tid, tag, simid))
        continue
    msgs = list(sim.messages or [])

    # ── ① T2_PROCEDURE_LEFT — resign 창마다 미충족 노드 ──────────────────────
    resign_pts, fire = 0, None
    for i, m in enumerate(msgs):
        if str(getattr(m, "role", "")) != "assistant":
            continue
        if calls_of(m):
            continue
        c = getattr(m, "content", None)
        if not (isinstance(c, str) and c.strip()):
            continue
        resign_pts += 1
        done = G._executed_tool_counts(msgs[:i])
        rows, pids = [], []
        for p in PROC.active_procedures(PROCS, done):
            for nid, tools, ok in PROC.checklist(p, done):
                if ok is False:
                    rows.append((nid, list(tools or [])))
                    if p.get("id") not in pids:
                        pids.append(p.get("id"))
        if rows and fire is None:
            fire = (i, resign_pts, len(rows), pids, [r[0] for r in rows])

    # ── ② T2_EPLAN_ENUM_SUBTRACT — 선언 열거자 호출 여부 ─────────────────────
    enum_calls = []
    for i, m in enumerate(msgs):
        for tc in calls_of(m):
            nm = G._eff_tool_name(tc)
            if nm in ENUMS or str(getattr(tc, "name", "")) in ENUMS:
                enum_calls.append((i, nm))

    # ── ③ T2_CALL_FORM_FIX — 손님-측 도구를 래퍼로 부른 적 있나 ─────────────
    cdut, gave = [], []
    for i, m in enumerate(msgs):
        for tc in calls_of(m):
            n = str(getattr(tc, "name", ""))
            if n == "call_discoverable_user_tool":
                a = G._args_dict(tc) or {}
                cdut.append((i, str(a.get("discoverable_tool_name") or ""),
                             str(getattr(m, "role", "")), str(getattr(tc, "requestor", ""))))
            if n == "give_discoverable_user_tool":
                a = G._args_dict(tc) or {}
                gave.append((i, str(a.get("discoverable_tool_name") or "")))

    # ── ④ T2_STOP_FIRST_TOOLCALL — 병렬 발사(파는 것) ───────────────────────
    asst, par, mx = 0, 0, 0
    for m in msgs:
        if str(getattr(m, "role", "")) != "assistant":
            continue
        asst += 1
        k = len(calls_of(m))
        mx = max(mx, k)
        if k >= 2:
            par += 1

    # ── ⑤ T2_DENY_HOWTO — 궤적의 호출 도구별 선언 존재 여부 ─────────────────
    called_names = collections.Counter()
    for m in msgs:
        if str(getattr(m, "role", "")) != "assistant":
            continue
        for tc in calls_of(m):
            called_names[G._eff_tool_name(tc)] += 1
    ht = {n: len(howto(n, False)) for n in called_names}
    ht_p = {n: len(howto(n, True)) for n in called_names}
    ht_hit = sorted(n for n, L in ht.items() if L > 0)
    htp_only = sorted(n for n in called_names if ht[n] == 0 and ht_p[n] > 0)

    # ── ⑥ 로그 실측 ──────────────────────────────────────────────────────────
    lines = sim_log_lines(tag, tid)
    lg = collections.Counter()
    proc_left_live, deny_bodies = [], collections.Counter()
    if lines is None:
        lg["NOLOG"] = 1
    else:
        for ln in lines:
            for k, rx in PATS.items():
                mm = rx.search(ln)
                if not mm:
                    continue
                lg[k] += 1
                if k == "PROC_checklist":
                    proc_left_live.append((mm.group(1), mm.group(4)))
                elif k == "TOOLOBS_err":
                    body = mm.group(1)
                    t = OURS_RE.match(body)
                    if t:
                        deny_bodies["[%s]" % t.group(1)] += 1
                    elif body.startswith(FB_GENERIC[:40]):
                        deny_bodies["_FB_GENERIC"] += 1
                    else:
                        deny_bodies["env"] += 1

    print("== %s  tag=%s  reward=%s  turns=%d asst=%d" %
          (tid, tag, getattr(sim, "reward_info", None) and
           getattr(sim.reward_info, "reward", "?"), len(msgs), asst))
    print("  PROCLEFT resign_pts=%d fire=%s" % (resign_pts, fire))
    print("  PROCLEFT_live checklist_lines=%d left_nonempty=%d sample=%s"
          % (lg["PROC_checklist"],
             sum(1 for _p, l in proc_left_live if l not in ("[]",)),
             proc_left_live[:2]))
    print("  EPLAN L1_deny=%d L2_deny=%d enum_calls=%s"
          % (lg["EPLAN_L1_deny"], lg["EPLAN_L2_deny"], enum_calls[:6]))
    print("  CALLFORM cdut=%s gave=%s givereq_log=%d" % (cdut[:4], gave[:4], lg["GIVEREQ"]))
    print("  STOPFIRST parallel_turns=%d max_calls=%d" % (par, mx))
    print("  DENYHOWTO ours_deny=%s folded=%d howto_tools=%s params_only=%s"
          % (dict(deny_bodies), lg["STACK_folded"], ht_hit[:8], htp_only[:8]))
    AGG["procleft_fire"] += 1 if fire else 0
    AGG["eplan_l1"] += lg["EPLAN_L1_deny"]
    AGG["eplan_l1_tasks"] += 1 if lg["EPLAN_L1_deny"] else 0
    AGG["eplan_l1_and_enum"] += 1 if (lg["EPLAN_L1_deny"] and enum_calls) else 0
    AGG["cdut_tasks"] += 1 if cdut else 0
    AGG["par_tasks"] += 1 if par else 0
    AGG["par_turns"] += par
    AGG["ourdeny_tasks"] += 1 if sum(v for k, v in deny_bodies.items() if k != "env") else 0
    AGG["ourdeny_events"] += sum(v for k, v in deny_bodies.items() if k != "env")
    AGG["nolog"] += lg["NOLOG"]
    print()

print("== AGG ==")
for k in sorted(AGG):
    print("  %-22s %d" % (k, AGG[k]))
print()
print("== howto 전수(궤적서 불린 도구) ==")
for n in sorted(HOWTO_CACHE and {k[0] for k in HOWTO_CACHE}):
    a, b = len(howto(n, False)), len(howto(n, True))
    if a or b:
        print("  %-46s base=%4d params=%4d" % (n, a, b))
