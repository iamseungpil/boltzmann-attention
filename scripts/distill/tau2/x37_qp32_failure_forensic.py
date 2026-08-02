#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x37: qp32p1 **실패 전수 per-step 포렌식** — 실패 원인 확정 (무료·로컬·영속 데이터).

사용자 지시(2026-08-02): *"실패한 태스크 전수 per step 포렌식 돌려서 실패 원인 확정하라."*

규율:
  · [[08]] 집계에서 결론 직행 금지 — 태스크마다 **채점 기준(`reward_basis`)이 다르다**(C251: DB 24 / ACTION 8).
    먼저 기준을 읽고, 그 기준이 무엇 때문에 깨졌는지를 per-step으로 짚는다.
  · **C274 오염 회피**: 공식 `action_checks`는 중첩 JSON 문자열을 리터럴 비교해 miss의 24.5%가 가짜였다.
    ⇒ 여기서는 `compare_args`만 놓고 **의미 대조**(중첩 JSON 파싱·공백/대소문자 정규화)를 따로 계산하고,
    **공식 판정과 불일치하는 건수를 같이 보고**한다(진단은 내 대조로, 판정은 공식대로 — 섞지 않는다).
  · [[03b]] 정규화는 **진단 전용**이다. 이 스크립트는 어떤 점수도 바꾸지 않는다.

산출: 실패 sim마다 ⑴채점 기준 ⑵깨진 지점(누락 write / 인자 불일치 / 여분 write / 비정상 종료)
     ⑶ per-step 근거(실제 호출 인자) → 마지막에 기전별 군집.
"""
import argparse
import collections
import glob
import gzip
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--simdir", default=os.path.join(HERE, "..", "..", "..",
                                                 "reports", "facet_rft_2026", "sim_results"))
ap.add_argument("--tags", default="bank_qp32p1_gpu0_20260802,bank_qp32p1_gpu1_20260802")
ap.add_argument("--detail", type=int, default=0, help="1이면 sim별 호출 목록까지 출력")
A = ap.parse_args()


def norm(v):
    """의미 대조용 정규화 — 중첩 JSON 문자열을 풀고, 공백/대소문자를 흡수한다(진단 전용)."""
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in "[{":
            try:
                return norm(json.loads(s))
            except Exception:
                pass
        return " ".join(s.split()).lower()
    if isinstance(v, dict):
        return {str(k).lower(): norm(x) for k, x in sorted(v.items())}
    if isinstance(v, list):
        return [norm(x) for x in v]
    if isinstance(v, (int, float)):
        return float(v)
    return v


def calls_of(sim):
    """★계측기 정정(2026-08-02): 초판은 **assistant 호출만** 모아 `call_discoverable_user_tool`
    (손님이 실행·`role='user'`·`requestor='user'`)을 전부 '미실행'으로 셌다 — 실패의 20/24가 한
    기전에 몰린 것이 그 아티팩트였다. 모든 role의 tool_calls를 모으고 **requestor를 보존**한다."""
    byid = {m.get("id"): str(m.get("content") or "")
            for m in (sim.get("messages") or []) if (m.get("role") or "") == "tool"}
    out = []
    for m in (sim.get("messages") or []):
        for tc in (m.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
            a = fn.get("arguments", tc.get("arguments"))
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    a = {"__raw": a}
            out.append({"name": str(fn.get("name") or ""), "args": a if isinstance(a, dict) else {},
                        "req": tc.get("requestor") or m.get("role"),
                        "out": byid.get(tc.get("id"), "")})
    return out


SIMS = []
for tag in A.tags.split(","):
    p = os.path.join(A.simdir, tag.strip() + ".results.json.gz")
    if not os.path.exists(p):
        print("⚠없음: %s" % p)
        continue
    d = json.load(gzip.open(p, "rt", encoding="utf-8"))
    for s in d.get("simulations", []):
        SIMS.append((tag.strip(), s))
print("적재 %d sim" % len(SIMS))

fails = [(t, s) for t, s in SIMS if ((s.get("reward_info") or {}).get("reward") or 0) < 1]
print("실패 %d / 성공 %d\n" % (len(fails), len(SIMS) - len(fails)))

MECH = collections.Counter()
DISAGREE = 0
DIR = collections.Counter()
DIR_EX = []
rows = []
for tag, s in fails:
    ri = s.get("reward_info") or {}
    basis = ri.get("reward_basis") or []
    bd = ri.get("reward_breakdown") or {}
    acts = ri.get("action_checks") or []
    calls = calls_of(s)
    writes = [c for c in calls if c["name"] not in ("KB_search", "KB_search_bm25", "KB_search_dense")]
    # ── 의미 대조 (compare_args 한정) ─────────────────────────────────────────
    missing, argmiss, matched = [], [], []
    used = set()                       # ★소비: 한 호출이 여러 gold에 중복 매칭되지 않게
    for ac in acts:
        a = ac.get("action") or {}
        nm, want = a.get("name"), (a.get("arguments") or {})
        req = a.get("requestor")
        cmp_keys = a.get("compare_args")
        cand = [(i, c) for i, c in enumerate(calls)
                if (c["name"] == nm or c["name"].startswith(str(nm) + "_"))
                and (req is None or c["req"] == req)]
        if not cand:
            missing.append((nm, ac.get("action_match")))
            continue
        # ★compare_args=[] 는 "비교할 인자 없음"(존재만 보면 됨) — 전체 인자 폴백은 내 버그였다.
        keys = cmp_keys if cmp_keys is not None else list(want.keys())
        hit = None
        for i, c in cand:
            if i in used:
                continue
            if all(norm(c["args"].get(k)) == norm(want.get(k)) for k in keys):
                hit = c
                used.add(i)
                break
        if hit is None:
            cand = [c for _i, c in cand]
            diffs = []
            c0 = cand[0]
            for k in keys:
                if norm(c0["args"].get(k)) != norm(want.get(k)):
                    diffs.append("%s: gold=%r got=%r" % (k, want.get(k), c0["args"].get(k)))
            argmiss.append((nm, diffs))
        else:
            matched.append(nm)
        if (hit is not None) != bool(ac.get("action_match")):
            DISAGREE += 1
            d = "공식miss·내match(C274형 오염)" if hit is not None else "공식match·내miss(내 대조가 엄격)"
            DIR[d] += 1
            if len(DIR_EX) < 8:
                DIR_EX.append((s.get("task_id"), nm, d))
    gold_names = {(x.get("action") or {}).get("name") for x in acts}
    extra_w = [c["name"] for c in writes
               if c["name"] not in gold_names
               and not c["name"].startswith(("get_", "check_", "verify_", "list_", "search"))]
    term = s.get("termination_reason")
    # ── 기전 판정 (순서 = 배타적) ─────────────────────────────────────────────
    if term in ("context_window_exceeded", "max_steps", "too_many_errors", "infrastructure_error"):
        mech = "ⓔ비정상 종료(%s)" % term
    elif not acts:
        mech = "ⓕ gold 액션 없음(기준=%s)" % ",".join(basis)
    elif missing:
        mech = "ⓐ gold 액션 미실행 %d건" % len(missing)
    elif argmiss:
        mech = "ⓑ 인자 불일치 %d건" % len(argmiss)
    elif extra_w:
        mech = "ⓒ 여분 write(over-action) %d건" % len(set(extra_w))
    else:
        mech = "ⓓ 액션은 다 맞았는데 실패(DB 해시·상태 차이)"
    MECH[mech.split("(")[0].split(" ")[0]] += 1
    rows.append({"task": s.get("task_id"), "gpu": tag[-4:], "basis": ",".join(basis),
                 "bd": bd, "term": term, "ncall": len(calls), "mech": mech,
                 "missing": missing, "argmiss": argmiss, "extra": sorted(set(extra_w)),
                 "matched": matched, "nacts": len(acts)})

rows.sort(key=lambda r: (r["mech"], r["task"]))
print("=" * 96)
print("실패 전수 (기전순)")
for r in rows:
    print("\n── %s [%s] · 기준=%s %s · 종료=%s · 호출 %d · gold액션 %d"
          % (r["task"], r["gpu"], r["basis"], r["bd"], r["term"], r["ncall"], r["nacts"]))
    print("   기전: %s" % r["mech"])
    for nm, off in r["missing"][:4]:
        print("     · 미실행 gold: %s   (공식 action_match=%s)" % (nm, off))
    for nm, diffs in r["argmiss"][:3]:
        print("     · 인자 불일치: %s" % nm)
        for d in diffs[:3]:
            print("         %s" % d[:150])
    if r["extra"]:
        print("     · 여분 write: %s" % ", ".join(r["extra"][:6]))

print("\n" + "=" * 96)
print("기전 군집")
for k, v in MECH.most_common():
    print("  %-28s %d" % (k, v))
print("\n★공식 action_match ↔ 내 의미 대조 불일치 %d건 — **방향별**" % DISAGREE)
for k, v in DIR.most_common():
    print("   %-40s %d" % (k, v))
for t, nm, d in DIR_EX:
    print("     예: %-10s %-32s → %s" % (t, nm, d))
