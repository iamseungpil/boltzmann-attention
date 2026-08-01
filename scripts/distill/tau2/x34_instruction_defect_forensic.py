#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x34: **폭주 궤적의 지시-결함** 전수 포렌식 (무료·로컬·영속 데이터만).

사용자 지시(2026-08-01): *"대부분의 무한 폭주는 지시사항의 문제로 보인다. 이전의 폭주 사례
정밀 per-step 포렌식하여 지시사항이 모순이거나, 지시가 없거나, 잘못된 경우를 모두 찾아라."*

배경 = x33 정정: 008 폭주는 "스텁 무시"가 아니라 **우리 A2 `no_record_template`이 종료 조건 없는
루프를 지시**한 것이었다(조회→`No records found`→"call this tool again"→무한). 027은 **도달 수단
부재**였다. ⇒ 폭주를 에이전트 결함으로 분류하기 전에 **우리가 뭐라고 말했는지**를 먼저 센다.

방법(per-step·집계 라벨 불신·[[08]]):
  1. 폭주 시뮬(호출>임계) 전수에서 **엔진/스캐폴드 저작 텍스트**를 마커로 추출(도구 출력에 우리가
     주입한 것 — env 원문과 구분).
  2. **반복 사건**(같은 (도구,인자) 재발행) 직전의 엔진 텍스트를 귀속 → 템플릿별 계수.
  3. 템플릿을 축자로 출력한다. 분류(모순/부재/불가능/허위)는 **사람이 원문을 읽고** 한다 —
     스크립트가 의미를 판정하면 그게 곧 [[03b]] 위반이다. 여기선 *증거만* 만든다.
  4. 각 템플릿에 **이행 가능성 증거**를 붙인다: 그 문구가 지시한 도구를 실제로 불렀는가,
     그 결과가 무엇이었나(예: `No records found`가 반복되면 전제 불성립 = 이행 불가능).

출력 = 템플릿 랭킹(반복-유발 순) + 축자 + 사례 + 이행 증거.
"""
import argparse
import collections
import glob
import gzip
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.join(HERE, "..", "..", "..",
                                              "reports", "facet_rft_2026", "sim_results"))
ap.add_argument("--glob", default="*.results.json.gz")
ap.add_argument("--calls_hi", type=int, default=30)
ap.add_argument("--top", type=int, default=18)
ap.add_argument("--out", default="")
A = ap.parse_args()

# ── 엔진/스캐폴드 저작 텍스트의 마커 ─────────────────────────────────────────
#   대괄호 태그는 전부 우리 것이고, 나머지는 A2 템플릿의 고유 선두 어구다.
MARKERS = [
    "[DUPLICATE-READ]", "[GROUNDING WARNING]", "[GUIDANCE]", "[coverage]", "[quote-pin]",
    "[T2_", "★FEEDBACK", "[UNAVAILABLE]", "[POLICY]", "[DENIED]", "[REMINDER]", "[PLAN]",
    "[CHAIN]", "[VERIFY]", "[STEP]", "[NOTE]", "[ABSTAIN]", "[PRE-ACTION",
    "NOT_VERIFIED", "Failed to log", "could not be verified", "you must",
]
# A2에서 실제 문구를 끌어와 마커를 보강(도메인 문구를 스크립트에 박지 않기 위함)
A2_LEADS = []
for fn in ("banking_knowledge.specific.json", "banking_knowledge.gate.json"):
    p = os.path.join(HERE, "a2", fn)
    if not os.path.exists(p):
        continue
    try:
        blob = json.load(open(p, encoding="utf-8"))
    except Exception:
        continue

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, str) and len(v) > 60 and (
                        k.endswith("_template") or k.endswith("_note") or k.endswith("_prompt")
                        or k.endswith("_msg") or "reminder" in k or "feedback" in k):
                    A2_LEADS.append((k, v))
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(blob)
LEAD_BY_TEXT = {re.sub(r"\{[^}]*\}", "", v)[:40].strip(): k for k, v in A2_LEADS}
print("A2 문구 후보 %d개 로드(마커 보강용)" % len(A2_LEADS))


def engine_spans(text):
    """도구 출력에서 엔진 저작 조각들을 (마커, 축자) 로 뽑는다."""
    out = []
    t = str(text or "")
    for m in MARKERS:
        i = t.find(m)
        if i >= 0:
            out.append((m, t[i:i + 700]))
    for lead, key in LEAD_BY_TEXT.items():
        if len(lead) >= 25 and lead in t:
            out.append(("A2:" + key, t[t.find(lead):t.find(lead) + 700]))
    return out


def sig(marker, span):
    """템플릿 서명 = 마커 + 가변부(숫자·따옴표 내용) 제거한 선두."""
    s = re.sub(r"'[^']*'", "'…'", span)
    s = re.sub(r'"[^"]*"', '"…"', s)
    s = re.sub(r"\d+", "#", s)
    s = re.sub(r"\s+", " ", s).strip()
    return (marker, s[:150])


FILES = sorted(glob.glob(os.path.join(A.dir, A.glob)))
print("입력 %d파일" % len(FILES))

TPL = collections.defaultdict(lambda: {"n_seen": 0, "n_then_repeat": 0, "n_then_same": 0,
                                       "sims": set(), "verbatim": "", "next_tools": collections.Counter()})
RUNAWAYS = []
NORECORD = collections.Counter()

for f in FILES:
    try:
        d = json.load(gzip.open(f, "rt", encoding="utf-8"))
    except Exception:
        continue
    tag = os.path.basename(f).replace(".results.json.gz", "")
    for s in d.get("simulations", []):
        msgs = s.get("messages") or []
        byid, calls = {}, []
        for m in msgs:
            if (m.get("role") or "") == "tool":
                byid[m.get("id") or m.get("tool_call_id")] = str(m.get("content") or "")
        for m in msgs:
            if (m.get("role") or "") != "assistant":
                continue
            for tc in (m.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                args = fn.get("arguments", tc.get("arguments"))
                if not isinstance(args, str):
                    args = json.dumps(args, sort_keys=True, ensure_ascii=False)
                calls.append({"name": str(fn.get("name") or tc.get("name")), "args": args[:4000],
                              "out": byid.get(tc.get("id"), "")})
        if len(calls) <= A.calls_hi:
            continue
        RUNAWAYS.append((tag, s.get("task_id"), s.get("trial"), s.get("termination_reason"),
                         len(calls)))
        seen = set()
        for i, c in enumerate(calls):
            key = (c["name"], c["args"])
            nxt = calls[i + 1] if i + 1 < len(calls) else None
            for marker, span in engine_spans(c["out"]):
                g = sig(marker, span)
                e = TPL[g]
                e["n_seen"] += 1
                e["sims"].add("%s/%s" % (tag, s.get("task_id")))
                if not e["verbatim"]:
                    e["verbatim"] = span
                if nxt is not None:
                    e["next_tools"][nxt["name"]] += 1
                    nk = (nxt["name"], nxt["args"])
                    if nk in seen or nk == key:
                        e["n_then_repeat"] += 1
                    if nk == key:
                        e["n_then_same"] += 1
            if "No records found" in (c["out"] or ""):
                NORECORD[c["name"]] += 1
            seen.add(key)

print("폭주 시뮬 %d건 (호출>%d)\n" % (len(RUNAWAYS), A.calls_hi))
print("=" * 92)
print("[템플릿 랭킹] 이 문구가 뜬 **직후** 에이전트가 이미 낸 호출을 다시 낸 횟수 순")
print("  n_seen=발화 · then_repeat=직후 재호출 · then_SAME=직후 **완전 동일** 재호출\n")
rank = sorted(TPL.items(), key=lambda kv: -kv[1]["n_then_repeat"])
for g, e in rank[:A.top]:
    marker, s150 = g
    print("─" * 92)
    print("MARKER %s  |  n_seen %4d · then_repeat %4d · **then_SAME %4d** · 시뮬 %d개"
          % (marker, e["n_seen"], e["n_then_repeat"], e["n_then_same"], len(e["sims"])))
    print("  축자: %s" % e["verbatim"][:520].replace("\n", " | "))
    print("  직후 도구 top: %s" % ", ".join("%s×%d" % (k, v) for k, v in e["next_tools"].most_common(4)))
    print("  사례: %s" % ", ".join(sorted(e["sims"])[:3]))

print("\n" + "=" * 92)
print("[전제 불성립 증거] 폭주 궤적에서 'No records found'를 돌려준 도구")
for k, v in NORECORD.most_common(10):
    print("  %-42s %4d" % (k, v))

if A.out:
    json.dump({"runaways": RUNAWAYS,
               "templates": [{"marker": g[0], "sig": g[1], "verbatim": e["verbatim"],
                              "n_seen": e["n_seen"], "n_then_repeat": e["n_then_repeat"],
                              "n_then_same": e["n_then_same"], "sims": sorted(e["sims"])[:20],
                              "next_tools": e["next_tools"].most_common(6)}
                             for g, e in rank]},
              open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n→ %s" % A.out)
