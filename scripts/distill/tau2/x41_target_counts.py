#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x41: 두 설계서가 GO 조건으로 건 **무료 표적 계수 5종** (전수·로컬·GPU 0).

정본 = `FAILURE_AXES_REDESIGN_2026_08_02` §6 · `RUNAWAY_AXIS_REDESIGN_2026_08_02` §6.
규율(C263): **표적이 공집합이면 그 처방은 NO-GO.** 비율 단독 금지 — 반드시 **시뮬 수 병기**([[08]]).

A 채널 오분류   : 도구 종류(스캐폴드/agent-discoverable/user-discoverable)를 어긋나게 호출한 건수
                  ★레지스트리는 **env 반환에서 도출**한다(성공한 unlock/give의 축자) — 우리가 이름을 적지 않는다.
B 터미널-턴     : 손님 발화에 이관 토큰이 실재하는데 그 뒤로 transfer 호출이 없는 sim
C fit 판별력    : `check_card_application_fit` 반환 적격 카드 수 분포
D 대형 read 재유입: 동일 (도구,인자) 대형 출력이 2회 이상 **전문** 재유입된 sim + 절감 가능 문자량
E 배치화        : 스칼라 인자 자리에 **배열**을 넣은 호출(029형)
"""
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
SIMDIR = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
BIG = 6000                    # 대형 read 임계(문자)

SIMS = []
for f in sorted(glob.glob(os.path.join(SIMDIR, "*.results.json.gz"))):
    try:
        d = json.load(gzip.open(f, "rt", encoding="utf-8"))
    except Exception:
        continue
    tag = os.path.basename(f).replace(".results.json.gz", "")
    for s in d.get("simulations", []):
        byid = {m.get("id"): str(m.get("content") or "")
                for m in (s.get("messages") or []) if (m.get("role") or "") == "tool"}
        seq = []
        for m in (s.get("messages") or []):
            r = m.get("role") or ""
            if r == "user" and (m.get("content") or "").strip():
                seq.append({"k": "user", "text": str(m.get("content"))})
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
                seq.append({"k": "call", "name": str(fn.get("name") or ""),
                            "args": a if isinstance(a, dict) else {},
                            "req": tc.get("requestor") or r,
                            "out": byid.get(tc.get("id"), "")})
        SIMS.append({"tag": tag, "task": s.get("task_id"), "seq": seq,
                     "calls": [x for x in seq if x["k"] == "call"]})
print("전수 %d sim · 총 호출 %d\n" % (len(SIMS), sum(len(s["calls"]) for s in SIMS)))

# ── 레지스트리 도출 (env 반환 축자) ───────────────────────────────────────────
AGENT_T, USER_T = set(), set()
for s in SIMS:
    for c in s["calls"]:
        o = c["out"] or ""
        nm = (c["args"].get("agent_tool_name") or c["args"].get("discoverable_tool_name") or "")
        if c["name"] == "unlock_discoverable_agent_tool" and "Tool unlocked:" in o:
            m = re.search(r"Tool unlocked:\s*([A-Za-z0-9_]+)", o)
            if m:
                AGENT_T.add(m.group(1))
        if c["name"] == "give_discoverable_user_tool" and nm and "rror" not in o[:40]:
            USER_T.add(str(nm))
SCAFFOLD = {"get_reward_discrepancies", "check_card_application_fit", "verify_identity",
            "check_card_closure_eligibility", "get_interest_correction",
            "check_rebate_qualification"}
print("[레지스트리 도출] agent-discoverable %d · user-discoverable %d (env 반환 축자에서)"
      % (len(AGENT_T), len(USER_T)))
print("  agent: %s" % ", ".join(sorted(AGENT_T)[:8]))
print("  user : %s" % ", ".join(sorted(USER_T)[:8]))

# ── A 채널 오분류 ─────────────────────────────────────────────────────────────
print("\n" + "=" * 88)
print("A. 채널 오분류 — 종류가 어긋난 unlock/give/call")
kinds = collections.Counter()
Asims = set()
for s in SIMS:
    for c in s["calls"]:
        nm = str(c["args"].get("agent_tool_name") or c["args"].get("discoverable_tool_name") or "")
        if not nm:
            continue
        if c["name"] == "unlock_discoverable_agent_tool":
            if nm in SCAFFOLD:
                kinds["스캐폴드 도구를 unlock"] += 1
                Asims.add((s["tag"], s["task"]))
            elif nm in USER_T and nm not in AGENT_T:
                kinds["user 도구를 agent로 unlock"] += 1
                Asims.add((s["tag"], s["task"]))
        if c["name"] == "give_discoverable_user_tool" and nm in AGENT_T and nm not in USER_T:
            kinds["agent 도구를 user에게 give"] += 1
            Asims.add((s["tag"], s["task"]))
        if c["name"] == "call_discoverable_agent_tool" and nm in USER_T and nm not in AGENT_T:
            kinds["user 도구를 agent 채널로 call"] += 1
            Asims.add((s["tag"], s["task"]))
for k, v in kinds.most_common():
    print("  %-30s %4d" % (k, v))
print("  ⇒ 총 %d회 · **%d sim** %s" % (sum(kinds.values()), len(Asims),
                                      "GO" if len(Asims) >= 5 else "표적 소규모 — 재검토"))

# ── B 터미널-턴 ───────────────────────────────────────────────────────────────
print("\n" + "=" * 88)
print("B. 터미널-턴 — 손님이 이관을 요청/동의했는데 그 뒤 transfer 호출 없음")
TOK = ("###TRANSFER###",)
Bsims, Bhits = set(), 0
for s in SIMS:
    idx = [i for i, e in enumerate(s["seq"])
           if e["k"] == "user" and any(t in e["text"] for t in TOK)]
    if not idx:
        continue
    after = [e for e in s["seq"][idx[0] + 1:]
             if e["k"] == "call" and e["name"] == "transfer_to_human_agents"]
    if not after:
        Bhits += 1
        Bsims.add((s["tag"], s["task"]))
print("  이관 토큰 실재 sim 중 이후 transfer 호출 **없음** = %d sim" % len(Bsims))
print("  예: %s" % ", ".join("%s/%s" % t for t in sorted(Bsims)[:6]))
print("  ⇒ %s" % ("GO" if len(Bsims) >= 5 else "표적 소규모 — 재검토"))

# ── C fit 판별력 ──────────────────────────────────────────────────────────────
print("\n" + "=" * 88)
print("C. `check_card_application_fit` 적격 카드 수 분포")
dist = collections.Counter()
Csims = set()
for s in SIMS:
    for c in s["calls"]:
        if c["name"] != "check_card_application_fit":
            continue
        n = len(re.findall(r"'card': '([^']+)'", c["out"] or ""))
        if n:
            dist[n] += 1
            if n >= 2:
                Csims.add((s["tag"], s["task"]))
for k in sorted(dist):
    print("   적격 %2d장 : %4d회" % (k, dist[k]))
tot = sum(dist.values())
multi = sum(v for k, v in dist.items() if k >= 2)
print("  ⇒ 호출 %d 중 **≥2장 = %d (%.0f%%)** · %d sim %s"
      % (tot, multi, 100.0 * multi / max(tot, 1), len(Csims),
         "GO" if len(Csims) >= 5 else "표적 소규모"))

# ── D 대형 read 전문 재유입 ───────────────────────────────────────────────────
print("\n" + "=" * 88)
print("D. 대형 read(>%d자) 동일 호출의 **전문** 재유입" % BIG)
Dsims, waste, events = set(), 0, 0
for s in SIMS:
    seen = {}
    for c in s["calls"]:
        k = (c["name"], json.dumps(c["args"], sort_keys=True, ensure_ascii=False))
        o = c["out"] or ""
        if len(o) < BIG or "[DUPLICATE-READ]" in o:
            continue
        if k in seen:
            events += 1
            waste += len(o)
            Dsims.add((s["tag"], s["task"]))
        seen[k] = len(o)
print("  전문 재유입 %d회 · **%d sim** · 절감 가능 %s자"
      % (events, len(Dsims), "{:,}".format(waste)))
print("  예: %s" % ", ".join("%s/%s" % t for t in sorted(Dsims)[:6]))
print("  ⇒ %s" % ("GO" if len(Dsims) >= 5 else "표적 소규모 — 재검토"))

# ── E 배치화 ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 88)
print("E. 배치화 — 스칼라 인자 자리에 배열")
Esims, ehits = set(), collections.Counter()
for s in SIMS:
    for c in s["calls"]:
        def scan(d, path=""):
            if isinstance(d, dict):
                for k2, v in d.items():
                    if isinstance(v, str) and v.strip()[:1] == "{":
                        try:
                            scan(json.loads(v), path + k2 + ".")
                            continue
                        except Exception:
                            pass
                    if isinstance(v, list) and len(v) > 1 and all(isinstance(x, str) for x in v):
                        ehits[path + k2] += 1
                        Esims.add((s["tag"], s["task"]))
                    elif isinstance(v, dict):
                        scan(v, path + k2 + ".")
        scan(c["args"])
for k, v in ehits.most_common(8):
    print("   %-40s %3d" % (k, v))
print("  ⇒ 총 %d회 · **%d sim** %s" % (sum(ehits.values()), len(Esims),
                                      "GO" if len(Esims) >= 5 else "표적 소규모 — 재검토"))
