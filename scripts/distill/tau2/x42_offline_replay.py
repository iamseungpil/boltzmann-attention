#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x42: 구현된 축-레버를 **실제 실패 궤적에 오프라인 재생** (무료·로컬·GPU 0).

픽스처 통과 ≠ 실데이터 발화([[30]]). 여기서는 qp32p1 실패 궤적의 **실제 인자·실제 반환**에
구현 함수를 그대로 먹여, 각 레버가 **언제 몇 번** 발화하는지와 **어느 태스크의 결정점을 짚는지**를 센다.
판정은 바꾸지 않는다(순수 관측).
"""
import collections
import glob
import gzip
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import t2_axis_levers as AX      # noqa: E402

TPL = json.load(open(os.path.join(HERE, "a2", "base", "shared.json"),
                     encoding="utf-8"))["axis_notes"]
SIMDIR = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
TAGS = ["bank_qp32p1_gpu0_20260802", "bank_qp32p1_gpu1_20260802"]

# 레지스트리 = env 반환 축자에서 도출(런타임 도출과 동형·오프라인 대체)
AGENT_T, USER_T = set(), set()
ALL = []
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
            if r in ("assistant", "user") and (m.get("content") or "").strip():
                seq.append({"k": r, "text": str(m.get("content"))})
            for tc in (m.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                a = fn.get("arguments", tc.get("arguments"))
                if isinstance(a, str):
                    try:
                        a = json.loads(a)
                    except Exception:
                        a = {}
                o = byid.get(tc.get("id"), "")
                seq.append({"k": "call", "name": str(fn.get("name") or ""),
                            "args": a if isinstance(a, dict) else {}, "out": o})
                if "Tool unlocked:" in o:
                    mm = re.search(r"Tool unlocked:\s*([A-Za-z0-9_]+)", o)
                    if mm:
                        AGENT_T.add(mm.group(1))
                if (str(fn.get("name")) == "give_discoverable_user_tool"
                        and "rror" not in o[:40]):
                    nm = (a or {}).get("discoverable_tool_name")
                    if nm:
                        USER_T.add(str(nm))
        ALL.append({"tag": tag, "task": s.get("task_id"), "seq": seq})
SCAFFOLD = {"get_reward_discrepancies", "check_card_application_fit", "verify_identity",
            "check_card_closure_eligibility", "get_interest_correction",
            "check_rebate_qualification"}
print("레지스트리(전수 도출): agent %d · user %d · scaffold %d"
      % (len(AGENT_T), len(USER_T), len(SCAFFOLD)))

TARGET = [s for s in ALL if s["tag"] in TAGS]
print("재생 대상 = qp32p1 %d sim\n" % len(TARGET))

fire = collections.Counter()
bytask = collections.defaultdict(collections.Counter)
for s in TARGET:
    unlocked, called = set(), set()
    said = ""
    # ★엔진과 동일한 발화 상한(T2_AXIS_NOTE_CAP=2) — 상한 없이 재생하면 026에서 55회가 나온다
    _fired = {}

    def _allow(k):
        _fired[k] = _fired.get(k, 0) + 1
        return _fired[k] <= 2
    for e in s["seq"]:
        if e["k"] == "assistant":
            said += " " + e["text"]
            continue
        if e["k"] == "user":
            n = AX.terminal_turn_note(e["text"], TPL["transfer_tokens"],
                                      any("transfer" in c for c in called), TPL)
            if n and _allow(("terminal", "")):
                fire["terminal_turn(잠재)"] += 1
                bytask[s["task"]]["terminal_turn"] += 1
            continue
        called.add(e["name"])
        a = e["args"] or {}
        for k in ("agent_tool_name", "discoverable_tool_name"):
            if a.get(k):
                called.add(str(a[k]))
        if "Tool unlocked:" in (e["out"] or ""):
            mm = re.search(r"Tool unlocked:\s*([A-Za-z0-9_]+)", e["out"])
            if mm:
                unlocked.add(mm.group(1))
        n = AX.channel_note(e["name"], a, SCAFFOLD, AGENT_T, USER_T, unlocked, TPL)
        if n:
            fire["channel"] += 1
            bytask[s["task"]]["channel"] += 1
        n = AX.scalar_array_note(a, TPL)
        if n:
            fire["scalar_array"] += 1
            bytask[s["task"]]["scalar_array"] += 1
        if e["name"] == "check_card_application_fit":
            n = AX.fit_diff_note(e["out"] or "", TPL)
            if n:
                fire["fit_diff"] += 1
                bytask[s["task"]]["fit_diff"] += 1
        for _m in AX.mention_note(said, called, AGENT_T, USER_T, unlocked, TPL):
            if _allow(("mention", _m[:60])):
                fire["mention"] += 1
                bytask[s["task"]]["mention"] += 1

print("=" * 80)
print("[레버별 발화 수 — 실제 궤적 재생]")
for k, v in fire.most_common():
    print("  %-22s %5d" % (k, v))
print("\n[태스크별 — 어느 실패의 결정점을 짚었나]")
for t in sorted(bytask):
    print("  %-10s %s" % (t, dict(bytask[t])))
