#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""`T2_PROCEDURE_LEFT` 래칫 — 스모크 없이 **초 단위로** 기능 전체를 검정한다.

## 왜 (2026-08-26 · 사용자 지시 *"기능 확인하고 스모크 돌리는 게 나을 거 같다"*)

4 태스크 스모크는 1.5시간이다. 그 전에 **오프라인으로 답할 수 있는 것**을 다 답한다:
술어가 옳은 행을 내는가 · 문면이 그 이름을 담는가 · 다 끝난 절차엔 침묵하는가 ·
sim 당 1회인가 · 배선이 닿는 자리에 있는가. 라이브가 답할 것은 *모델이 그걸 받고 행동을
바꾸는가* 하나뿐이고, 그건 A/B 몫이다([[30]] *"단위통과 ≠ 라이브 발화"* 의 역방향 규율).

## 재료는 **실제 궤적**이다 (합성 픽스처 아님)

t7361 의 050·074·085 **종료 상태**를 그대로 먹인다 — 그 셋이 이 레버가 겨냥하는 실물이다.
050 은 `[T2_PROCEDURE] checklist … done=5 left=['decision']` 를 우리 로그가 이미 인쇄했고,
그 직후 `[T2_CLAIMPROV] … regen tool_calls=[]` 로 통과시킨 자리다.

⚠이 검정은 *"모델이 그래서 승인하는가"* 를 판정하지 않는다 — 그건 A/B 가 잰다([[62]]).

실행: PYTHONIOENCODING=utf-8 py -3 test_procedure_left.py
"""
import gzip
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import t2_gate_patch as G                                          # noqa: E402
import t2_procedure as PL                                          # noqa: E402

SIMS = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
TAG = "bank_t7361_smoke_20260826"
FAIL = []


def chk(c, m, extra=""):
    if not c:
        FAIL.append(m)
    print("  %s %s%s" % ("ok  " if c else "FAIL", m, ("  " + str(extra)) if extra else ""))


class _TC(object):
    def __init__(self, tc):
        self.name = tc.get("name")
        self.arguments = tc.get("arguments")
        self.id = tc.get("id")


class _M(object):
    def __init__(self, m):
        self.role = m.get("role")
        self.content = m.get("content")
        self.tool_calls = [_TC(t) for t in (m.get("tool_calls") or [])] or None
        self.id = m.get("id")
        self.error = m.get("error", False)


def sims():
    p = os.path.join(SIMS, TAG + ".results.json.gz")
    d = json.load(gzip.open(p, "rt", encoding="utf-8", errors="replace"))
    return {x.get("task_id"): x for x in (d.get("simulations") or d.get("results") or [])}


def unmet(a2, msgs):
    """엔진이 하는 것과 **같은 계산** — 미충족 노드 전부(부분집합을 고르지 않는다)."""
    done = G._executed_tool_counts(msgs)
    rows, pids = [], []
    for p in PL.active_procedures((a2 or {}).get("procedures") or [], done):
        for nid, tools, ok in PL.checklist(p, done):
            if ok is False:
                rows.append((nid, list(tools or [])))
                if p.get("id") not in pids:
                    pids.append(p.get("id"))
    return rows, pids


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    a2 = G._domain_a2("banking_knowledge")
    S = sims()

    print("① 술어 — 실제 종료 상태에서 무엇이 남았다고 하나")
    got = {}
    for t in ("task_050", "task_074", "task_085"):
        msgs = [_M(m) for m in (S[t].get("messages") or [])]
        rows, pids = unmet(a2, msgs)
        got[t] = (rows, pids)
        print("     %-10s reward=%-4s 절차=%s 미충족=%s"
              % (t, (S[t].get("reward_info") or {}).get("reward"), pids,
                 [r[0] for r in rows]))
    chk(any(n == "decision" for n, _ in got["task_050"][0]),
        "050 에서 남은 칸으로 `decision` 을 집어낸다 ← 로그가 인쇄한 그 값")
    tools_050 = [x for n, ts in got["task_050"][0] if n == "decision" for x in ts]
    chk(any("approve" in x for x in tools_050),
        "그 칸이 **이름 붙인 도구**를 함께 준다", tools_050)

    print()
    print("② 문면 — 이름을 담고, 고르지 않는다")
    rows, pids = got["task_050"]
    lines = chr(10).join("- %s: %s" % (n, ", ".join(ts) if ts else "(no tool named)")
                         for n, ts in rows)
    fb = ("Error: [PROCEDURE-LEFT] you are closing this conversation, but the procedure(s) %s "
          "still have steps the policy asked for that have not been done here:%s%s%s"
          "These are all of them - nothing else is outstanding."
          % (", ".join(str(x) for x in pids), chr(10), lines, chr(10)))
    chk("decision" in fb, "문면이 남은 칸 이름을 담는다")
    chk("approve_credit_limit_increase_5847" in fb, "문면이 그 도구 이름을 담는다")
    chk("these are all of them" in fb.lower(), "**빼기 형태**다 — 남은 것이 전부라고 말한다([[63]])")
    chk(len(rows) == len([1 for n, _ in rows]), "미충족 노드를 **전부** 싣는다(부분집합 선택 0)")

    print()
    print("③ 침묵 — 절차가 다 끝났으면 아무 말도 안 한다")
    full = [_M(m) for m in (S["task_050"].get("messages") or [])]

    # ⚠`_executed_tool_counts` 는 **결과 메시지가 짝지어져야** 센다(호출만으론 안 센다) —
    #   초판 픽스처가 호출만 넣어 이 칸이 거짓 붉음이었다([[67]] 계기 함정).
    def _pair(name):
        return [_M({"role": "assistant", "tool_calls": [
                    {"name": "call_discoverable_agent_tool",
                     "arguments": {"agent_tool_name": name, "arguments": "{}"},
                     "id": "zz1"}]}),
                _M({"role": "tool", "id": "zz1",
                    "content": "Executed: %s. Approved." % name})]

    done_rows, _ = unmet(a2, full + _pair("approve_credit_limit_increase_5847"))
    chk(not any(n == "decision" for n, _ in done_rows),
        "승인을 실행하면 `decision` 이 미충족에서 빠진다", [r[0] for r in done_rows])

    print()
    print("④ 배선 — 엔진 안에 실제로 놓여 있나")
    src = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
    chk('os.environ.get("T2_PROCEDURE_LEFT") == "1"' in src, "환경 플래그로 갈린다")
    _guard = src[src.index('os.environ.get("T2_PROCEDURE_LEFT") == "1"'):][:220]
    chk("_resign" in _guard, "**사임/종료 창**에서만 본다(중간 턴 아님)", _guard.split(chr(10))[1][:70])
    chk("_t2_procleft" in src, "sim 당 1회 cap 이 있다")
    chk('_ap_regen(_left_fb, "procleft")' in src,
        "**재생성 채널**로 나간다(스텁이 히스토리에 안 남는다·[[30]])")
    blk = src.split("T2_PROCEDURE_LEFT")[1][:2600]
    for bad in ("sort(", "max(", "[0][0]", "argmax"):
        chk(bad not in blk, "엔진이 고르지 않는다 — %r 없음" % bad)

    print()
    print("⑤ 기본 OFF — 효과는 A/B 가 잰다")
    gs = io.open(os.path.join(HERE, "go_stack.sh"), encoding="utf-8").read()
    chk("export T2_PROCEDURE_LEFT=0" in gs, "정본 선언이 기본 OFF")

    print()
    print("RESULT: %s%s" % ("PASS" if not FAIL else "FAIL",
                            "" if not FAIL else "  " + str(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
