# -*- coding: utf-8 -*-
"""P2 — 에코-그라운딩 수리 재현 검정 (2026-08-21 · t7335 094 실측 · C203 기지 결함).

결함(수리 전): `[GROUNDING WARNING]` 경고문이 도구 출력으로 원장(ledger corpus)에 남아,
1차에 드롭된 값(094: actual_apy=5.25·period_start=10/01/2025)이 **경고문 문자열로 실재**하게
되고, 같은 값의 2차 재전송이 존재 검사를 통과했다(1차 저지의 자기-무력화).

수리: `_strip_own_feedback` — grounding 대조 코퍼스(ledger)에 한해, 우리 층이 찍는 태그
(`[GROUNDING WARNING]`)부터 그 줄 끝까지 제거. 모델이 보는 반환문은 불변.

검정 축:
  ① 094 재현 — 경고문 속 값(수치·날짜)은 접지 출처로 **불인정**(드롭)
  ② 결함 재현(양성 대조) — strip 을 무력화하면 같은 값이 **통과**한다(수리 전 거동 = 결함 실재)
  ③ 정상 회귀 — 진짜 레코드 덤프에 있는 값은 여전히 통과(false-drop 0)
  ④ 뒷줄 보존 — 경고문 **뒷줄**(도구 산출 본문)의 값은 여전히 접지 출처로 유효
  ⑤ `_strip_own_feedback` 단위 — 다중 태그·개행 없는 꼬리·비-경고 텍스트 보존

오프라인 전용(유료 X·[[09]]). 실행: py -3 test_ground_warning_echo.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_scaffold_get as sg  # noqa: E402

A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
DECLS = {d["name"]: d for d in A2["scaffold_get_tools"]}

# ── 통제 코퍼스 ────────────────────────────────────────────────────────────────
# 094 1차 호출이 남긴 경고 에코(빌드부 `:2382` 포맷 축자·값 5.25/기간은 레코드에 없는 값).
WARN_ECHO = ("[GROUNDING WARNING] 3 input value(s) could not be verified against the account "
             "records / knowledge base and were dropped: actual_apy=5.25 (not found in the "
             "records — re-read the exact value); period_start=10/01/2025 (not found in the "
             "records — re-read the exact value); period_end=10/31/2025 (not found in the "
             "records — re-read the exact value).\n"
             "Interest correction could not be computed - check your arguments.")
# 진짜 레코드 덤프(env 기계 포맷) — principal 96000 만 실재. 5.25/10월 기간은 없다.
DUMP = ("Found 1 record(s) in 'bank_accounts':\n"
        "1. Record ID: sav_gold_1\n   current_holdings: 96000.00\n"
        "   last interest line: MONTHLY INTEREST CREDIT $408.00 on 11/01/2025")


class _FakeEnv:
    domain_name = "banking"


class _FakeOrch:
    environment = _FakeEnv()


def _install(outputs):
    """엔진 코퍼스 소스를 통제값으로 몽키패치(라이브 원장 대신·`_evidence_ctx` 동형 lower)."""
    sg._DOC_CACHE.clear()
    sg._load_domain_docs = lambda domain: []
    sg._evidence_ctx = lambda orch: {
        "__tool_outputs": {k: v.lower() for k, v in outputs.items()},
        "__user_text": "",
    }


def _run(name, ctx):
    return sg._ground_operands(_FakeOrch(), DECLS[name], ctx)


PASS, FAIL = [], []


def check(label, cond):
    (PASS if cond else FAIL).append(label)
    print(("  PASS " if cond else "  FAIL ") + label)


def main():
    # ── ① 094 재현: 경고문 속 값은 접지 출처가 아니다 ────────────────────────────
    print("[094 재현] 경고 에코만 원장에 있는 값(5.25·10/01/2025)은 드롭된다")
    _install({"get_interest_correction": WARN_ECHO, "call_discoverable_agent_tool": DUMP})
    ctx = {"expected_apy": 5.5, "actual_apy": 5.25, "principal": 96000,
           "period_start": "10/01/2025", "period_end": "10/31/2025"}
    flags = _run("get_interest_correction", ctx)
    check("094: actual_apy=5.25 (경고 에코) 드롭", ctx["actual_apy"] is None)
    check("094: period_start (경고 에코) 드롭", ctx["period_start"] is None)
    check("094: period_end (경고 에코) 드롭", ctx["period_end"] is None)
    check("094: principal=96000 (진짜 덤프) 유지", ctx["principal"] == 96000)
    check("094: 플래그 3건(에코 3필드만)", len(flags) == 3)

    # ── ② 결함 재현(양성 대조): strip 무력화 = 수리 전 거동 = 에코 통과 ─────────────
    print("[결함 재현] _strip_own_feedback 를 무력화하면 같은 값이 통과한다(수리 전 거동)")
    _orig = sg._strip_own_feedback
    sg._strip_own_feedback = lambda t: t          # 수리 전 거동 에뮬레이션
    try:
        ctx = {"expected_apy": 5.5, "actual_apy": 5.25, "principal": 96000,
               "period_start": "10/01/2025", "period_end": "10/31/2025"}
        flags = _run("get_interest_correction", ctx)
        check("결함: strip 없이는 5.25 가 '실재'로 통과(자기-무력화 재현)",
              ctx["actual_apy"] == 5.25 and len(flags) == 0)
    finally:
        sg._strip_own_feedback = _orig

    # ── ③ 정상 회귀: 진짜 덤프에 있는 값은 여전히 통과 ──────────────────────────────
    print("[정상 회귀] 레코드 덤프 실재값(408 파생 아님·96000)은 false-drop 0")
    _install({"get_interest_correction": WARN_ECHO,
              "call_discoverable_agent_tool": DUMP + "\n   applied_apy: 4.25\n"
                                                     "   period: 11/01/2025 - 11/30/2025"})
    ctx = {"expected_apy": 5.5, "actual_apy": 4.25, "principal": 96000,
           "period_start": "11/01/2025", "period_end": "11/30/2025"}
    flags = _run("get_interest_correction", ctx)
    check("정상: 덤프 실재 actual_apy=4.25 유지·플래그 0",
          ctx["actual_apy"] == 4.25 and len(flags) == 0)

    # ── ④ 뒷줄 보존: 경고 헤더만 지워지고 같은 출력의 본문은 코퍼스에 남는다 ──────────
    print("[뒷줄 보존] 경고문 뒷줄(도구 산출 본문)의 값은 접지 출처로 유효")
    _install({"get_interest_correction":
              WARN_ECHO + "\nThe correction used principal 96000 and applied_apy 3.85."})
    ctx = {"expected_apy": 5.5, "actual_apy": 3.85, "principal": 96000,
           "period_start": None, "period_end": None}
    flags = _run("get_interest_correction", ctx)
    check("뒷줄: 본문 실재 3.85·96000 유지·플래그 0",
          ctx["actual_apy"] == 3.85 and ctx["principal"] == 96000 and len(flags) == 0)

    # ── ⑤ _strip_own_feedback 단위 ────────────────────────────────────────────────
    print("[strip 단위] 다중 태그·개행 없는 꼬리·비-경고 보존")
    f = sg._strip_own_feedback
    check("단위: 태그~줄끝 제거·뒷줄 보존",
          f("[GROUNDING WARNING] x=5.25 dropped.\nbody 96000") == "\nbody 96000")
    check("단위: 개행 없는 꼬리 전체 제거",
          f("head\n[grounding warning] x=5.25") == "head\n")
    check("단위: 다중 태그 각각 제거",
          f("[GROUNDING WARNING] a=1.\nkeep1\n[GROUNDING WARNING] b=2.\nkeep2")
          == "\nkeep1\n\nkeep2")
    check("단위: 비-경고 텍스트 불변", f("no tag here 5.25") == "no tag here 5.25")
    check("단위: 빈 입력 안전", f(None) == "" and f("") == "")

    print("\n== 결과: %d PASS / %d FAIL ==" % (len(PASS), len(FAIL)))
    if FAIL:
        print("FAILED:")
        for x in FAIL:
            print("  - " + x)
        sys.exit(1)
    print("ALL PASS — 경고문 속 값은 접지 출처로 불인정(094 재현 닫힘)·정상값 회귀 0.")


if __name__ == "__main__":
    main()
