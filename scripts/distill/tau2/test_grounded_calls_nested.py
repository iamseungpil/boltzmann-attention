# -*- coding: utf-8 -*-
"""중첩 계약(`{"tool", "arguments": {...}}`)의 근거검산 — **계약과 검산기가 같은 모양을 봐야 한다**.

사건(2026-08-14 야간·072 실물): x311 이 write-착수 계약을 도메인-일반형
`{"tool": ..., "arguments": {...}}` 로 일반화했는데, 검산부 `grounded_calls` 는 평면 구조만 알아
`c.items()` 가 **`arguments` dict 하나**를 값으로 내놨다. dict 의 문자열 형태가 코퍼스에 있을 리
없으므로 **우리 형식을 지킨 제안은 100% 기각**됐다:

    [T2_WRITE_SUB] 제안 1건 → 근거검산 통과 0건     ← 072 에서 8회 반복(t7290·t7291 양쪽)

073 이 통과했던 자리는 모델이 우리 형식을 **어기고** 평면으로 답한 경우였다. 즉 레버가 사는지
죽는지가 **모델이 우리 지시를 따르느냐 아니냐에 반비례**하고 있었다.

이 검정이 고정하는 것: 두 모양 다 통과 · 날조는 두 모양 다 기각 · 빈 인자는 통과 아님.
"""
import io
import os
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_subcall as SC                                            # noqa: E402

FAIL = []
NAMES = {"apply_checking_account_credit_5829"}
CORPUS = ["btxn_6a3453e0afd9 charged $3.50, documented fee $0.00, difference $3.50",
          "Transactions for account chk_538bfb9cba"]


def chk(c, m):
    if not c:
        FAIL.append(m)
    print("  %s %s" % ("ok  " if c else "FAIL", m))


def call(nested, amount=3.50, acct="chk_538bfb9cba", tool=None):
    t = tool or sorted(NAMES)[0]
    args = {"account_id": acct, "amount": amount}
    return {"tool": t, "arguments": args} if nested else dict({"tool": t}, **args)


def n(c):
    return len(SC.grounded_calls([c], CORPUS, NAMES))


def main():
    print("[계약 두 모양이 같게 판정되나]")
    chk(n(call(False)) == 1, "평면 계약 통과")
    chk(n(call(True)) == 1, "중첩 계약 통과 ← 이 자리가 072 에서 죽어 있었다")

    print("[날조는 두 모양 다 기각]")
    chk(n(call(False, amount=99.99)) == 0, "평면·근거 없는 금액 기각")
    chk(n(call(True, amount=99.99)) == 0, "중첩·근거 없는 금액 기각")
    chk(n(call(True, acct="chk_nonexistent")) == 0, "중첩·근거 없는 계좌 기각")

    print("[형식-불문 수치 매칭 유지(9.50↔9.5 계보)]")
    chk(n(call(True, amount="$3.50")) == 1, "통화 기호·문자열 형태도 통과")
    chk(n(call(True, amount=3.5)) == 1, "3.5 ≡ 3.50")

    print("[경계]")
    chk(len(SC.grounded_calls([{"tool": sorted(NAMES)[0], "arguments": {}}],
                              CORPUS, NAMES)) == 0, "빈 인자는 통과 아님")
    chk(len(SC.grounded_calls([call(True, tool="not_a_real_tool_0000")],
                              CORPUS, NAMES)) == 0, "레지스트리 밖 이름 기각")
    chk(len(SC.grounded_calls([], CORPUS, NAMES)) == 0, "빈 목록")

    print("\n%s (%d fail)" % ("FAIL" if FAIL else "PASS", len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
