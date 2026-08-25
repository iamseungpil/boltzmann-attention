# -*- coding: utf-8 -*-
"""T2_WRITE_ARG_FAB 래칫 (2026-08-25) — 이름 패턴 없이, 선언과 값의 모양으로만 판정하는가.

왜 이 검정이 필요한가: 앞선 판은 인자 **이름**이 식별자처럼 생겼나를 추측했다
(`identifying_arg_types` + `_hint_hit`). 사용자 지적으로 철회했고, 대신 env 가 unlock 때
찍는 **선언**으로 갈랐다. 그 교체가 조용히 되돌아가면(이름 목록이 다시 자라면) 같은 표류가
재발하므로 여기서 고정한다.

이 검정이 지키는 것:
  ① `_declared_params` 가 env 고정 포맷에서 (타입·열거여부)를 읽고, **형식이 아니면 빈 결과**
  ② `_looks_placeholder` 가 실측된 값들에서 정확히 갈린다 — 날조는 참, gold 값은 거짓
  ③ 게이트 술어에 **이름 패턴이 없다** — 그 블록이 `_hint_hit` 을 부르지 않는다
  ④ 기본 OFF
  ⑤ 되돌려주는 문면은 새로 쓴 것이 아니라 **이미 검증된 것**(REGEN_FEEDBACK·C45)
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# env 가 실제로 찍는 블록 축자 (t7354 grpB1 task_040 msg5 에서 잘라 온 것).
SPEC = """Tool unlocked: file_credit_card_transaction_dispute_4829
Description: File a formal dispute for a credit card transaction.

Tool: file_credit_card_transaction_dispute_4829
Description: File a formal dispute for a credit card transaction.
Parameters:
  - transaction_id: string (required) - The unique identifier for the transaction being disputed
  - card_last_4_digits: string (required) - Last 4 digits of the credit card number
  - contacted_merchant: boolean (required) - Whether the user attempted to resolve the issue first
  - dispute_reason: string (required) - Reason for the dispute. Must be one of: 'duplicate_charge'
  - partial_refund_amount: number (optional) - Amount requested for partial refund
"""

# 실측된 값들 (t7354 전수). 왼쪽 = 날조로 걸려야 하는 것, 오른쪽 = 통과해야 하는 것.
FAB = ["1234", "TRXN1234567890", "TRXN1234567893"]
REAL = ["0581", "1652", "11/14/2025", "11/05/2025", "txn_25e23705f61f",
        "cc_01f21c9970_bgold", "215-555-0267"]


class _M(object):
    def __init__(self, c):
        self.content = c


def main():
    import t2_gate_patch as G

    # ① 선언 읽기 · 형식 아니면 빈 결과
    dp = G._declared_params([_M(SPEC)])
    assert dp.get("transaction_id") == ("string", []), dp.get("transaction_id")
    assert dp.get("card_last_4_digits") == ("string", []), dp.get("card_last_4_digits")
    assert dp.get("contacted_merchant") == ("boolean", []), dp.get("contacted_merchant")
    assert dp.get("dispute_reason") == ("string", ["duplicate_charge"]), dp.get("dispute_reason")
    assert dp.get("partial_refund_amount") == ("number", [])
    assert G._declared_params([_M("그냥 산문입니다. 매개변수 같은 것은 없습니다.")]) == {}, \
        "형식이 아닌데 무언가를 읽었다 — fail-open 이 깨졌다"

    # ② 값의 모양이 실측대로 갈린다
    bad = [v for v in FAB if not G._looks_placeholder(v)]
    assert not bad, "날조로 걸려야 하는데 안 걸리는 값: %r" % bad
    over = [v for v in REAL if G._looks_placeholder(v)]
    assert not over, "실값인데 자리표시자로 잡히는 값(오차단): %r" % over

    src = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
    i = src.index('os.environ.get("T2_WRITE_ARG_FAB")')
    # ⚠블록 경계는 **이 레버의 끝**까지다 — 이웃 블록의 주석까지 삼키면 검정이 무너진다
    #   (2026-08-25: T2_SPEC_ARG_FACTS 주석의 `card_action` 이 잡혔다).
    blk = src[i:src.index("T2_SPEC_ARG_FACTS", i)]

    # ③ 이름 패턴 금지
    assert "_hint_hit" not in blk, "게이트가 다시 **이름 패턴**을 쓰기 시작했다([[59]] 정신)"
    for bad_tok in ("digit", "card", "last_4"):
        assert bad_tok not in blk, "게이트에 도메인/이름 리터럴이 들어왔다: %r" % bad_tok
    # 세 술어가 전부 살아 있다
    for need in ("_declared_params", "_looks_placeholder", "_ctx_has"):
        assert need in blk, "술어 %r 가 사라졌다" % need

    # ⑤ 문면은 이미 검증된 것을 쓴다
    assert "REGEN_FEEDBACK.format" in blk, "새 문면을 지어 쓰고 있다 — 검증분을 써라(C45)"

    # ④ 기본 OFF
    gs = io.open(os.path.join(HERE, "go_stack.sh"), encoding="utf-8").read()
    assert "T2_WRITE_ARG_FAB=0" in gs, "기본 OFF 가 아니다"

    print("OK T2_WRITE_ARG_FAB: 선언 읽기 4칸 · fail-open · 날조 %d/%d · 오차단 0 · "
          "이름 패턴 0 · 기본 OFF" % (len(FAB), len(FAB)))


if __name__ == "__main__":
    main()
