# -*- coding: utf-8 -*-
"""provenance 가 **discoverable 래퍼 안쪽**까지 보는지 검정 (2026-08-15·085).

왜: banking 의 write 는 거의 전부
`call_discoverable_agent_tool({"agent_tool_name": …, "arguments": "<JSON 문자열>"})` 다.
최상위 키만 훑던 판에서는 안쪽 `transaction_id`·`card_id` 가 **한 번도 검사되지 않았고**,
085 는 `transaction_id='tx111111'`·`card_id='card123456'` 을 그대로 냈다.

여기서 못 박는 것:
  ⒜ 안쪽 날조 인자를 **잡는다**
  ⒝ 안쪽 값이 문맥에 **있으면 통과**한다(over-block 0)
  ⒞ 래퍼가 아닌 평범한 호출의 거동은 **그대로**다(회귀 0)
  ⒟ 게이트는 여전히 `T2_PROVENANCE` 로만 열린다(이 수리의 즉시 거동 변화 = 0)

실행: seka python test_provenance_nested.py
"""
import io
import json
import os
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_gate_patch as G                                          # noqa: E402

FAIL = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


class TC(object):
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


def wrapped(inner):
    return TC("call_discoverable_agent_tool",
              {"agent_tool_name": "file_debit_card_transaction_dispute_6281",
               "arguments": json.dumps(inner)})


def main():
    # 085 실물 값. 문맥에는 unlock 출력(도구 이름)만 있고 거래 id 는 없다 — 실제 궤적과 같다.
    ctx = ("tool unlocked: file_debit_card_transaction_dispute_6281 "
           "user f7d3a82c91 account chk_b4d92f7c28").lower()

    bad = wrapped({"transaction_id": "tx111111", "card_id": "card123456",
                   "account_id": "chk_b4d92f7c28", "user_id": "f7d3a82c91"})
    r = G._provenance_deny(bad, ctx)
    check(r is not None, "안쪽 날조 인자를 잡는다")
    if r:
        check("tx111111" in r[1] or "card123456" in r[1],
              "거절문이 **어느 값**인지 말한다([[64]]): %s" % r[1][:70])

    good = wrapped({"transaction_id": "chk_b4d92f7c28", "user_id": "f7d3a82c91"})
    check(G._provenance_deny(good, ctx) is None, "문맥에 있는 값은 통과한다(over-block 0)")

    # ⒞ 평범한(래퍼 아님) 호출 — 종전 거동 유지
    plain_bad = TC("get_user_information_by_id", {"user_id": "zz9999zzz"})
    plain_ok = TC("get_user_information_by_id", {"user_id": "f7d3a82c91"})
    check(G._provenance_deny(plain_bad, ctx) is not None, "평범한 호출의 날조도 여전히 잡는다")
    check(G._provenance_deny(plain_ok, ctx) is None, "평범한 호출의 실값은 여전히 통과한다")

    # 파싱 불가한 중첩은 조용히 종전대로(예외 0)
    weird = TC("call_discoverable_agent_tool",
               {"agent_tool_name": "x", "arguments": "not json at all"})
    try:
        G._provenance_deny(weird, ctx)
        check(True, "파싱 불가 중첩에서 예외를 내지 않는다")
    except Exception as e:
        check(False, "파싱 불가 중첩에서 예외: %r" % (e,))

    # ⒟ 게이트 플래그가 유일한 스위치인지(소스 확인)
    src = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "t2_gate_patch.py"), encoding="utf-8").read()
    check('prov_on = os.environ.get("T2_PROVENANCE") == "1"' in src,
          "게이트는 T2_PROVENANCE 로만 열린다(이 수리의 즉시 거동 변화 0)")

    print("\n%s" % ("PASS" if not FAIL else "FAIL: " + " · ".join(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
