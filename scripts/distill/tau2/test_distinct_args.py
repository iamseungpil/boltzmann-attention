# -*- coding: utf-8 -*-
"""★§T-10 — 선언된 인자 쌍이 **같은 값**이면 잡는다(엔진은 고르지 않는다).

근거(082 실물·x722B): `file_debit_card_transaction_dispute_6281` 4행 전부에서
`customer_max_liability_amount` 가 gold(50)와 어긋났고, **3행은 `disputed_amount` 와 값이 정확히
같았다**(347.5·89.99·100.0) — 정책 상한 자리에 거래 금액을 복사했다.
⊖ 부호표: gold 에서 두 값이 같은 경우 **0/19**(gold 는 50×17 · 500×2).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_interpreter as G
import t2_gate_patch as GP

A2 = G.load_domain_a2("banking_knowledge")
TOOL = "file_debit_card_transaction_dispute_6281"


class TC(object):
    def __init__(self, name, arguments):
        self.name, self.arguments, self.id = name, arguments, "tc_1"


def _dispatch(inner):
    return TC("call_discoverable_agent_tool",
              {"agent_tool_name": TOOL, "arguments": json.dumps(inner, ensure_ascii=False)})


def t_declaration_is_merged():
    assert (A2.get("distinct_args") or {}).get(TOOL), "A2 정본·gate 동기화 실패([[24]])"


DOCS = [{"title": "Understanding Regulation E: Your Debit Card Consumer Protections",
         "content": ("Understanding Regulation E … your liability is limited based on how quickly "
                     "you report it: - Within 2 business days: Maximum $50 liability - Within 60 "
                     "days: Maximum $500 liability")}]


def t_copy_of_sibling_fires():
    got = GP.distinct_args_violation(_dispatch(
        {"disputed_amount": 347.5, "customer_max_liability_amount": 347.5}), A2, docs=DOCS)
    assert got and got[1] == "customer_max_liability_amount" and got[2] == "disputed_amount", got
    fb = got[4].lower()
    # ★[[64]] — 반려는 **구체적으로 무엇을 하면 되는지** 말해야 한다: 무엇이 틀렸나 · 어디서 읽나 ·
    #   다음에 무엇을 하나. 세 칸이 다 있어야 한다(036 의 OPERATOR-SCOPE 는 셋째 칸이 없어
    #   에이전트가 "technical error" 로 읽고 포기했다).
    assert "not the amount in dispute" in fb, "무엇이 틀렸는지"
    assert "re-send this call" in fb, "다음에 무엇을 하는지"
    # ★사용자 지적: *"그냥 재시도하거나 반려하면 안 고쳐진다"* — 그래서 **재료를 붙여 배달**한다.
    #   (이 인자는 env 필수·기본값 없음이라 [[63]] 의 *제거*를 쓸 수 없다.)
    assert "policy (" in fb and "verbatim" in fb, "정책 문서를 축자로 실어야 한다"
    assert "$50" in got[4] and "$500" in got[4], "등급을 고를 재료가 실제로 들어와야 한다"


def t_gold_shape_never_fires():
    """gold 는 50 또는 500 이고 disputed 와 다르다 — ⊖=0 의 단위검정 대응."""
    for pair in ((347.5, 50), (89.99, 50), (100.0, 50), (523.17, 50), (47.5, 50), (600.0, 500)):
        assert GP.distinct_args_violation(_dispatch(
            {"disputed_amount": pair[0], "customer_max_liability_amount": pair[1]}), A2) is None, pair


def t_numeric_equality_not_string_equality():
    """`100` 과 `100.0` 은 같은 값이다 — 문자열 비교면 놓친다."""
    assert GP.distinct_args_violation(_dispatch(
        {"disputed_amount": "100", "customer_max_liability_amount": 100.0}), A2) is not None


def t_undeclared_tool_untouched():
    tc = TC("some_other_tool", {"a": 5, "b": 5})
    assert GP.distinct_args_violation(tc, A2) is None


def t_missing_arg_untouched():
    assert GP.distinct_args_violation(_dispatch({"disputed_amount": 5}), A2) is None


def t_no_declaration_is_noop():
    assert GP.distinct_args_violation(_dispatch(
        {"disputed_amount": 5, "customer_max_liability_amount": 5}), {}) is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("t_")]
    for f in fns:
        f()
        print("ok %s" % f.__name__)
    print("PASS %d/%d" % (len(fns), len(fns)))
