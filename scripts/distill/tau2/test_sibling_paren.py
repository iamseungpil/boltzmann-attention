# -*- coding: utf-8 -*-
"""★§T-8 형제-괄호 술어 — **거동** 검정(소스 문자열 검사 아님·§S-1 교훈).

근거: 065 는 gold 4행 중 3행이 맞고 `account_class` 한 칸만 어긋났다 —
`"Green Account (savings)"` ↔ gold `"Green Account"`, 그런데 같은 호출에 `account_type="savings"`
가 이미 있다. env 는 이 인자를 검증하지 않는다.
⊖ 부호표: banking gold 1,976 · 다도메인 영속 gold 3,577 문자열 인자 **둘 다 0건**.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_gate_patch as G


class TC(object):
    def __init__(self, name, arguments):
        self.name, self.arguments, self.id = name, arguments, "tc_1"


def t_direct_call_fires():
    tc = TC("open_bank_account_4821",
            {"user_id": "u1", "account_type": "savings", "account_class": "Green Account (savings)"})
    got = G.sibling_paren_arg(tc)
    assert got and got[1] == "account_class" and got[3] == "(savings)", got


def t_dispatcher_call_fires():
    """★W-2 — 실물은 디스패처 경유다. 언랩 없으면 이 술어는 라이브에서 죽는다."""
    inner = {"user_id": "rp65a7b3c4", "account_type": "savings",
             "account_class": "Green Account (savings)"}
    tc = TC("call_discoverable_agent_tool",
            {"agent_tool_name": "open_bank_account_4821",
             "arguments": json.dumps(inner, ensure_ascii=False)})
    got = G.sibling_paren_arg(tc)
    assert got, "디스패처 경유가 안 잡히면 이 레버는 무의미하다"
    assert got[0] == "open_bank_account_4821" and got[1] == "account_class"


def t_paren_not_matching_a_sibling_passes():
    tc = TC("open_bank_account_4821",
            {"account_type": "checking", "account_class": "Green Account (savings)"})
    assert G.sibling_paren_arg(tc) is None, "형제 값이 아니면 건드리지 않는다"


def t_no_paren_passes():
    tc = TC("open_bank_account_4821",
            {"account_type": "savings", "account_class": "Green Account"})
    assert G.sibling_paren_arg(tc) is None


def t_empty_paren_passes():
    tc = TC("x", {"a": "", "b": "name ()"})
    assert G.sibling_paren_arg(tc) is None


def t_case_and_space_insensitive():
    tc = TC("x", {"kind": "Savings", "label": "Green Account ( savings )"})
    assert G.sibling_paren_arg(tc) is not None


def t_gold_shape_never_fires():
    """gold 형태(괄호 없음)는 어떤 조합에서도 발화하지 않는다 — ⊖=0 의 단위검정 대응."""
    for args in ({"account_type": "savings", "account_class": "Green Account"},
                 {"account_type": "checking", "account_class": "Evergreen Account"},
                 {"reason": "annual_fee", "card_id": "cc_1"}):
        assert G.sibling_paren_arg(TC("t", args)) is None, args


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("t_")]
    for f in fns:
        f()
        print("ok %s" % f.__name__)
    print("PASS %d/%d" % (len(fns), len(fns)))
