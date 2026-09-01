# -*- coding: utf-8 -*-
"""★§T-15 — 같은 값이 **다른 그룹**에 들어오면 같은 혜택을 두 번 센 것이다.

094·095 동일 기전(실물): `card 0.025`("Gold Rewards Card | +0.025%")와
`relationship 0.025`("you receive a 0.025% relationship bonus…")를 함께 보낸다.
`card` 는 `max1` 이라 0.025 가 0.6 에 밀려 버려지고, 같은 혜택이 `relationship`(sum)으로
다시 들어와 **6.85 → 6.875** ⇒ 094 `amount 140→148` · 095 `98→100`.

⊖ 부호표(영속 전 런 전수): 교차그룹 동일값 **있음 → 정답 0 · 오답 15** / 없음 → 정답 18 · 오답 154.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_interpreter as G
import t2_gate_patch as GP

A2 = G.load_domain_a2("banking_knowledge")
TOOL = "get_correct_savings_apy"


class TC(object):
    def __init__(self, name, arguments):
        self.name, self.arguments, self.id = name, arguments, "tc_1"


def _call(components):
    return TC(TOOL, {"components": json.dumps(components, ensure_ascii=False)})


REAL_095 = [{"kind": "base", "value": 5.5}, {"kind": "checking", "value": 0.75},
            {"kind": "checking", "value": 0.1}, {"kind": "card", "value": 0.15},
            {"kind": "card", "value": 0.025}, {"kind": "card", "value": 0.6},
            {"kind": "relationship", "value": 0.025}]


def t_real_095_shape_fires():
    got = GP.group_dup_value(_call(REAL_095), A2)
    assert got and abs(got[1] - 0.025) < 1e-9 and got[2] == ["card", "relationship"], got


def t_same_value_within_one_group_is_fine():
    """한 그룹 안의 같은 값은 중복계상이 아니다(같은 종류를 여러 건 보유할 수 있다)."""
    assert GP.group_dup_value(_call(
        [{"kind": "card", "value": 0.6}, {"kind": "card", "value": 0.6}]), A2) is None


def t_correct_shape_does_not_fire():
    """gold 가 요구하는 6.85 구성(base 5.5 + checking 0.75 + card 0.6)은 무발화."""
    assert GP.group_dup_value(_call(
        [{"kind": "base", "value": 5.5}, {"kind": "checking", "value": 0.75},
         {"kind": "card", "value": 0.6}]), A2) is None


def t_undeclared_tool_untouched():
    assert GP.group_dup_value(TC("some_tool", {"components": "[]"}), A2) is None


def t_wording_states_a_fact_and_does_not_presume_error():
    """⛔이 규칙의 출처는 **gold 이지 정책이 아니다**(사용자 지적·리뷰 M). 서로 다른 혜택이
    우연히 같은 값일 수 있으므로 **오류로 단정하면 gold-fit** 이다([[23]]). 문면은 사실만 말한다."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "t2_gate_patch.py"), encoding="utf-8").read()
    i = src.index("[T2_GROUP_DUP]")
    seg = src[i:i + 420]
    assert "사실 진술" in seg and "판단은" in seg
    assert "pop(" not in seg, "엔진이 행을 지우면 안 된다"
    # 근거 한계가 코드에 남아 있어야 한다
    assert "정책에는" in src[src.index("def group_dup_value"):src.index("def distinct_args_violation")]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("t_")]
    for f in fns:
        f()
        print("ok %s" % f.__name__)
    print("PASS %d/%d" % (len(fns), len(fns)))
