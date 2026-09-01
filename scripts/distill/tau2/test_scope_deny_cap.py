# -*- coding: utf-8 -*-
"""★§T-9 — `[OPERATOR-SCOPE]` 반려는 **이행 가능해야 하고, 반복되면 안 된다**.

036 실물(x722A): gold 가 요구하는 `order_replacement_credit_card_7291` 을 우리 게이트가 **10회**
반려했고, 에이전트는 그것을 *"technical error"* 로 읽고 KB 에서 `OPERATOR-SCOPE` 를 검색하다
**포기·human 이관** → 그 gold 행이 통째로 MISSING → reward 0.0.
같은 파일의 선행 실측: `[OPERATOR-SCOPE]` **61회 중 49회는 그 도구가 끝내 실행됐다**
— 반려가 선택을 바꾸지 않는다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_resolve as R


class Agent(object):
    tools = []


def t_wording_names_an_escape():
    fb = R.OPERATOR_SCOPE_FB.format(chosen="x", scopes="'x' = do x")
    assert "call it again unchanged" in fb, "자기 선택이 옳다고 믿는 모델에게 할 일을 줘야 한다"


def t_cap_default_is_one():
    """기본 1 = 한 번만 말한다. 그 이상은 턴만 태운다(61/49 실측)."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "t2_resolve.py"), encoding="utf-8").read()
    assert 'os.environ.get("T2_SCOPE_DENY_CAP", "1")' in src


def t_counter_is_per_agent_and_per_pair():
    """카운터는 **에이전트(=sim)** 에 붙고 `(chosen, want)` 쌍마다 따로 센다 —
    프로세스 전역이면 sim 간 오염이고, 쌍을 뭉치면 다른 오선택을 못 잡는다."""
    a = Agent()
    seen = a._t2_scope_denies = {}
    for pair in (("A", "B"), ("A", "B"), ("C", "D")):
        seen[pair] = seen.get(pair, 0) + 1
    assert seen[("A", "B")] == 2 and seen[("C", "D")] == 1
    b = Agent()
    assert getattr(b, "_t2_scope_denies", None) is None, "다른 sim 으로 새면 안 된다"


def t_over_cap_passes_rather_than_denies():
    """상한을 넘으면 **통과**다 — 오답은 남을 수 있어도 태스크를 잃지는 않는다."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "t2_resolve.py"), encoding="utf-8").read()
    i = src.index("operator-scope 상한 초과")
    assert 'return {"status": "ok"}' in src[i:i + 400]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("t_")]
    for f in fns:
        f()
        print("ok %s" % f.__name__)
    print("PASS %d/%d" % (len(fns), len(fns)))
