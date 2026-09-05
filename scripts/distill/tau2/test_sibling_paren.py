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


def t_case_insensitive_but_not_pattern_based():
    """대소문자는 흡수하되 **패턴은 쓰지 않는다**([[59]]). 그래서 괄호 안 공백처럼 실물에 없는
    변형은 **일부러 안 잡는다** — 흡수하려면 정규화가 필요하고 그것이 곧 패턴 매칭이다."""
    assert G.sibling_paren_arg(TC("x", {"kind": "Savings", "label": "Green Account (savings)"})) is not None
    assert G.sibling_paren_arg(TC("x", {"kind": "savings", "label": "Green Account ( savings )"})) is None


def t_gold_shape_never_fires():
    """gold 형태(괄호 없음)는 어떤 조합에서도 발화하지 않는다 — ⊖=0 의 단위검정 대응."""
    for args in ({"account_type": "savings", "account_class": "Green Account"},
                 {"account_type": "checking", "account_class": "Evergreen Account"},
                 {"reason": "annual_fee", "card_id": "cc_1"}):
        assert G.sibling_paren_arg(TC("t", args)) is None, args


# ═══════════════════════════════════════════════════════════════════════════════
# ★2026-09-05 무장(`sibling_paren_strip`) — **거동** 검정.
#   여기까지는 술어의 *반환* 만 봤다. 반환을 아무도 안 쓰면 레버는 없는 것이다([[81]]) —
#   아래 4개는 «커밋될 인자» 가 실제로 갈리는지, 그리고 **안 켜면 한 글자도 안 바뀌는지**를 본다.
#   효과 프로브 정본 = `reports/facet_rft_2026/x771_068_effect.py`(발화 103·갈림 3·해악 0).
# ═══════════════════════════════════════════════════════════════════════════════
def _inner(tc):
    """이 호출이 env 로 나갈 때의 **안쪽** 인자(= DB 에 반영될 것)."""
    ar = tc.arguments if isinstance(tc.arguments, dict) else {}
    sub = ar.get("arguments")
    if isinstance(sub, str):
        return json.loads(sub)
    return sub if isinstance(sub, dict) else ar


def t_strip_changes_the_committed_arg():
    """065 실물 모양(디스패처 경유) — 커밋값이 gold 문자열이 된다."""
    inner = {"user_id": "rp65a7b3c4", "account_type": "savings",
             "account_class": "Green Account (savings)"}
    tc = TC("call_discoverable_agent_tool",
            {"agent_tool_name": "open_bank_account_4821",
             "arguments": json.dumps(inner, ensure_ascii=False)})
    out = G.sibling_paren_strip([tc])
    bag = _inner(tc)
    assert out and out[0][1] == "account_class", out
    assert bag["account_class"] == "Green Account", bag
    assert bag["account_type"] == "savings" and bag["user_id"] == "rp65a7b3c4", \
        "형제 인자·대상 id 는 한 글자도 안 건드린다"


def t_strip_direct_call_shape():
    tc = TC("open_bank_account_4821",
            {"account_type": "checking", "account_class": "Green Account (checking)"})
    G.sibling_paren_strip([tc])
    assert tc.arguments["account_class"] == "Green Account", tc.arguments


def t_strip_only_subtracts_never_invents():
    """★[[63]]/[[62]] — 새 문자열을 만들지 않는다. 결과의 모든 문자는 원본에서 온 것이고,
    빠진 것은 술어가 지목한 부분문자열뿐이다(공백 접기 제외)."""
    tc = TC("t", {"kind": "savings", "label": "Green Account (savings) Plus"})
    out = G.sibling_paren_strip([tc])
    old, new = out[0][2], out[0][3]
    assert new == "Green Account Plus", new
    assert " ".join(old.replace("(savings)", "").split()) == new, (old, new)


def t_no_sibling_no_change():
    """술어가 발화하지 않으면 거동 0 — 무차별 괄호 제거가 아니다."""
    tc = TC("open_bank_account_4821",
            {"account_type": "checking", "account_class": "Green Account (savings)"})
    before = json.dumps(tc.arguments, sort_keys=True, ensure_ascii=False)
    assert G.sibling_paren_strip([tc]) == []
    assert json.dumps(tc.arguments, sort_keys=True, ensure_ascii=False) == before


def t_gold_shape_survives_strip():
    """[[57]] 부정통제 — gold 형태는 STRIP 을 통과해도 **바이트 동일**이어야 한다."""
    for args in ({"account_type": "savings", "account_class": "Green Account"},
                 {"account_type": "checking", "account_class": "Evergreen Account"},
                 {"account_id": "chk_rp65a7b3c4"}):
        tc = TC("open_bank_account_4821", dict(args))
        assert G.sibling_paren_strip([tc]) == [], args
        assert tc.arguments == args, (tc.arguments, args)


def t_switch_off_changes_nothing():
    """★[[57]] 스위치 부정통제 — `unset`/`log`/`deny` 에서는 **한 글자도** 안 바뀐다.
    엔진 호출부(`unified()`)의 분기 술어와 **같은 조건식**을 여기서 실행한다: 무장은 `strip`
    에서만 일어난다. (호출부가 그 술어를 실제로 부르는지는 `test_lever_wiring.py` 가 본다.)"""
    for mode in (None, "log", "deny", "0", "1", "strip"):
        inner = {"account_type": "savings", "account_class": "Green Account (savings)"}
        tc = TC("call_discoverable_agent_tool",
                {"agent_tool_name": "open_bank_account_4821",
                 "arguments": json.dumps(inner, ensure_ascii=False)})
        if mode is None:
            os.environ.pop("T2_SIBLING_PAREN", None)
        else:
            os.environ["T2_SIBLING_PAREN"] = mode
        if os.environ.get("T2_SIBLING_PAREN") == "strip":      # 호출부와 같은 조건
            G.sibling_paren_strip([tc])
        got = _inner(tc)["account_class"]
        want = "Green Account" if mode == "strip" else "Green Account (savings)"
        assert got == want, (mode, got)
    os.environ.pop("T2_SIBLING_PAREN", None)


def t_deny_is_not_wired():
    """⛔`deny` 를 반려로 승격하지 않았음을 **거동**으로 못박는다(W-5 재발화 루프 회피).
    반려 경로가 생기면 이 검정이 깨진다 — 그때는 W-5 비용을 다시 재고 결정하라."""
    os.environ["T2_SIBLING_PAREN"] = "deny"
    try:
        tc = TC("open_bank_account_4821",
                {"account_type": "savings", "account_class": "Green Account (savings)"})
        before = dict(tc.arguments)
        if os.environ.get("T2_SIBLING_PAREN") == "strip":
            G.sibling_paren_strip([tc])
        assert tc.arguments == before
    finally:
        os.environ.pop("T2_SIBLING_PAREN", None)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("t_")]
    for f in fns:
        f()
        print("ok %s" % f.__name__)
    print("PASS %d/%d" % (len(fns), len(fns)))
