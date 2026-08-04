"""N1 — 손님에게 **문장으로** 불러주는 값도 같은 접지 검사를 태운다.

`requestor=user` gold는 손님이 직접 실행해야 채점된다. 그런데 `give_discoverable_user_tool`은
전수 194 sim에서 **80회 중 75회가 도구명만** 실어 보내고, 값은 본문에 실린다 — 손님이 실행한 인자값
157개 중 **142개(90%)가 에이전트 산문에 축자로 존재**한다. A2 `write_arg_grounding`의 give-측 규칙은
**호출 인자**를 보므로 인자가 없으면 검사할 것이 없다:

    010  `user_id`: "Your user ID."            ← 플레이스홀더가 그대로 손님에게 간다
    029  `user_id: 890389b165` · `transaction_id: txn_e647e242ce96`
    015  "referral link for the **Crypto-Cash Back Card**"  ← gold는 Platinum

015가 정본 사례다: A2에는 이미 P10 규칙이 있고(`card_name`을 `corpus_roles:["tool"]`로 못박음·정책
문서 축자 인용) **한 번도 발화하지 않았다.** 규칙이 없어서가 아니라 **닿지 않아서**다.

★이 모듈은 **후보만 만든다.** 판정은 기존 `_write_arg_ground_deny`가 그대로 한다 —
그래서 A2 저작 순증 0이고, 새 판정 로직도 없다.

★[[16]] P2-b 금지선: 폐기된 것은 *모델 산문의 의미 해석*이다. 여기서 보는 것은
**A2가 선언한 키 이름이 콜론과 함께 박힌 자리**뿐이고, 못 뽑으면 아무 일도 없다.
그리고 **애매해도 아무 일도 없다**(아래 보수 규칙) — 오추출로 정당한 안내를 막는 것이
이 레버의 유일한 부작용이라 구조적으로 유계화한다.
"""

import os
import re

MAX_TOKENS = 6
# `key: value` / `key = value` — 백틱·별표는 마크다운 강조라 무시한다.
_PAIR = re.compile(r"[`*_]{0,2}([a-z][a-z0-9_]{2,40})[`*_]{0,2}\s*[:=]\s*([^\n]{1,120})")
# 값은 첫 구분자까지만: 마침표·쉼표·세미콜론·여는 괄호·백틱·별표·줄바꿈
_STOP = re.compile(r"[.,;(\[`*]|\s-\s")


def enabled():
    return os.environ.get("T2_HANDOFF_ARG_GROUND") == "1"


def _clean(raw):
    """값 후보를 보수적으로 자른다. 애매하면 None(=포기)."""
    v = _STOP.split(raw, 1)[0]
    v = v.strip().strip('"').strip("'").strip()
    if not v:
        return None
    if len(v.split()) > MAX_TOKENS:      # 절 전체가 잡힌 것 — 포기
        return None
    return v


def extract(text, keys):
    """본문에서 (선언된 키 → 값). 같은 키가 서로 다른 값으로 두 번 나오면 그 키는 버린다."""
    out, seen = {}, {}
    if not isinstance(text, str) or not keys:
        return out
    for k, raw in _PAIR.findall(text):
        if k not in keys:
            continue
        v = _clean(raw)
        if v is None:
            seen[k] = None                       # 애매 = 그 키 전체 포기
            continue
        if k in seen and seen[k] not in (None, v):
            seen[k] = None
            continue
        seen.setdefault(k, v)
    return {k: v for k, v in seen.items() if v}


def declared_keys(specs, applies_to):
    keys = set()
    for sp in specs or []:
        if sp.get("applies_to") != applies_to:
            continue
        keys |= {str(a) for a in (sp.get("grounded_args") or [])}
    return keys


class _Shim:
    """`_write_arg_ground_deny`가 기대하는 최소 모양 — 추출한 값을 그 함수에 그대로 넘긴다."""

    def __init__(self, name, args):
        self.name = name
        self.arguments = dict(args)


def check(deny_fn, messages, text, specs, applies_to):
    """산문에 실린 선언 인자의 접지 판정. 통과·비대상이면 None(=거동 불변)."""
    if not enabled():
        return None
    keys = declared_keys(specs, applies_to)
    if not keys:
        return None
    found = extract(text, keys)
    if not found:
        return None
    try:
        return deny_fn(messages, _Shim(applies_to, found), specs)
    except Exception:
        return None


def selftest():
    class M:
        def __init__(self, role, content):
            self.role, self.content = role, content

    specs = [{"applies_to": "give_discoverable_user_tool",
              "grounded_args": ["user_id", "transaction_id", "card_name"],
              "feedback": "Error: [WRITE-GROUNDING] '{val}' for {arg} is not in the records."}]
    keys = declared_keys(specs, "give_discoverable_user_tool")
    os.environ["T2_HANDOFF_ARG_GROUND"] = "1"

    assert extract("- `user_id`: 890389b165\n- `transaction_id`: txn_e647", keys) == \
        {"user_id": "890389b165", "transaction_id": "txn_e647"}
    print("  ok   선언 키만 뽑는다")

    assert extract("`card_name`: Silver Rewards Card (다만 Platinum을 권합니다)", keys) == \
        {"card_name": "Silver Rewards Card"}
    print("  ok   값은 구분자까지 — 괄호 이후는 값이 아니다")

    assert extract("`card_name`: the one that seems best for your everyday spending needs", keys) == {}
    print("  ok   토큰 상한 초과 = 포기(오추출 방지)")

    assert extract("`user_id`: 111\n`user_id`: 222", keys) == {}
    print("  ok   같은 키 다른 값 = 포기")

    assert extract("`annual_income`: 100000", keys) == {}
    print("  ok   미선언 키는 무시")

    def deny(messages, tc, sp):
        blob = "\n".join(m.content for m in messages if m.role == "tool")
        for k, v in tc.arguments.items():
            if v not in blob:
                return sp[0]["feedback"].replace("{arg}", k).replace("{val}", v)
        return None

    msgs = [M("tool", "records: user_id 890389b165")]
    assert check(deny, msgs, "`user_id`: 890389b165", specs, "give_discoverable_user_tool") is None
    print("  ok   원장에 있으면 통과")

    r = check(deny, msgs, "`user_id`: Your user ID", specs, "give_discoverable_user_tool")
    assert r and "Your user ID" in r, r
    print("  ok   플레이스홀더는 걸린다 (010 실사례)")

    os.environ["T2_HANDOFF_ARG_GROUND"] = "0"
    assert check(deny, msgs, "`user_id`: Your user ID", specs, "give_discoverable_user_tool") is None
    os.environ["T2_HANDOFF_ARG_GROUND"] = "1"
    print("  ok   플래그 OFF면 무발화")
    print("PASS (8/8)")


if __name__ == "__main__":
    selftest()
