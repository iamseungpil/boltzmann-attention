"""N2b — 모르는 불리언·범주를 값으로 채우지 말고 물어라.

001 축자: 에이전트가 `[true/false]` 플레이스홀더로 묻고, 손님이 답하지 않자 다음 턴에
`rho_bank_subscription: false`로 **스스로 채웠다.** gold는 `true`. 007도 같다 —
`invited:"false"` · `premium_subscriber:"false"` · `needs_purchase_protection:"false"` 셋 다
대화에 근거가 없다. 전수 194 sim: 이 계열 불리언 인자 **136건 중 106건이 false/None**이고
**35건은 대화에 주제조차 없다**(`true`는 30건).

★술어는 **권위 소재**로 닫는다([[52]]·설계서 §2.2) — "근거가 대화에 있나"를 의미로 판정하면
그것은 열린 술어이고 N3를 기각한 사유에 걸린다:

    인자명이 회수된 레코드의 **필드로 실재** → 그 레코드가 권위자 → 통과
    실재하지 않음                          → 권위자는 손님뿐 → 손님이 그 값을 **말했을 때만** 통과

전수 실측이 이 분기가 실제로 가른다는 것을 보인다: `invited`는 레코드 필드로 3회 실재/26회 부재,
`premium_subscriber`·`needs_purchase_protection`·`business`는 **0회 실재**다.

자료형은 **env 도구 스키마에서 기계도출**한다([[23]] opex 0) — A2에 자료형을 적지 않는다.

★[[05]] 경계
  · 값을 **만들지 않는다.** 엔진은 "이 값을 네가 알 수 없다"고만 말하고 무엇을 물을지는 모델이 정한다.
  · 폐기된 폴백: *"예산 소진이면 미기입"* — 픽스처가 무효화했다(설계서 §2.4b).
    `apply_for_credit_card(rho_bank_subscription: bool = False)`이고 우리 `catalog_filter`도
    `not ctx.get("invited")`라 **빈 값이 false로 되돌아간다.** 그래서 폴백은 미기입이 아니라 **보류**다.
"""

import json
import os
import re

TRUTHY = ("true", "yes", "y", "1")
FALSY = ("false", "no", "n", "0")


def enabled():
    return os.environ.get("T2_ASK_UNKNOWN_BOOL") == "1"


def _schema_of(orch, tool_name):
    """env 도구 스키마의 properties. 못 찾으면 빈 dict = 무발화."""
    env = getattr(orch, "environment", None) or getattr(
        getattr(orch, "_t2_orch", None), "environment", None)
    for holder in (getattr(env, "tools", None), getattr(env, "user_tools", None)):
        for t in (holder if isinstance(holder, (list, tuple)) else [holder]):
            if t is None:
                continue
            try:
                sch = t.openai_schema
            except Exception:
                continue
            fn = (sch or {}).get("function") or {}
            if fn.get("name") != tool_name:
                continue
            return ((fn.get("parameters") or {}).get("properties") or {})
    return {}


def _closed_type(prop):
    """불리언이거나 열거형이면 True — 값이 자유롭지 않은 인자만 대상이다."""
    if not isinstance(prop, dict):
        return False
    return prop.get("type") == "boolean" or bool(prop.get("enum"))


def _is_record_field(name, messages):
    """인자명이 회수된 레코드의 **필드**로 나타나는가(값이 아니라 키로)."""
    pats = ('"%s"' % name, "'%s'" % name, "%s:" % name)
    for m in messages or []:
        if getattr(m, "role", None) != "tool":
            continue
        c = getattr(m, "content", None)
        if not isinstance(c, str):
            continue
        if any(p in c for p in pats):
            return True
    return False


def _customer_stated(name, value, messages):
    """손님이 그 값을 말했는가 — 주제어(인자명 토큰)와 값이 **같은 발화**에 있을 때만."""
    toks = [t for t in re.split(r"[_\s]+", str(name)) if len(t) > 2]
    val = str(value).strip().lower()
    yes = val in TRUTHY
    no = val in FALSY
    for m in messages or []:
        if getattr(m, "role", None) != "user":
            continue
        c = getattr(m, "content", None)
        if not isinstance(c, str):
            continue
        low = c.lower()
        if not any(t in low for t in toks):
            continue
        if yes and not re.search(r"\b(not|no|don'?t|never)\b", low):
            return True
        if no and re.search(r"\b(not|no|don'?t|never)\b", low):
            return True
        if not yes and not no and val and val in low:      # enum 값 축자
            return True
    return False


def unknown_args(orch, tool_name, args, messages):
    """값이 실려 있으나 **알 수 없는** 닫힌-자료형 인자들. 없으면 빈 리스트."""
    if not enabled():
        return []
    props = _schema_of(orch, tool_name)
    if not props:
        return []
    out = []
    for k, v in (args or {}).items():
        if v is None or not _closed_type(props.get(k)):
            continue
        if _is_record_field(k, messages):        # 레코드가 권위자 → 통과
            continue
        if _customer_stated(k, v, messages):     # 손님이 말했다 → 통과
            continue
        out.append((k, str(v)))
    return out


def feedback(tool_name, pairs):
    """엔진이 낼 문구. 값을 제안하지 않는다 — 무엇을 물을지는 모델이 정한다."""
    items = ", ".join("%s=%s" % (k, v) for k, v in pairs)
    return ("Error: [UNKNOWN-VALUE] you supplied %s to %s, but that fact is not in any record "
            "you retrieved and the customer never stated it - you are asserting something you "
            "do not know. Only the customer can answer this. Ask them for it and call this tool "
            "again with their answer. Do NOT leave the argument out either: the tool substitutes "
            "a default, which is the same as asserting a value." % (items, tool_name))


def selftest():
    class M:
        def __init__(self, role, content):
            self.role, self.content = role, content

    class T:
        def __init__(self, name, props):
            self._n, self._p = name, props

        @property
        def openai_schema(self):
            return {"function": {"name": self._n, "parameters": {"properties": self._p}}}

    class Env:
        def __init__(self, tools):
            self.tools, self.user_tools = tools, None

    class O:
        def __init__(self, tools):
            self.environment = Env(tools)

    props = {"rho_bank_subscription": {"type": "boolean"},
             "invited": {"type": "boolean"},
             "customer_name": {"type": "string"},
             "tier": {"type": "string", "enum": ["gold", "silver"]}}
    orch = O([T("apply_for_credit_card", props)])
    os.environ["T2_ASK_UNKNOWN_BOOL"] = "1"

    msgs = [M("user", "I want a card"), M("tool", '{"user_id": "a1", "name": "Sarah"}')]
    got = unknown_args(orch, "apply_for_credit_card", {"rho_bank_subscription": False}, msgs)
    assert [k for k, _ in got] == ["rho_bank_subscription"], got
    print("  ok   레코드에도 발화에도 없는 불리언 = 미지")

    rec = msgs + [M("tool", '{"rho_bank_subscription": true, "user_id": "a1"}')]
    assert unknown_args(orch, "apply_for_credit_card", {"rho_bank_subscription": False}, rec) == []
    print("  ok   레코드 필드로 실재하면 통과(권위자=레코드)")

    said = [M("user", "yes, I have a rho bank subscription")]
    assert unknown_args(orch, "apply_for_credit_card", {"rho_bank_subscription": True}, said) == []
    print("  ok   손님이 말했으면 통과(권위자=손님)")

    saidno = [M("user", "no, I don't have a subscription")]
    assert unknown_args(orch, "apply_for_credit_card", {"rho_bank_subscription": False}, saidno) == []
    print("  ok   부정 발화도 근거")

    assert unknown_args(orch, "apply_for_credit_card", {"customer_name": "Sarah"}, msgs) == []
    print("  ok   자유 문자열은 대상 아님(닫힌 자료형만)")

    got = unknown_args(orch, "apply_for_credit_card", {"tier": "gold"}, msgs)
    assert [k for k, _ in got] == ["tier"], got
    print("  ok   enum도 대상")

    assert unknown_args(orch, "no_such_tool", {"invited": True}, msgs) == []
    print("  ok   스키마 없으면 무발화")

    os.environ["T2_ASK_UNKNOWN_BOOL"] = "0"
    assert unknown_args(orch, "apply_for_credit_card", {"invited": False}, msgs) == []
    os.environ["T2_ASK_UNKNOWN_BOOL"] = "1"
    print("  ok   플래그 OFF면 무발화")
    print("PASS (8/8)")


if __name__ == "__main__":
    selftest()
