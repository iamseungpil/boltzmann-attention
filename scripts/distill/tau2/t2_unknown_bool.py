"""N2b — 모르는 불리언·범주를 값으로 채우지 말고 물어라.

001 축자: 에이전트가 `[true/false]` 플레이스홀더로 묻고, 손님이 답하지 않자 다음 턴에
`rho_bank_subscription: false`로 **스스로 채웠다.** gold는 `true`. 007도 같다 —
`invited:"false"` · `premium_subscriber:"false"` · `needs_purchase_protection:"false"` 셋 다
대화에 근거가 없다. 전수 194 sim: 이 계열 불리언 인자 **136건 중 106건이 false/None**이고
**35건은 대화에 주제조차 없다**(`true`는 30건).

★술어는 **권위 소재**로 닫는다([[52]]·설계서 §2.2) — "근거가 대화에 있나"를 의미로 판정하면
그것은 열린 술어이고 N3를 기각한 사유에 걸린다. 정확히 말하면 이 술어가 재는 것은
**여기서 CWA가 허가되는가**다 — 부재를 거짓으로 접는 추론은 외연을 닫아 준 권위자가 있을 때만 건전하다:

    인자명이 회수된 레코드의 **필드로 실재** → 레코드가 외연을 닫는다 → CWA 허가 → 통과
    실재하지 않음                          → 닫을 권위자가 손님뿐 → 손님이 **말했을 때만** 통과

⚠**표면을 가려야 한다**(`x76_cwa_licence_surface.py` 전수): tool 메시지에 이름이 *보이기만* 하면
통과시키면 N97에서 공급된 닫힌-자료형 인자 **317건 중 181건**을 허가하는데, **레코드가 실제로
닫아 준 것은 4건**뿐이다(값이 실린 오-허가 176건). N97B는 255 대 **106**이다. 허가의 출처는
KB 정책 산문과 env TypeError 문자열이었다 — 둘 다 개념을 **언급**할 뿐 종족을 **열거**하지 않는다.
`business`는 N97에서 언급 939회에 레코드 필드 **0**, `invited`도 **0**,
`premium_subscriber`·`rho_bank_subscription`은 언급조차 각각 1·0회다.

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
    """인자명이 회수된 **레코드의 필드**인가 — **키 자리**에서만 센다.

    ⚠구판은 role=tool 메시지 **전체에 대한 부분문자열 스캔**(`'"n"' ∨ "'n'" ∨ "n:"`)이었다.
    tool 메시지는 한 표면이 아니다 — 레코드(행)·KB 히트(산문)·env 오류 문자열의 셋이고
    **뒤 둘은 외연을 닫지 않는다.** 구판이 권위자로 받아들인 것들(축자):

      · KB 산문 — *"- Time to qualify once you are **invited:** 1 month(s)."*
        (`doc_credit_cards_diamond_elite_card_009`) 문장이 콜론으로 끝났을 뿐이다.
        문서는 개념을 **언급**하지 누가 그것을 만족하는지 **열거**하지 않는다.
      · env TypeError — *"missing 12 required positional arguments: … **'contacted_merchant'** …"*
        인자 이름을 나열한 오류 문자열이 사실의 권위자가 된다.

    전수 실측(`x76_cwa_licence_surface.py`) — 공급된 닫힌-자료형 인자:

        arm    레코드가 닫아 준 것   구판이 통과시킨 것   그중 값이 실린 오-허가
        N97          4                   181                  176
        N97B       106                   255                  149

    교정 술어 = **키 자리**뿐이다. 키는 콜론 **앞**에 오고, 언급은 그렇지 않다:

        '"name":' ∨ "'name':"        구조화 반환의 키
        ^[ \\t]*name[ \\t]*:          줄-지향 레코드 나열의 키(줄머리)

    두 arm 720개 인자 자리에서 이 술어는 레코드-표면 판정과 **완전 일치**한다
    (N97 4·4·0·0 / N97B 106·106·0·0 = 오-허가 0 · 참-허가 손실 0).

    ★출력 형식 리터럴을 쓰지 않는다. `Found N record(s) in '…'` 머리말로 앵커하면 같은 수가
    나오지만 그것은 tau2 DB 도구의 **렌더링**이라 ABox/env를 갈면 레코드 분기가 조용히 죽는다
    ([[05]] 전이). 키 자리는 형식과 무관하다. 잔여 실패형은 KB 문서가 `name:`을 **줄머리**에
    쓰는 정의문이고(두 arm 720자리 중 0건) 그때는 허가 쪽으로 넘어간다 = 레버가 꺼질 뿐
    정당한 행동을 막지 않는다.
    """
    for m in messages or []:
        if getattr(m, "role", None) != "tool":
            continue
        c = getattr(m, "content", None)
        if not isinstance(c, str):
            continue
        if ('"%s":' % name) in c or ("'%s':" % name) in c:
            return True
        if re.search(r"^[ \t]*%s[ \t]*:" % re.escape(name), c, re.M):
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

    rec = msgs + [M("tool", "Found 1 record(s) in 'users':\n  rho_bank_subscription: true"
                            "\n  user_id: a1")]
    assert unknown_args(orch, "apply_for_credit_card", {"rho_bank_subscription": False}, rec) == []
    print("  ok   레코드 필드로 실재하면 통과(권위자=레코드)")

    prose = msgs + [M("tool", "1. Rho-Bank+ terms ID: doc_x Score: 12.3 Content: rho_bank_subscription benefits ...")]
    got = unknown_args(orch, "apply_for_credit_card", {"rho_bank_subscription": False}, prose)
    assert [k for k, _ in got] == ["rho_bank_subscription"], got
    print("  ok   KB 산문의 *언급*은 허가가 아니다")

    # x76이 실측한 오-허가 두 표면. 둘 다 콜론/따옴표를 담고 있어 구판을 통과했다.
    sentence = msgs + [M("tool", "4. Diamond Elite Card: Invitation Promo\n   Score: 0.63\n"
                                 "   Content: - Time to qualify once you are invited: 1 month(s).")]
    got = unknown_args(orch, "apply_for_credit_card", {"invited": False}, sentence)
    assert [k for k, _ in got] == ["invited"], got
    print("  ok   콜론으로 끝난 **문장**은 키가 아니다(N97 축자·구판 통과)")

    typeerr = msgs + [M("tool", "Error: Invalid arguments: file_dispute() missing 12 required "
                                "positional arguments: 'card_action', 'invited', 'user_id'")]
    got = unknown_args(orch, "apply_for_credit_card", {"invited": False}, typeerr)
    assert [k for k, _ in got] == ["invited"], got
    print("  ok   인자명을 나열한 **오류 문자열**은 권위자가 아니다(N97 축자·구판 통과)")

    jsonrec = msgs + [M("tool", '{"user_id": "a1", "invited": false}')]
    assert unknown_args(orch, "apply_for_credit_card", {"invited": False}, jsonrec) == []
    print("  ok   구조화 반환의 키 자리는 허가(형식 리터럴 없이)")

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
    print("PASS (12/12)")


if __name__ == "__main__":
    selftest()
