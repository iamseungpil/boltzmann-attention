# -*- coding: utf-8 -*-
"""C4 역할 — **도구의 실행 주체와 인자의 소비자는 env 레지스트리에서 도출된다.**

계약 정의 = `GENERAL_CONTRACTS_DESIGN_2026_08_06.md` §2-C4.

왜 모듈이 필요한가: 충돌 census(`CONFLICT_ARBITRATION_THEORY_2026_08_06` §1b)의 **T1 사실 모순**은
중재 대상이 아니다 — 우리 층 두 문구가 같은 명제에 **반대 진리값**을 말한 것이고(계열 A: `[ACTION]`
"손님 도구다" ↔ `unified_regen` "네 도구다"), 처방은 *"같은 명제를 말하는 모든 문구는 **하나의 술어
함수**에서 답을 받는다. 판정 불가면 그 문장을 뺀다"* 이다.

> **불변식 I1**: 우리 층이 같은 명제에 대해 서로 다른 진리값을 말하는 턴 수 = **0**.

지금 이 판정은 엔진 여러 곳에 **인라인으로 복제**돼 있다(`_exec_side` 지역 함수 등). 복제가 있는 한
I1은 코드로 보장되지 않는다 — 그래서 여기 한 곳에 둔다. 도메인 데이터 0(전부 레지스트리 파생).

⚠env를 **근거로 인용하지 않는다**([[25]]): env는 user-sim과 같은 외부 주장이다. 여기서 하는 일은
"우리가 가진 도구 목록"이라는 **우리 쪽 사실**을 읽는 것뿐이고, 그것으로 손님에게 무엇이 참인지
단정하지 않는다.
"""

__all__ = ["executor_of", "AGENT", "USER", "UNKNOWN", "consumers_of"]

AGENT, USER, UNKNOWN = "assistant", "user", None


def _names(obj, attr):
    got = getattr(obj, attr, None)
    if callable(got):
        try:
            got = got()
        except Exception:
            return set()
    out = set()
    for t in (got or []):
        n = getattr(t, "name", None)
        if n:
            out.add(str(n))
    return out


def executor_of(tool_name, agent=None, env=None):
    """이 도구를 **누가 실행하는가** — 세 집합의 3갈래 판정. 모르면 `UNKNOWN`.

    `UNKNOWN`을 침묵으로 쓰는 것이 계약이다: 판정할 수 없으면 그 문장을 빼야지, 추측해서
    말하면 T1(사실 모순)이 다시 생긴다.
    """
    n = str(tool_name or "")
    if not n:
        return UNKNOWN
    if n in _names(agent, "tools"):
        return AGENT
    ut = getattr(env, "user_tools", None) if env is not None else None
    if ut is not None:
        if n in _names(ut, "tools"):
            return USER
        if n in _names(ut, "get_discoverable_tools"):
            return USER
    return UNKNOWN


def consumers_of(arg_name, agent=None, env=None):
    """그 인자를 **실제로 받는 호출**들 — 시그니처에서 도출(설계서 C4-ⓑ).

    *"이 값을 얻으라"* 만 말하면 048처럼 **아무 데도 안 쓸 값을 열 턴 쫓는다**. 어디에 쓰이는지를
    같이 말할 수 있어야 그 지시가 실행 가능해진다. 표 없음·판단 0 — 스키마 조회뿐이다.
    """
    want = str(arg_name or "")
    out = []
    if not want:
        return out
    for holder, _ in ((agent, "tools"), (getattr(env, "user_tools", None), "tools")):
        got = getattr(holder, "tools", None) if holder is not None else None
        for t in (got or []):
            try:
                sc = t.openai_schema
                fn = sc.get("function") if isinstance(sc.get("function"), dict) else sc
                props = (fn.get("parameters") or {}).get("properties") or {}
            except Exception:
                continue
            if want in props:
                n = getattr(t, "name", None)
                if n and n not in out:
                    out.append(str(n))
    return out
