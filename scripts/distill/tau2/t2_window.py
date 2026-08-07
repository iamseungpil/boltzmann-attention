# -*- coding: utf-8 -*-
"""C6 창 — **우리 층은 모델이 나가려는 순간에 말한다. 사임 턴만 기다리지 않는다.**

계약 다섯 개는 *무엇을 판정하는가*(술어)를 나눈다. 그런데 실측 충돌 census
(`CONFLICT_ARBITRATION_THEORY_2026_08_06` §1b)의 여섯 종 중 **T6(창 부재)** 는 술어 문제가 아니다 —
**아무도 말하지 않는 것**이고, 실패 198 sim 중 **109(55%)** 가 여기다. 그래서 다섯 상자를 아무리
잘 나눠도 남는다. 축이 직교하므로 여섯째 계약으로 세운다(사용자 지시 2026-08-07).

**101/102 부검이 같은 자리를 독립적으로 짚었다**:
  · `[ORDER]`가 turn 4·6·8 이후 영구 침묵 — 우리 층 자체는 turn 58까지 살아 있는데도
  · 예산을 없애도 push 탐지가 sim당 3회 그대로(=cap이 아니라 **창**이 구속조건)
  · 그런데 실제 write는 산문에서 나온다: 제출 유형이 직전 답변에 **101 87/87 · 102 61/63**

⇒ 창 = 사임 턴 ∪ **종결성 행동을 실행하려는 턴** ∪ **그 행동을 손님에게 지시하는 답변**.
마지막 항이 지금 비어 있는 자리이고, 그 자리가 DB 채점을 결정한다.

⚠**차단 없이 표면화만으로 시작한다**(이론 문서 §3-T6 경고): 004형 *"마지막 턴 소각"* 재현 위험.
창을 넓히는 것은 발화 기회를 늘리는 일이므로 Δspurious를 함께 재야 한다(등대 제1원리).
"""

__all__ = ["RESIGN", "ACTING", "INSTRUCTING", "opened", "why"]

RESIGN, ACTING, INSTRUCTING = "resign", "acting", "instructing"


def _text(m):
    c = getattr(m, "content", None)
    if c is None and isinstance(m, dict):
        c = m.get("content")
    return c if isinstance(c, str) else ""


def _calls(m):
    return list(getattr(m, "tool_calls", None) or (m.get("tool_calls") if isinstance(m, dict) else None) or [])


def opened(am, targets=(), name_of=None):
    """이 답변이 발화 창을 여는가 — 세 종류 중 열린 것들의 집합.

    `targets` = 이 대화에서 아직 미충족인 표적 행동 이름들(C2가 준다).
    `name_of` = 도구 호출 → 이름(디스패처를 벗기는 지식은 호출자 쪽에 둔다).

    **INSTRUCTING의 술어는 이름 언급이 아니라 표적 이름의 등장이다.** 산문을 해석하지 않는다 —
    표적 이름이 답변에 있는가만 본다(닫힌 술어·[[22]]). 손님이 그 이름으로 도구를 부르기 때문에
    이 술어가 실제 write 경로와 일치한다(87/87 · 61/63).
    """
    out = set()
    calls = _calls(am)
    txt = _text(am)
    if not calls and txt.strip():
        out.add(RESIGN)
    if calls and targets:
        got = {(name_of(c) if name_of else getattr(c, "name", None)) for c in calls}
        if got & set(targets):
            out.add(ACTING)
    if txt and targets:
        for t in targets:
            if t and t in txt:
                out.add(INSTRUCTING)
                break
    return out


def why(kinds):
    """왜 열렸는지 한 줄 — 로그·사이드카가 창 종류별로 계수될 수 있게."""
    if not kinds:
        return ""
    order = [ACTING, INSTRUCTING, RESIGN]
    return "+".join(k for k in order if k in kinds)


if __name__ == "__main__":                       # 자기검정(도메인 무관)
    class M:
        def __init__(self, content=None, tool_calls=None):
            self.content, self.tool_calls = content, tool_calls

    class C:
        def __init__(self, name):
            self.name = name

    T = ["do_the_thing"]
    assert opened(M(content="I cannot help."), T) == {RESIGN}
    assert opened(M(tool_calls=[C("do_the_thing")]), T) == {ACTING}
    assert INSTRUCTING in opened(M(content="please run do_the_thing yourself"), T)
    assert opened(M(content="hello"), T) == {RESIGN}          # 표적 미언급 = 사임만
    assert opened(M(content="", tool_calls=[C("other")]), T) == set()
    print("t2_window self-test OK ·", why(opened(M(content="run do_the_thing"), T)))
