"""P3 — 엔진이 도구 이름을 말할 때 **부를 수 있는 형태**로 말하게 한다.

`[READ-FIRST]`와 `require_tool_before`는 필요한 도구를 **base 이름**으로 통지한다
(`get_credit_limit_increase_history`). 그런데 실제 잠금 이름은 `_4829` 접미사를 달고 있어
**통지 그대로는 호출이 불가능하다.** 현행 문구는 그 사실을 알고 "KB에서 전체 이름을 찾아라"라고
지시하는데, N97 전수 정독이 그 지시가 갈리는 지점을 보여줬다 — 같은 통지에서

  task_050  KB 검색 → `_4829` 잠금 → 완주
  task_052  shell grep ×6 → 동일 검사 4회 → 문맥 초과 사망
  task_051  shell grep ×7 → "도구 없음" 단정 → 이관

찾으라고 하는 대신 **그냥 준다.** 접미사는 env 레지스트리에서 기계 도출되므로
(`t2_axis_levers.registry_from_env` · C208 "env 레지스트리에서 기계-도출되므로 opex 0")
A2 저작이 0이고 도메인 리터럴도 0이다.

보수적으로 동작한다 — base 하나가 레지스트리에서 **정확히 하나**로 풀릴 때만 지목한다.
0개(미등록)나 2개 이상(모호)이면 그 항목은 건너뛰고 기존 문구가 그대로 남는다.
"""

import os
import re

_SUFFIX = re.compile(r"_\d{3,4}$")


def _fam(name):
    return _SUFFIX.sub("", name or "")


def enabled():
    return os.environ.get("T2_CALLABLE_HINT") == "1"


def registry(orch):
    """env가 표시한 discoverable 도구 이름 전체(접미사 포함). 실패하면 빈 집합 = 무발화."""
    try:
        import t2_axis_levers as _AX
        agent_d, user_d = _AX.registry_from_env(orch)
        return set(agent_d or ()) | set(user_d or ())
    except Exception:
        return set()


def resolve(orch, bases):
    """base 이름 → 접미사 포함 실명. 유일하게 풀리는 것만 돌려준다.

    반환 = [(base, full), ...] — 순서는 입력 순서를 따른다.
    """
    reg = registry(orch)
    if not reg:
        return []
    out = []
    for b in bases:
        hits = sorted(n for n in reg if _fam(n) == _fam(b))
        if len(hits) == 1:            # 0=미등록 · 2+=모호 → 지목하지 않는다
            out.append((b, hits[0]))
    return out


def hint(orch, bases, unlock_tool, dispatch_tool):
    """통지에 덧붙일 호출형 문장. 하나도 못 풀면 빈 문자열(문구 변화 0)."""
    if not enabled() or not (unlock_tool and dispatch_tool):
        return ""
    pairs = resolve(orch, bases)
    if not pairs:
        return ""
    forms = "; ".join('%s(agent_tool_name="%s") then %s with that name'
                      % (unlock_tool, full, dispatch_tool) for _, full in pairs)
    unresolved = [b for b in bases if b not in {p[0] for p in pairs}]
    tail = ("" if not unresolved else
            " The remaining names (%s) must still be looked up in the knowledge base."
            % ", ".join(unresolved))
    # 효과 지점의 박동 — 이 레버는 기존 문구에 문장을 덧붙일 뿐이라 태그가 없다. 발화 증명이
    # 없으면 x43(구현 완료·발화 0)을 반복한다([[30]]).
    try:
        from t2_lever_beat import beat as _beat
        _beat("T2_CALLABLE_HINT", "%d resolved" % len(pairs))
    except Exception:
        pass
    return (" Their exact callable forms are: %s.%s" % (forms, tail))


def selftest():
    class _Tk:
        pass

    class _Env:
        pass

    reg = {"get_credit_limit_increase_history_4829", "get_payment_history_6183",
           "get_all_user_accounts_by_user_id_3847", "close_debit_card_4721"}

    import t2_callable_hint as M
    M.registry = lambda orch: reg                      # env 접근을 대체

    os.environ["T2_CALLABLE_HINT"] = "1"
    h = M.hint(None, ["get_credit_limit_increase_history", "get_payment_history"],
               "unlock_discoverable_agent_tool", "call_discoverable_agent_tool")
    assert "get_credit_limit_increase_history_4829" in h, h
    assert "get_payment_history_6183" in h, h
    print("  ok   유일 해소 2건 → 호출형 2개 동봉")

    h2 = M.hint(None, ["get_payment_history", "no_such_tool"],
                "unlock_discoverable_agent_tool", "call_discoverable_agent_tool")
    assert "get_payment_history_6183" in h2 and "no_such_tool" in h2, h2
    assert "knowledge base" in h2
    print("  ok   미등록은 지목 안 하고 기존 안내로 남긴다")

    M.registry = lambda orch: {"x_1111", "x_2222"}
    assert M.hint(None, ["x"], "u", "c") == ""
    print("  ok   모호(2+)하면 지목하지 않는다")

    M.registry = lambda orch: set()
    assert M.hint(None, ["get_payment_history"], "u", "c") == ""
    print("  ok   레지스트리 도출 실패 = 무발화")

    M.registry = lambda orch: reg
    os.environ["T2_CALLABLE_HINT"] = "0"
    assert M.hint(None, ["get_payment_history"], "u", "c") == ""
    print("  ok   플래그 OFF면 문구 변화 0")

    os.environ["T2_CALLABLE_HINT"] = "1"
    assert M.hint(None, ["get_payment_history"], None, "c") == ""
    print("  ok   디스패처 미선언 도메인이면 무발화")
    print("PASS (6/6)")


if __name__ == "__main__":
    selftest()
