# -*- coding: utf-8 -*-
"""V7 — 도구 **호출 서명** compliance 게이트 (A2 EXT 선언 구동·2026-07-31).

설계 정본 = `ENGINE_LITERAL_REMEDIATION_DESIGN_2026_07_30.md` §12.

★근거(정책 축자·gold 아님): banking 정책이 서명을 **두 번** 규정한다 —
  "Use the `give_discoverable_user_tool(discoverable_tool_name)` function"
  "Explain to the user what the tool does and how to use it, **and what arguments to provide**.
   Just explaining isn't enough, you must use the `give_discoverable_user_tool(discoverable_tool_name)`."
⇒ **인자는 유저에게 말로 설명하고 호출은 이름만**이 도메인 사실이다. 실측: give 호출 105회 중
**82회(78%)가 `arguments`를 실어** 채점 규약상 자동 불일치(`PRED_EXTRA_KEY`)를 만든다.

★[[05]] 경계: 서명 **내용**은 도메인 사실이라 **A2(EXT·스키마 상수)**에 둔다 — 엔진에 두면 위반.
  엔진은 "선언된 키 집합 밖의 키가 있는가"라는 **도메인-일반 닫힌 술어**만 판정한다.
★[[10]] 경계: 엔진은 **인자를 떼지 않는다**. deny + 재발행 요구뿐이고 **선택은 모델**이 한다
  (C151에서 reroute/strip = gaming으로 기각·deny+regen = compliance로 채택).

A2 형태(도메인별 EXT):
    "tool_signatures": {"give_discoverable_user_tool": ["discoverable_tool_name"]}

기본 **OFF** — `T2_TOOL_SIGNATURE=1`일 때만 동작(비교성 보존·롤백 = 플래그).
"""
import os

FEEDBACK = ("[SIGNATURE] `{tool}` takes only {allowed} in this domain; you also passed {extra}. "
            "Re-issue the call with the declared argument(s) only. If the user needs argument "
            "values, state them in your message to the user instead of putting them in this call.")


def declared_signature(tool_name, a2):
    """A2가 이 도구의 서명을 선언했나. 미선언이면 None = **레버 skip**(U2′ 안전측)."""
    sig = ((a2 or {}).get("tool_signatures") or {}).get(tool_name)
    return list(sig) if isinstance(sig, (list, tuple)) and sig else None


def signature_violation(tool_name, args, a2):
    """선언 서명 밖 키가 실렸으면 피드백 문자열, 아니면 None (순수 함수)."""
    if os.environ.get("T2_TOOL_SIGNATURE") != "1":
        return None
    allowed = declared_signature(tool_name, a2)
    if allowed is None:
        return None
    extra = [k for k in (args or {}) if k not in allowed]
    if not extra:
        return None
    return FEEDBACK.format(tool=tool_name, allowed=", ".join("`%s`" % k for k in allowed),
                           extra=", ".join("`%s`" % k for k in sorted(extra)))


if __name__ == "__main__":
    A2 = {"tool_signatures": {"give_discoverable_user_tool": ["discoverable_tool_name"]}}
    cases = [
        ("선언 서명대로", "give_discoverable_user_tool", {"discoverable_tool_name": "x"}, False),
        ("여분 arguments", "give_discoverable_user_tool",
         {"discoverable_tool_name": "x", "arguments": "{}"}, True),
        ("미선언 도구", "call_discoverable_agent_tool", {"agent_tool_name": "y", "arguments": "{}"}, False),
        ("빈 인자", "give_discoverable_user_tool", {}, False),
    ]
    os.environ["T2_TOOL_SIGNATURE"] = "1"
    ok = 0
    for name, tool, args, want in cases:
        got = signature_violation(tool, args, A2) is not None
        ok += (got == want)
        print("  %-16s want=%-5s got=%-5s %s" % (name, want, got, "OK" if got == want else "FAIL"))
    # A2 미선언 도메인 = 전면 skip
    assert signature_violation("give_discoverable_user_tool",
                               {"discoverable_tool_name": "x", "arguments": "{}"}, {}) is None
    print("  미선언 도메인 skip: OK")
    os.environ["T2_TOOL_SIGNATURE"] = "0"
    assert signature_violation("give_discoverable_user_tool",
                               {"discoverable_tool_name": "x", "arguments": "{}"}, A2) is None
    print("  기본 OFF no-op: OK")
    print("selftest %d/%d" % (ok, len(cases)))
