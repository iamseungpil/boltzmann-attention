# -*- coding: utf-8 -*-
"""T2_GIVE_REQUIRED 래칫 (2026-08-26) — 목록이 env 도출인가, 그리고 엔진이 **부르지 않는가**.

왜: 사용자 제안(*"dispatch 해야 할 call 은 리스트로 선언하고 엔진이 자동으로 부르게"*)의 앞쪽만
받았다. 목록은 **이미 env 에 있고**(레지스트리의 user-side), 뒤쪽(자동 호출)은 받지 않았다 —
`give_discoverable_user_tool` 은 변이 도구이고 표적 셋이 전부 `basis=['DB']` 라, 엔진이 부르면
우리 층이 gold 가 요구하는 상태 변경을 대신 수행하는 것이 된다([[05]]③·[[03b]]).
그 경계가 조용히 넘어가지 않도록 여기서 고정한다.

이 검정이 지키는 것:
  ① 술어가 **env 레지스트리**에서 나온다 — 도메인 도구 이름 리터럴 0
  ② 이미 준 도구는 지목하지 않는다(중복 지시 금지)
  ③ 레지스트리를 못 얻으면 **무발화**(fail-open)
  ④ 문면이 **정확한 호출**을 담는다([[64]] 무엇을 하면 풀리나)
  ⑤ ⛔엔진이 `give` 를 **실행하지 않는다** — 그 블록에 호출 생성이 없다
  ⑥ 기본 OFF · 상한 존재
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class _TC(object):
    def __init__(self, name, args):
        self.name = name
        self.arguments = args


class _M(object):
    def __init__(self, role, calls=None, content=""):
        self.role = role
        self.tool_calls = calls or []
        self.content = content


class _Orch(object):
    """env 레지스트리를 흉내 낸다 — `registry_from_env` 가 (agent, user) 를 돌려준다."""
    def __init__(self, agent, user):
        self._reg = (agent, user)


def main():
    import t2_gate_patch as G
    import t2_axis_levers as AX

    AGENT = {"file_credit_card_transaction_dispute_4829"}
    USER = {"submit_cash_back_dispute_0589", "deposit_check_3847"}
    real = AX.registry_from_env
    AX.registry_from_env = lambda o: getattr(o, "_reg", (set(), set()))
    try:
        orch = _Orch(AGENT, USER)
        tried = _M("user", [_TC("call_discoverable_user_tool",
                                {"discoverable_tool_name": "submit_cash_back_dispute_0589"})])

        # ① 술어가 걸린다 · ④ 정확한 호출을 담는다
        fb = G._give_required_fb([tried, tried], orch)
        assert fb, "손님이 미전달 도구를 불렀는데 무발화다"
        assert 'give_discoverable_user_tool(discoverable_tool_name="submit_cash_back_dispute_0589")' in fb, fb
        assert "2 time(s)" in fb, "몇 번 거절됐는지를 안 말한다: %r" % fb[-90:]

        # ② 이미 준 도구는 지목하지 않는다
        gave = _M("assistant", [_TC("give_discoverable_user_tool",
                                    {"discoverable_tool_name": "submit_cash_back_dispute_0589"})])
        assert G._give_required_fb([tried, gave], orch) is None, "이미 줬는데 또 시킨다"

        # 레지스트리 밖 이름은 무시한다(도메인 리터럴로 넓히지 않는다)
        other = _M("user", [_TC("call_discoverable_user_tool",
                                {"discoverable_tool_name": "not_a_registered_tool"})])
        assert G._give_required_fb([other], orch) is None, "레지스트리 밖 이름을 지목했다"

        # ③ fail-open
        assert G._give_required_fb([tried], _Orch(set(), set())) is None, "레지스트리 없이 발화했다"
        assert G._give_required_fb([], orch) is None
    finally:
        AX.registry_from_env = real

    src = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
    i = src.index("def _give_required_fb(")
    body = src[i:src.index("\ndef ", i + 1)]
    q = body.index('"""')
    code = body[body.index('"""', q + 3) + 3:]
    code = "\n".join(l for l in code.splitlines() if not l.strip().startswith("#"))
    # ① 도메인 리터럴 0
    for bad in ("dispute", "deposit", "card", "referral", "transaction"):
        assert bad not in code, "술어에 도메인 도구 이름이 들어왔다: %r" % bad
    # ⑤ 엔진이 실행하지 않는다
    j = src.index('os.environ.get("T2_GIVE_REQUIRED")')
    blk = src[j:src.index("T2_VALUE_ACQUIRE (C119)", j)]
    for bad in ("ToolCall(", "orig(self", "_autofetch", "invoke", "execute"):
        assert bad not in blk and bad not in code, (
            "엔진이 give 를 **실행**하려 한다(%r) — 변이 도구이고 표적이 DB 축이다"
            "([[05]]③·[[03b]])" % bad)

    # ⑥ 기본 OFF · 상한
    gs = io.open(os.path.join(HERE, "go_stack.sh"), encoding="utf-8").read()
    assert "T2_GIVE_REQUIRED=0" in gs, "기본 OFF 가 아니다"
    assert "T2_GIVE_REQUIRED_CAP" in gs, "상한이 없다"
    print("OK T2_GIVE_REQUIRED: env 레지스트리 도출 · 중복 지시 0 · fail-open 2종 · "
          "정확한 호출 문면 · **엔진 실행 0** · 기본 OFF")


if __name__ == "__main__":
    main()
