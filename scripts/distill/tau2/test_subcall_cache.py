# -*- coding: utf-8 -*-
"""회귀 — 같은 물음을 두 번 묻지 않는다 (`T2_SUBCALL_CACHE`·temperature 0 한정).

★결손 (2026-08-31 · x697 라이브 전수):
    intent_operator_formalize 6회 = **24,560토큰 = 그 런 생성의 75%**
    그중 5회가 프롬프트까지 동일 — prompt 239 · gen 4,108 · reason 18,220B · content 40B
    중복분만 **16,432토큰 = 런 전체의 50%**
  창이 *손님 발화 마지막 6개* 라, 손님이 말하지 않은 턴에는 프롬프트가 글자 하나 안 바뀐다.

⚠**정보 손실 0인 조건에서만** 캐시한다: temperature==0(응답이 결정론·실측에서 reason 바이트까지
  동일). 온도가 있으면 재표집이 의미이므로 캐시하지 않는다 — 닫힌 술어 하나([[22]]).
⚠범위는 에이전트(=sim) 하나. 프로세스 전역이면 sim 간 오염이다.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_subcall as SC


class _Msg:
    def __init__(self, content): self.content = content


class _LA:
    """호출 횟수를 세는 가짜 생성기."""
    def __init__(self, answer='{"tool": "submit_referral"}'):
        self.calls, self.answer = 0, answer

    def generate(self, model=None, tools=None, messages=None, call_name=None, **kw):
        self.calls += 1
        return _Msg(self.answer)


class _Agent:
    def __init__(self, temp=0.0):
        self.llm = "m"
        self.llm_args = {"temperature": temp} if temp is not None else {}


def _UM(content=None, **kw):
    return _Msg(content)


def test_identical_prompt_is_asked_once():
    os.environ["T2_SUBCALL_CACHE"] = "1"
    la, ag = _LA(), _Agent(0.0)
    outs = [SC.sub_generate(ag, la, _UM, "same question", "intent_operator_formalize")
            for _ in range(5)]
    assert la.calls == 1, la.calls
    assert len(set(outs)) == 1 and outs[0].startswith('{"tool"')


def test_different_prompt_is_asked_again():
    la, ag = _LA(), _Agent(0.0)
    SC.sub_generate(ag, la, _UM, "q1", "intent_operator_formalize")
    SC.sub_generate(ag, la, _UM, "q2", "intent_operator_formalize")
    assert la.calls == 2


def test_same_prompt_different_sub_is_asked_again():
    """캐시 키에 서브 이름이 있어야 한다 — 다른 서브는 같은 글을 달리 읽는다."""
    la, ag = _LA(), _Agent(0.0)
    SC.sub_generate(ag, la, _UM, "q", "intent_operator_formalize")
    SC.sub_generate(ag, la, _UM, "q", "recommend_formalize")
    assert la.calls == 2


def test_no_cache_when_temperature_is_nonzero():
    la, ag = _LA(), _Agent(0.7)
    for _ in range(3):
        SC.sub_generate(ag, la, _UM, "same", "intent_operator_formalize")
    assert la.calls == 3, "온도가 있으면 재표집이 의미다 — 캐시하면 안 된다"


def test_cache_is_per_agent_not_global():
    la = _LA()
    a1, a2 = _Agent(0.0), _Agent(0.0)
    SC.sub_generate(a1, la, _UM, "same", "intent_operator_formalize")
    SC.sub_generate(a2, la, _UM, "same", "intent_operator_formalize")
    assert la.calls == 2, "sim 간 오염 금지"


def test_flag_off_restores_old_behaviour():
    os.environ["T2_SUBCALL_CACHE"] = "0"
    try:
        la, ag = _LA(), _Agent(0.0)
        for _ in range(3):
            SC.sub_generate(ag, la, _UM, "same", "intent_operator_formalize")
        assert la.calls == 3
    finally:
        os.environ["T2_SUBCALL_CACHE"] = "1"


def test_probe_cap_is_independent_of_the_global_cap():
    """프로브 상한이 전역을 따라 부풀면 사고 예산이 배가된다(2026-08-31 실측 퇴행)."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2_run_gated.py"),
               encoding="utf-8").read()
    assert '_kw["max_tokens"] = _jmt if _jmt else _cur' in src
    assert 'max(int(_cur or 0), _jmt)' not in src, "전역과 max 를 취하면 전역이 커질 때 프로브가 부푼다"


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
    print("ALL PASS")
