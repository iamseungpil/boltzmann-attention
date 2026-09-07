#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""F2 — 에이전트가 **이미 들고 있는** 행동 도구를 배포로 넘기라고 하지 않는가 (x808 §7-5 · x817 §7).

구판 `RECOMMEND_OFFER_FB` 는 `{offer}` 를 A2 `offer_tool`(배포 도구)로 **고정**했다. 실물(x808 §7-2):
`task_023` turn 64 에 우리가 "give_discoverable_user_tool 로 넘겨라"라고 했고, 모델은 **자기 도구
목록에 이미 있는 이름을 KB 에서 찾기 시작**했다. base 4/4 는 그 도구를 직접 불러 통과했다.

[[70]] 부호표(회수분 전수 · 발화 62 sim · 20 태스크):
   gold 이 직접 호출을 요구      24 통과 / 36 실패  · 19 태스크   ← 사는 쪽
   gold 이 배포(give)를 요구      0 /  0  ·  0 태스크   ← **파는 쪽 0**
⇒ 절충 없음.

반증조건: ①보유 도구인데 배포 문면이 나가면 FAIL ②미보유인데 직접호출 문면이 나가면 FAIL
"""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import t2_resolve as RS

ACTION, OFFER, OPERAND = "do_the_thing", "hand_to_user_tool", "flavour"
A2 = {"recommendation_verify": {"action_tool": ACTION, "offer_tool": OFFER,
                                "operand": OPERAND, "offer_name_key": "target_name"}}


class _T(object):
    def __init__(self, n): self.name = n


class _Agent(object):
    def __init__(self, held): self.tools = [_T(n) for n in held]


class _M(object):
    def __init__(self, role, content="", tool_calls=None):
        self.role, self.content, self.tool_calls = role, content, tool_calls or []


def _run(held, msgs=None, correct="THE_RIGHT_ONE"):
    """`_formalize_recommendation` 은 LLM 서브콜이라 대역으로 고정한다(엔진 경로만 검정)."""
    orig = RS._formalize_recommendation
    RS._formalize_recommendation = lambda *a, **k: (True, correct)
    try:
        am = _M("assistant", "here are some options you might like")
        # `la`/`UserMessage` 는 서브콜용 핸들 — None 이면 함수가 진입 전에 빠진다(:1166).
        return RS.resolve_recommendation(am, msgs if msgs is not None else [],
                                         A2, agent=_Agent(held), la=object(),
                                         UserMessage=object(), transfer_tools=set())
    finally:
        RS._formalize_recommendation = orig


class F2OfferDirect(unittest.TestCase):

    def test_agent_holds_it_prescribes_direct_call(self):
        r = _run([ACTION])
        self.assertEqual(r.get("status"), "deny", r)
        fb = r["feedback"]
        self.assertIn("call it directly", fb)
        self.assertNotIn(OFFER, fb, "보유 도구인데 배포 도구를 지목했다 (023 을 죽인 그 문면)")
        self.assertIn(ACTION, fb)
        self.assertIn("THE_RIGHT_ONE", fb)

    def test_agent_lacks_it_keeps_the_give_wording(self):
        """미보유면 배포가 유일한 길이다 — 구판 문면 그대로(거동 변화 0)."""
        r = _run(["something_else"])
        self.assertEqual(r.get("status"), "deny", r)
        self.assertIn(OFFER, r["feedback"])
        self.assertNotIn("call it directly", r["feedback"])

    def test_already_called_directly_is_not_nagged(self):
        """직접 부른 뒤에도 계속 말하면 [[64]] 처방이 소음이 된다."""
        msgs = [_M("assistant", "", [type("tc", (), {"name": ACTION, "arguments": {}})()])]
        r = _run([ACTION], msgs=msgs)
        self.assertEqual(r.get("status"), "ok", r)

    def test_direct_text_never_tells_it_to_search(self):
        """023 실물: 모델이 «이미 가진 이름»을 KB 에서 찾기 시작했다. 그 길을 문면이 닫아야 한다."""
        fb = _run([ACTION])["feedback"]
        self.assertIn("do not search the knowledge base", fb.lower())

    def test_agent_holds_predicate_is_safe(self):
        self.assertFalse(RS._agent_holds(None, ACTION))
        self.assertFalse(RS._agent_holds(_Agent([]), ACTION))
        self.assertFalse(RS._agent_holds(_Agent([ACTION]), None))
        self.assertTrue(RS._agent_holds(_Agent([ACTION]), ACTION))

    def test_no_domain_literal_in_the_new_text(self):
        src = RS.RECOMMEND_OFFER_DIRECT_FB
        for lit in ("apply_for_credit_card", "give_discoverable_user_tool", "card_type"):
            self.assertNotIn(lit, src, "새 문면에 도메인 리터럴 %s ([[05]])" % lit)


if __name__ == "__main__":
    unittest.main(verbosity=2)
