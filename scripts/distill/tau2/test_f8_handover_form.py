#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""F8‴ — `[ACTION]` 이 **호출 형식과 도구를 쌍으로** 말하는가 (사용자 제안 2026-09-07 · x817 §14).

구판 문면은 1종이었다: *"tell the customer to run {tool} themselves"*.
그러나 discoverable 은 **먼저 건네야** 손님이 부를 수 있다. 도메인 정책 축자:
  *"Just explaining isn't enough, you must use the `give_discoverable_user_tool(
    discoverable_tool_name)` function"*

[[70]] 실측(회수분 전수): `[ACTION]` 발화 **1,824회 중 1,653(91%)**이 «아직 안 건네진» 도구를
지목하면서 «시켜라»라고 했다 — **49 태스크**. `task_015` 는 그 자리에서 gold 의 give 까지 막혔다.
2026-08-23 R8-⑶ 이 이 모순을 이미 지적했고("무엇을 하면 풀리는지를 **틀리게** 말하게 된다"),
이것이 그 처방이다.

반증조건: ①건네지지 않았는데 «시켜라»가 나가면 FAIL ②이미 건네졌는데 «건네라»가 나가면 FAIL
          ③give 도구 이름이 엔진 리터럴이면 FAIL([[05]])
"""
import io, os, re, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import t2_gate_patch as GP

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
GIVE, TARGET = "hand_over_tool", "customer_side_tool"


class _TC(object):
    def __init__(self, name, args): self.name, self.arguments = name, args


class _M(object):
    def __init__(self, tool_calls=None): self.tool_calls = tool_calls or []


class F8Handover(unittest.TestCase):

    def test_tool_given_detects_handover(self):
        msgs = [_M([_TC(GIVE, {"discoverable_tool_name": TARGET})])]
        self.assertTrue(GP._tool_given(msgs, GIVE, TARGET))

    def test_tool_given_false_when_not_handed(self):
        self.assertFalse(GP._tool_given([], GIVE, TARGET))
        self.assertFalse(GP._tool_given([_M([_TC(GIVE, {"discoverable_tool_name": "other"})])],
                                        GIVE, TARGET))

    def test_handover_branch_exists_and_is_conditional(self):
        """분기가 `_tool_given` 을 술어로 쓰고, 이미 건네진 경우는 구판 문면으로 간다."""
        self.assertIn("_handed = bool(_tool_given(state.messages, _gtool, _utgt))", SRC)
        self.assertIn("if _gtool and not _handed:", SRC)
        i = SRC.index("if _gtool and not _handed:")
        tail = SRC[i:i + 3000]
        self.assertIn("else:", tail, "이미 건네진 경우의 분기가 없다")
        self.assertIn('get("user_action_feedback")', tail, "구판 문면으로 되돌아가지 않는다")

    def test_handover_text_pairs_form_with_tool(self):
        """★사용자 제안의 핵심 — **호출 형식과 도구를 쌍으로** 말해야 한다."""
        i = SRC.index("user_action_handover_feedback")
        blk = SRC[i:i + 1200]
        self.assertIn("{give}", blk, "호출 형식(래퍼 도구)이 문면에 없다")
        self.assertIn("{tool}", blk, "대상 도구가 문면에 없다")
        self.assertIn("discoverable_tool_name='{tool}'", blk, "인자 형식이 없다 — 형식×도구 쌍이 아니다")
        self.assertIn("Explaining is not enough", blk, "정책 축자의 요지가 빠졌다")

    def test_give_tool_name_comes_from_a2(self):
        """[[05]] — give 도구 이름은 **A2 선언**에서 온다. 엔진 리터럴이면 FAIL."""
        i = SRC.index("_gtool = ")
        blk = SRC[i:i + 400]
        self.assertIn('get("value_acquisition")', blk)
        self.assertIn('get("give_tool")', blk)
        self.assertNotIn('"give_discoverable_user_tool"', blk,
                         "엔진에 도메인 도구명 리터럴이 들어갔다")

    def test_no_a2_give_declaration_is_legacy(self):
        """A2 가 give 도구를 선언하지 않은 도메인 = 구판 거동 그대로([[54]])."""
        i = SRC.index("_gtool = ")
        self.assertIn("if _gtool and not _handed:", SRC[i:i + 900],
                      "_gtool 이 없을 때 구판으로 가는 가드가 없다")

    def test_instrument_fires_with_the_name(self):
        """[[81]] — 첫 런에서 발화를 확인할 계기가 있어야 한다."""
        self.assertIn("[T2_ACTIONREQ] handover:", SRC)


if __name__ == "__main__":
    unittest.main(verbosity=2)
