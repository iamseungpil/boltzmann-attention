#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""P6 — `T2_VALUE_ACQUIRE` 가 **선언된 write 가 이미 성공했으면 침묵**하는가 (x737 §9b R1).

이 넛지는 "값을 얻어 W 로 가라"는 말이다. W 가 이미 성공했으면 할 일이 없는 말이고,
그런데도 계속 나가면 모델을 이미 끝난 경로로 되돌린다.

x778 부호표(6,103 sim · VA 사이트 1,008): 발화 유지 1001/1008(99.3%) · 침묵 7 ·
gold 손실 1(task_041 · 그 sim 은 이미 0.0) · **통과 sim 손실 0**.

반증조건: ①W 미실행인데 침묵하면 FAIL(주석이 지키려던 표적을 잃는다)
          ②W 성공인데 발화하면 FAIL
"""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import t2_gate_patch as GP

SPEC = {"write": "W_tool", "arg": "the_arg", "acquire_tool": "acq_tool",
        "give_tool": "give_tool", "producer_marker": "Executed: acq_tool",
        "reask_signals": ["last 4"],
        "feedback": "[VALUE-ACQUIRE] get {arg} via {acquire_tool} then {write}"}


class _M(object):
    def __init__(self, role, content="", tool_calls=None):
        self.role, self.content, self.tool_calls = role, content, tool_calls or []


def _fb(executed):
    """재요청 신호가 이미 있는 대화 — 조건 ③ 통과 상태를 만든다."""
    am = _M("assistant", "can you give me the last 4 digits?")
    msgs = [_M("assistant", "please tell me the last 4")]
    return GP._value_acquire_fb(am, msgs, [SPEC], a2={}, executed=set(executed))


class P6ValueAcquireR1(unittest.TestCase):

    def test_speaks_when_write_not_done(self):
        """W 미실행 = 이 넛지의 본래 표적. 반드시 말해야 한다(over-block 0)."""
        self.assertIsNotNone(_fb([]), "W 미실행인데 침묵했다 — 031·048·051·053 표적을 잃는다")

    def test_silent_when_declared_write_succeeded(self):
        self.assertIsNone(_fb(["W_tool"]), "선언 write 가 이미 성공했는데도 말했다")

    def test_other_tool_success_does_not_silence(self):
        """무관한 도구가 성공한 것으로 침묵하면 안 된다 — 술어는 **선언된 W** 하나다."""
        self.assertIsNotNone(_fb(["some_other_tool", "acq_tool"]))

    def test_no_reask_signal_stays_silent(self):
        """조건 ③(재요청)이 없으면 애초에 말하지 않는다 — P6 과 무관하게 종전 거동."""
        am = _M("assistant", "here is your statement")
        self.assertIsNone(GP._value_acquire_fb(am, [], [SPEC], a2={}, executed=set()))

    def test_feedback_placeholders_filled(self):
        out = _fb([])
        for ph in ("{arg}", "{acquire_tool}", "{write}"):
            self.assertNotIn(ph, out, "치환되지 않은 자리표시자 %s" % ph)
        self.assertIn("the_arg", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
