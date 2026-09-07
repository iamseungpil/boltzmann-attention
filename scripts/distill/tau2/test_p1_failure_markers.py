#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""P1 — 실패 표지가 **A2 선언 5종 전부**로 판정되는가 (x737 §9b · [[25]]).

이 환경은 오류를 평문으로도 돌려준다(`Failed to …` · `NOT_VERIFIED` · `Unknown discoverable tool` …).
`"Error:"` 접두사 하나만 보면 **실패한 호출이 성공으로 계상**되고 의존 그래프가 조기 전진한다.
`_executed_tool_names` 는 2026-08-07(102 부검)에 고쳤는데 **쌍둥이 `_executed_tool_counts` 만**
안 고쳐져 같은 판정이 두 자리에서 갈렸다.

반증조건: 선언된 표지로 시작하는 결과가 성공으로 세어지면 FAIL.
          `a2` 미전달 시 구판과 다르게 동작하면 FAIL(진행 중 팔 보호).
"""
import io, json, os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import t2_gate_patch as GP

HERE = os.path.dirname(os.path.abspath(__file__))
A2 = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
MARKS = A2.get("failure_markers") or []
VERDICT = set(A2.get("verdict_markers") or [])
FAILMARKS = [m for m in MARKS if m not in VERDICT]


class _TC(object):
    def __init__(self, i, n): self.id, self.name = i, n


class _M(object):
    def __init__(self, role, tool_calls=None, mid=None, content="", error=False):
        self.role, self.tool_calls, self.id = role, tool_calls or [], mid
        self.content, self.error = content, error


def _conv(result_text, error=False):
    return [_M("assistant", [_TC("c1", "t_x")]),
            _M("tool", mid="c1", content=result_text, error=error)]


class P1FailureMarkers(unittest.TestCase):

    def test_a2_declares_five(self):
        self.assertEqual(len(MARKS), 5, "선언이 5종이 아니다: %s" % MARKS)
        self.assertIn("NOT_VERIFIED", MARKS)

    def test_each_failure_marker_counts_as_failure(self):
        for k in FAILMARKS:
            c = GP._executed_tool_counts(_conv(k + " something happened"), A2)
            self.assertEqual(c.get("t_x", 0), 0,
                             "선언 표지 %r 로 시작한 결과를 성공으로 셌다" % k)

    def test_verdict_marker_still_counts_as_executed(self):
        """★부호표 (x817 §7): 「도구는 돌았고 판정이 부정」인 표지는 **수행됨**으로 센다.

        회수분 실측 `NOT_VERIFIED` 3,353회 중 **통과 sim 704건**이 영향권이다 — 그것까지
        미실행으로 세면 절차 계수가 흔들린다([[70]] 파는 쪽). 이 검정이 그 704 를 지킨다.
        """
        self.assertTrue(VERDICT, "verdict_markers 선언이 없다 — 부호표가 반영되지 않았다")
        for k in VERDICT:
            c = GP._executed_tool_counts(_conv(k + " name mismatch"), A2)
            self.assertEqual(c.get("t_x", 0), 1,
                             "판정 표지 %r 를 미실행으로 셌다 (통과 sim 704건 위험)" % k)

    def test_plain_success_still_counts(self):
        c = GP._executed_tool_counts(_conv('{"ok": true}'), A2)
        self.assertEqual(c.get("t_x", 0), 1, "정상 결과를 실패로 셌다 — 레버가 과잉 억제된다")

    def test_without_a2_is_legacy_behaviour(self):
        """`a2` 미전달 = 구판(‘Error:’ 만) — 거동 변화 0 이어야 한다([[54]])."""
        self.assertEqual(GP._executed_tool_counts(_conv("Failed to reach the record")).get("t_x", 0), 1)
        self.assertEqual(GP._executed_tool_counts(_conv("Error: nope")).get("t_x", 0), 0)

    def test_error_flag_still_wins(self):
        self.assertEqual(GP._executed_tool_counts(_conv("fine", error=True), A2).get("t_x", 0), 0)

    def test_twin_helpers_agree_on_failure_markers(self):
        """실패 표지에서는 두 헬퍼가 같아야 한다 — 그 불일치가 P1 의 본체였다.

        ⚠판정 표지(`verdict_markers`)에서는 **의도적으로 갈린다**: `_executed_tool_names` 는
        「성공했나」를, `_executed_tool_counts` 는 「수행됐나」를 묻는다.
        """
        for k in FAILMARKS + ['{"ok": true}']:
            conv = _conv(k + " tail")
            in_names = "t_x" in GP._executed_tool_names(conv, A2)
            in_counts = GP._executed_tool_counts(conv, A2).get("t_x", 0) > 0
            self.assertEqual(in_names, in_counts, "두 헬퍼가 %r 에서 갈렸다" % k)

    def test_every_call_site_passes_a2(self):
        src = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
        bad = [l.strip() for l in src.split("\n")
               if "_executed_tool_counts(" in l and not l.lstrip().startswith("def")
               and not l.lstrip().startswith("#") and ", a2)" not in l]
        self.assertEqual(bad, [], "a2 를 안 넘기는 호출부가 남아 있다")


if __name__ == "__main__":
    unittest.main(verbosity=2)
