#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""P11 — 손님이 말한 **명시 수량**이 우리 **나열 추정**을 이기는가 (x737 §9b · x817).

`_enum_items` 는 쉼표 나열을 세는 추정치다. 손님이 "two items" 라고 **말했는데** 우리가 3으로
세면, E-PLAN L1 이 「이 요청은 여러 레코드에 걸친다」를 거짓 전제 위에 단정한다(task_004).

[[70]] 부호표(회수분 user 발화 37,119건 전수): 충돌 105건 =
  사는 쪽 실패 sim 95(34 태스크) : 파는 쪽 통과 sim 10(7 태스크 004 007 010 017 057 075 100)

반증조건: ①수량 미언급 발화의 거동이 바뀌면 FAIL(종전 보존) ②명시 수량보다 큰 추정이 남으면 FAIL
"""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import t2_eplan_patch as E

REAL_004 = ("I don't have my user ID handy. For verification, though, here are two items: "
            "- Email on the account: a@b.com - Phone number: 206-555-0293")


# 가짜 A2 eplan spec — 엔진에 도구명 하드코딩 0([[05]])임을 도구명 자체로 검증
SPEC = {"list_enumerator": "list_tool", "detail_reader": "detail_tool", "entity_key": "eid"}


def _led():
    return E.PlanLedger(SPEC)


class P11QtyBeatsEnum(unittest.TestCase):

    def test_raw_helpers_still_disagree(self):
        """근거 재확인 — 두 계기가 실제로 갈린다(이 수리의 전제)."""
        self.assertEqual(E._parse_qty(REAL_004), 2)
        self.assertEqual(E._enum_items(REAL_004), 3)

    def test_explicit_quantity_caps_the_estimate(self):
        l = _led(); l.accumulate_qty(REAL_004)
        self.assertEqual(l.enum_items, 2, "손님이 two 라 했는데 추정 3이 남았다")
        self.assertFalse(l.multi_entity_hint, "거짓 멀티엔티티 힌트가 켜졌다 (task_004 의 원인)")

    def test_no_quantity_stated_is_unchanged(self):
        """수량 미언급이면 종전 그대로 — 나열만 있는 발화는 손대지 않는다([[54]])."""
        t = "I need help with my checking, savings, and credit card accounts"
        raw = E._enum_items(t)
        l = _led(); l.accumulate_qty(t)
        self.assertEqual(l.enum_items, raw)
        self.assertEqual(E._parse_qty(t), 0, "이 문장에 명시 수량이 있으면 검정이 무의미하다")

    def test_quantity_above_estimate_does_not_inflate(self):
        """q > e 이면 추정은 그대로 — 캡이지 승격이 아니다."""
        t = "I want to exchange two laptops for a bigger screen"
        l = _led(); l.accumulate_qty(t)
        self.assertEqual(l.enum_items, E._enum_items(t))
        self.assertEqual(l.qty_mentioned, 2, "수량 채널은 종전대로 살아 있어야 한다")

    def test_large_explicit_quantity_keeps_hint(self):
        """손님이 3건이라 말하고 나열이 5면 3으로 깎이되 **힌트는 유지**되어야 한다."""
        l = _led()
        l.accumulate_qty("I have three items: alpha, beta, gamma, delta, and epsilon")
        self.assertGreaterEqual(l.enum_items, E._ENUM_MIN)
        self.assertTrue(l.multi_entity_hint)

    def test_running_max_still_holds(self):
        l = _led()
        l.accumulate_qty("checking, savings, and credit card")
        before = l.enum_items
        l.accumulate_qty(REAL_004)
        self.assertEqual(l.enum_items, before, "누적 최대가 깎였다 — 캡은 발화 단위여야 한다")


if __name__ == "__main__":
    unittest.main(verbosity=2)
