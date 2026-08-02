#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P12 회귀: axis 노트가 **mutating 도구 출력에 붙지 않는지** (041 R0 사고·2026-08-02).

사고: `call_discoverable_agent_tool`(mutating) 출력에 `[axis]` 노트가 붙어 tau2 평가 replay
(`environment.py` mutating 재실행 후 content 비교)가 불일치 → ValueError → sim 전체 재시도
(R0 6,579s 폐기). 원인: 가드가 `T2_SURFACE_BUS=1`일 때만 걸렸고 라이브(버스 OFF)는 무가드.

이 테스트는 순수 술어 계약만 검증한다(엔진 import 불필요):
  ⑴ mutating 도구 → 부착 금지  ⑵ 읽기 도구 → 부착 허용  ⑶ env 판정 불가 → 안전측(부착 금지)
`_dedup_cache_safe`는 env `_is_mutating_tool`에 위임하므로 그 계약을 스텁으로 재현한다.

Run: python test_p12_axis_replay.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class _FakeEnv:
    def __init__(self, mutating):
        self._m = set(mutating)

    def _is_mutating_tool(self, name):
        return name in self._m


class _FakeEnvBroken:
    def _is_mutating_tool(self, name):
        raise RuntimeError("env 판정 불가")


class _Orch:
    def __init__(self, env):
        self.environment = env


def main():
    from t2_gate_patch import _dedup_cache_safe

    MUT = ["call_discoverable_agent_tool", "give_discoverable_user_tool",
           "unlock_discoverable_agent_tool", "log_verification"]
    READ = ["get_credit_card_transactions_by_user", "check_card_application_fit",
            "KB_search", "get_reward_discrepancies"]

    orch = _Orch(_FakeEnv(MUT))
    fails = []

    # ⑴ mutating → 부착 금지(=_dedup_cache_safe False)
    for t in MUT:
        if _dedup_cache_safe(orch, t):
            fails.append("mutating인데 부착 허용: %s" % t)

    # ⑵ 읽기 → 부착 허용
    for t in READ:
        if not _dedup_cache_safe(orch, t):
            fails.append("읽기인데 부착 금지: %s" % t)

    # ⑶ env 예외/부재 → 안전측(금지)
    if _dedup_cache_safe(_Orch(_FakeEnvBroken()), "anything"):
        fails.append("env 예외인데 부착 허용(안전측 위반)")
    if _dedup_cache_safe(_Orch(None), "anything"):
        fails.append("env 부재인데 부착 허용(안전측 위반)")

    # ⑷ 041 실사고 재현: 사고 도구가 반드시 차단돼야 한다
    if _dedup_cache_safe(orch, "call_discoverable_agent_tool"):
        fails.append("041 사고 도구가 여전히 부착 허용")

    print("P12 회귀 — mutating %d종 / 읽기 %d종 / 예외 2종" % (len(MUT), len(READ)))
    if fails:
        for f in fails:
            print("  ❌ %s" % f)
        print("FAIL %d건" % len(fails))
        return 1
    print("  ✅ ALL PASS — mutating 부착 차단·읽기 허용·판정불가 안전측")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
