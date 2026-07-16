# -*- coding: utf-8 -*-
"""TOOLGATE requestor 격리 회귀 테스트 (버그: user gold 액션 차단 → task 통과불가).

검증:
  (1) requestor="user" 호출은 **가로채지 않는다** (원본 실행 경로로 나간다)
      — task_019 gold 4/6 = requestor:user `call_discoverable_user_tool`
  (2) requestor="assistant"의 미지 도구는 여전히 ASK로 간다 (기존 동작 보존)
  (3) 우리 A2 도구는 assistant 호출에만 결정론 계산으로 응답
tau2 의존 없이 exec2의 분기 로직만 격리 재현(엔진 코드와 동일 조건식).
사용: py -3 test_toolgate_requestor.py
"""
import sys


class TC:
    def __init__(self, name, requestor="assistant", tid="1"):
        self.name, self.requestor, self.id = name, requestor, tid
        self.arguments = {}


def route(tc, decls, known):
    """t2_scaffold_get.exec2의 분기와 동일 조건 (requestor 격리 → ours → TOOLGATE → rest)."""
    if getattr(tc, "requestor", "assistant") != "assistant":
        return "rest(원본 실행)"
    if tc.name in decls:
        return "ours(결정론 계산)"
    if tc.name not in known:
        return "TOOLGATE(ASK)"
    return "rest(원본 실행)"


DECLS = {"get_reward_discrepancies", "verify_identity"}
KNOWN = {"get_reward_discrepancies", "verify_identity", "log_verification",
         "give_discoverable_user_tool", "call_discoverable_agent_tool"}

CASES = [
    # (설명, tool, requestor, 기대)
    ("★gold 019_2~5: 사용자가 dispute 제출", "call_discoverable_user_tool", "user", "rest(원본 실행)"),
    ("사용자의 임의 도구", "apply_for_credit_card", "user", "rest(원본 실행)"),
    ("gold 019_1: 에이전트가 도구 제공", "give_discoverable_user_tool", "assistant", "rest(원본 실행)"),
    ("에이전트의 진짜 날조", "get_user_information_by_phone_number", "assistant", "TOOLGATE(ASK)"),
    ("우리 A2 도구", "get_reward_discrepancies", "assistant", "ours(결정론 계산)"),
]

fail = 0
for desc, name, req, want in CASES:
    got = route(TC(name, req), DECLS, KNOWN)
    ok = got == want
    fail += (not ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {desc:38s} {name:36s} requestor={req:9s} -> {got}")

print()
print("ALL PASS" if not fail else f"{fail} FAILED")
sys.exit(1 if fail else 0)
