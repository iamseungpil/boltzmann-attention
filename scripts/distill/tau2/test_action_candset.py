# -*- coding: utf-8 -*-
"""회귀 — **행동 선택과 호출 형식을 가른다** (`T2_ACTION_CANDSET`).

★결손 (2026-08-31 · x709 task_010 · 사용자 지적): A2 `action_tools` 가 행동과 호출형을 한 통에
  담고 있었다. 8개에서 고르라고 물으니 판단 프로브가 **21/21 회 `call_discoverable_agent_tool`**
  — 즉 *형식*을 답으로 냈다(사이드카 전수). gold 는 손님의 `submit_referral` 이었고, 그 오답이
  캐시로 굳어 `T2_ACTIONREQ` 가 밀 곳을 잃었다. 격리 x711(n=3): 현행 집합 → 형식 3/3.
★분담: 선택은 모델, **형식은 엔진**([[10]]). 형식은 선언에서 결정론으로 나온다 —
  자기 도구=직접 · agent-discoverable=unlock→call · user 도구=give→손님 실행.
⚠집합 차집합 하나뿐 — 의도 분류 0([[66]]) · 도메인 리터럴 0([[05]]).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_resolve as R

A2 = {"dispatcher_role_check": {"unlock_tool": "unlock_discoverable_agent_tool",
                                "agent_call": "call_discoverable_agent_tool",
                                "give_tool": "give_discoverable_user_tool",
                                "user_call": "call_discoverable_user_tool"},
      "eplan": {"dispatch_tool": "call_discoverable_agent_tool",
                "unlock_tool": "unlock_discoverable_agent_tool"}}
LIVE = ["apply_for_credit_card", "submit_referral", "change_user_email", "submit_transaction",
        "unlock_discoverable_agent_tool", "call_discoverable_agent_tool",
        "give_discoverable_user_tool", "call_discoverable_user_tool"]


def setup_function(_=None):
    os.environ["T2_ACTION_CANDSET"] = "1"
    R._T2_CANDSET_SAID = False


def test_call_forms_are_declared_not_guessed():
    assert R.call_form_tools(A2) == {"unlock_discoverable_agent_tool",
                                     "call_discoverable_agent_tool",
                                     "give_discoverable_user_tool",
                                     "call_discoverable_user_tool"}
    assert R.call_form_tools({}) == set()          # 선언이 없으면 아무것도 호출형이 아니다


def test_live_set_drops_exactly_the_four_call_forms():
    setup_function()
    got = R.action_candidates(LIVE, A2)
    assert got == {"apply_for_credit_card", "submit_referral",
                   "change_user_email", "submit_transaction"}, got


def test_no_declaration_keeps_everything():
    setup_function()
    assert R.action_candidates(LIVE, {}) == set(LIVE)


def test_flag_off_restores_old_set():
    os.environ["T2_ACTION_CANDSET"] = "0"
    try:
        assert R.action_candidates(LIVE, A2) == set(LIVE)
    finally:
        os.environ["T2_ACTION_CANDSET"] = "1"


def test_all_call_forms_falls_back_rather_than_emptying():
    """후보가 전부 호출형이면 빈 집합으로 프로브를 죽이지 않는다."""
    setup_function()
    only = ["call_discoverable_agent_tool", "give_discoverable_user_tool"]
    assert R.action_candidates(only, A2) == set(only)


def test_probe_filters_before_asking():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2_resolve.py"),
               encoding="utf-8").read()
    seg = src.split("def formalize_intent_tool(")[1][:2500]
    assert "action_candidates(action_tools" in seg, "프로브가 거르지 않고 묻는다"


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            setup_function(); f(); print("ok", n)
    print("ALL PASS")
