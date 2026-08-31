# -*- coding: utf-8 -*-
"""회귀 — 정책이 금지한 *쓰지 않을 도구 주기/잠금해제* 계기 (`T2_UNUSED_GRANT`).

★정책 축자(에이전트 시스템 프롬프트에 실린다 · `all_tools.md` → `{{component:additional_instructions}}`):
    "IMPORTANT: Do not unlock tools that you do not plan on giving to the user and actually using:
     this causes issues in database logging."
★왜 (base x644 task_010 실측): gold 액션 **2/2 만점**(action_reward 1.0·1.0)인데 `db_match=false`
  로 reward 0.0. 차이는 `give_discoverable_user_tool(get_referral_link)` 한 번 — 손님은 그것을 쓰지
  않고 `submit_referral` 을 실행했다. env 소스가 그 give 를 DB 행으로 만든다
  (`{"status": "GIVEN"}` → `add_to_db("user_discoverable_tools", …)`).
⚠계기뿐이다 — 아무것도 막지 않는다. 이름은 전부 A2 선언에서 온다(도메인 리터럴 0).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_gate_patch as G

A2 = {"dispatcher_role_check": {"give_tool": "give_discoverable_user_tool",
                                "user_call": "call_discoverable_user_tool",
                                "unlock_tool": "unlock_discoverable_agent_tool",
                                "agent_call": "call_discoverable_agent_tool"},
      "eplan": {"dispatch_name_key": "agent_tool_name"}}


class TC:
    def __init__(self, name, **args):
        self.name, self.arguments = name, args


class M:
    def __init__(self, role, calls=()):
        self.role, self.tool_calls, self.content = role, list(calls), ""


def test_given_and_used_is_clean():
    ms = [M("assistant", [TC("give_discoverable_user_tool", discoverable_tool_name="submit_referral")]),
          M("user", [TC("call_discoverable_user_tool", discoverable_tool_name="submit_referral")])]
    g = G.unused_grants(ms, A2)
    assert g["unused_given"] == [], g


def test_given_but_never_used_is_flagged():
    """base task_010 의 실물 형태 — 준 도구는 get_referral_link, 손님이 쓴 것은 submit_referral."""
    ms = [M("assistant", [TC("give_discoverable_user_tool", discoverable_tool_name="get_referral_link")]),
          M("user", [TC("submit_referral", user_id="76ad9cc60e")])]
    g = G.unused_grants(ms, A2)
    assert g["unused_given"] == ["get_referral_link"], g
    assert g["given"] == ["get_referral_link"]


def test_unlocked_but_never_called_is_flagged():
    ms = [M("assistant", [TC("unlock_discoverable_agent_tool", agent_tool_name="get_referral_link_991")])]
    g = G.unused_grants(ms, A2)
    assert g["unused_unlocked"] == ["get_referral_link"], "접미사를 떼고 짝지어야 한다"


def test_unlock_then_call_is_clean():
    ms = [M("assistant", [TC("unlock_discoverable_agent_tool", agent_tool_name="get_x_3847")]),
          M("assistant", [TC("call_discoverable_agent_tool", agent_tool_name="get_x_3847")])]
    assert G.unused_grants(ms, A2)["unused_unlocked"] == []


def test_direct_user_tool_counts_as_used():
    """손님이 디스패처를 안 거치고 직접 부른 것도 실행이다(실물 궤적의 두 형태)."""
    ms = [M("assistant", [TC("give_discoverable_user_tool", discoverable_tool_name="submit_referral")]),
          M("user", [TC("submit_referral", user_id="x")])]
    assert G.unused_grants(ms, A2)["unused_given"] == []


def test_no_declaration_no_crash():
    assert G.unused_grants([M("assistant", [TC("give_discoverable_user_tool", x="y")])], {}) == {
        "given": [], "unlocked": [], "unused_given": [], "unused_unlocked": []}


def test_instrument_is_wired_and_records_only():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2_gate_patch.py"),
               encoding="utf-8").read()
    assert 'T2_UNUSED_GRANT' in src and '_t2_ug_sig' in src
    seg = src.split("[T2_UNUSED_GRANT] 계기")[1][:1200]
    for forbidden in ("return None", "deny", "fb.append"):
        assert forbidden not in seg, "계기가 거동을 바꾸면 안 된다: %s" % forbidden


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
    print("ALL PASS")
