# -*- coding: utf-8 -*-
"""action-vs-advise = operator 해소 GET→FIND→(execute|ASK) unit (사용자 2026-07-13).
banking apply_for_credit_card 회피(task_001/003/007 형)·retail no-write 동형.
결정론 분기 검증(FIND=target은 주입·learn은 별도). transfer_tools=A2 도출."""
import sys, os, json, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("T2_GATE_KINDS", "auth,confirm")
import t2_resolve as R


class TC:
    def __init__(self, name): self.name, self.id, self.requestor = name, name, "assistant"
class AM:
    def __init__(self, tool_calls=None, content=None):
        self.role, self.tool_calls, self.content = "assistant", tool_calls, content

BANK = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
XFER = {"transfer_to_human_agents"}
OPS = {"action_tools": BANK["action_tools"]}


def test_deflect_advice_deny():
    """gold=apply인데 조언(도구호출 0)으로 종결 → action-required deny (task_001형)."""
    am = AM(content="To apply, visit our website and navigate to the credit card section...")
    r = R.resolve_action_operator(OPS, am, [], BANK, target_tool="apply_for_credit_card",
                                  transfer_tools=XFER)
    assert r["status"] == "deny" and r["reason"] == "action-required", r
    assert "apply_for_credit_card" in r["feedback"]
    print("PASS deflect_advice_deny (조언 회피→실행 강제)")


def test_transfer_giveup_deny():
    """target 있는데 transfer로 포기 → action-required (task_017형)."""
    am = AM(tool_calls=[TC("transfer_to_human_agents")])
    r = R.resolve_action_operator(OPS, am, [], BANK, target_tool="apply_for_credit_card",
                                  transfer_tools=XFER)
    assert r["status"] == "deny" and r["reason"] == "action-required", r
    print("PASS transfer_giveup_deny")


def test_no_target_ask():
    """회피이나 target 미해소 → ASK(조언/날조 대신 개방질문)."""
    am = AM(content="I'm not sure, maybe check the website.")
    r = R.resolve_action_operator(OPS, am, [], BANK, target_tool=None, transfer_tools=XFER)
    assert r["status"] == "deny" and r["reason"] == "action-ask", r
    print("PASS no_target_ask")


def test_calling_action_ok():
    """이미 action_tool 호출 중 = ok(정상 행동)."""
    am = AM(tool_calls=[TC("apply_for_credit_card")])
    r = R.resolve_action_operator(OPS, am, [], BANK, target_tool="apply_for_credit_card",
                                  transfer_tools=XFER)
    assert r["status"] == "ok", r
    print("PASS calling_action_ok")


def test_doing_other_work_ok():
    """조회 등 다른 도구 호출 중 = 진행중 ok(회피 아님·Δspurious)."""
    am = AM(tool_calls=[TC("get_credit_card_accounts_by_user")])
    r = R.resolve_action_operator(OPS, am, [], BANK, target_tool="apply_for_credit_card",
                                  transfer_tools=XFER)
    assert r["status"] == "ok", r
    print("PASS doing_other_work_ok (Δspurious)")


def test_no_action_tools_noop():
    r = R.resolve_action_operator({}, AM(content="hi"), [], {}, target_tool=None, transfer_tools=XFER)
    assert r["status"] == "ok"
    print("PASS no_action_tools_noop")


def test_agent_ending_helper():
    assert R._agent_ending(AM(content="advice"), XFER) is True          # 조언
    assert R._agent_ending(AM(tool_calls=[TC("transfer_to_human_agents")]), XFER) is True  # transfer
    assert R._agent_ending(AM(tool_calls=[TC("get_x")]), XFER) is False # 조회=진행
    print("PASS agent_ending_helper")


if __name__ == "__main__":
    test_deflect_advice_deny()
    test_transfer_giveup_deny()
    test_no_target_ask()
    test_calling_action_ok()
    test_doing_other_work_ok()
    test_no_action_tools_noop()
    test_agent_ending_helper()
    print("ALL PASS (7/7) - action=operator GET->FIND->(execute|ASK)")
