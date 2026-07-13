# -*- coding: utf-8 -*-
"""통일 인터프리터 오프라인 실험 (UNIFIED_OPERAND_A2 §7-4·리뷰 U1 작동적-전이 시작점).

두 주장 분리(U1):
  (a) 서술적: 루프가 실패를 라벨링 — trivial, 여기서 안 다룸.
  (b) 작동적: 같은 디스패처(resolve_operand)가 (i) 개별레버 결정을 *재현*(동치) +
      (ii) retail·banking A2-스왑으로 *같은 코드*가 두 도메인 해소 = 전이. ← 이 테스트.
U3: operator=operand는 discoverable서만·direct-dispatch(retail)는 operator no-op.
"""
import sys, os, json, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("T2_GATE_KINDS", "auth,confirm")

import t2_resolve as R
import t2_gate_patch as G


class Msg:
    def __init__(self, role, content=None, error=False, id=None, tool_calls=None):
        self.role, self.content, self.error = role, content, error
        self.id, self.tool_calls = id, tool_calls


RETAIL = json.load(open(os.path.join(HERE, "a2", "retail.gate.json"), encoding="utf-8"))
BANK = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))

ORD_9672333 = json.dumps({"order_id": "#W9672333", "status": "pending",
                          "items": [{"item_id": "1684786391", "name": "Laptop"}]})
ORD_8528674 = json.dumps({"order_id": "#W8528674", "status": "delivered",
                          "items": [{"item_id": "6704763132", "name": "Speaker"}]})


# ── (i) 작동적 동치: 디스패처가 개별레버 결정을 재현 ──
def test_equiv_membership():
    """resolve_operand(membership) == membership_violation (t35 오바인딩)."""
    msgs = [Msg("tool", ORD_9672333), Msg("tool", ORD_8528674)]
    d = {"order_id": "#W8528674", "item_ids": ["1684786391"]}
    opspec = RETAIL["operands"]["modify_pending_order_items"]["item_ids"]
    r = R.resolve_operand(opspec, "modify_pending_order_items", "item_ids", d, msgs, RETAIL)
    assert r["status"] == "deny" and r["reason"] == "membership", r
    # 개별레버 직접 호출과 동일 판정
    assert G.membership_violation(d, {"entity_key": "order_id", "items_key": "item_ids",
                                      "items_id_path": ["items", "item_id"]}, msgs) is not None
    print("PASS equiv_membership (디스패처==L10)")


def test_equiv_provenance():
    """resolve_operand(provenance) == origin (t97 확인-세탁)."""
    msgs = [Msg("user", "use my nyc address"),
            Msg("tool", '{"orders":["#W6750959","#W3407479"]}'),
            Msg("assistant", "The address will be: 123 Broadway, New York, NY 10007"),
            Msg("user", "yes 123 Broadway is correct")]
    d = {"order_id": "#W6750959", "address1": "123 Broadway"}
    opspec = RETAIL["operands"]["modify_pending_order_address"]["address1"]
    r = R.resolve_operand(opspec, "modify_pending_order_address", "address1", d, msgs, RETAIL)
    assert r["status"] == "deny" and r["reason"] == "provenance", r
    print("PASS equiv_provenance (디스패처==L3)")


def test_provenance_user_first_ok():
    """user-first 주소 = allow (t43·Δspurious 무오차단)."""
    msgs = [Msg("user", "ship to 1234 S Michigan Ave")]
    d = {"order_id": "#W1", "address1": "1234 S Michigan Ave"}
    opspec = RETAIL["operands"]["modify_pending_order_address"]["address1"]
    r = R.resolve_operand(opspec, "modify_pending_order_address", "address1", d, msgs, RETAIL)
    assert r["status"] == "ok", r
    print("PASS provenance_user_first_ok (Δspurious)")


# ── (ii) operator 해소 (banking·§8b 도구명 날조 35.9% 타깃) ──
def test_operator_fabrication_deny():
    """발견 안 된 도구명 사용 = deny (banking 도구명 날조)."""
    disco = json.dumps({"available_tools": ["update_transaction_rewards_3847",
                                            "file_credit_card_transaction_dispute_4829"]})
    msgs = [Msg("user", "fix my rewards"), Msg("tool", disco)]
    d = {"agent_tool_name": "apply_magic_bonus_9999"}   # 발명(미발견)
    opspec = BANK["operands"]["unlock_discoverable_agent_tool"]["agent_tool_name"]
    r = R.resolve_operand(opspec, "unlock_discoverable_agent_tool", "agent_tool_name", d, msgs, BANK)
    assert r["status"] == "deny" and r["reason"] == "operator-fab", r
    print("PASS operator_fabrication_deny (banking 도구명 날조)")


def test_operator_grounded_ok():
    """발견된 도구명 사용 = allow."""
    disco = json.dumps({"available_tools": ["update_transaction_rewards_3847"]})
    msgs = [Msg("tool", disco)]
    d = {"agent_tool_name": "update_transaction_rewards_3847"}
    opspec = BANK["operands"]["unlock_discoverable_agent_tool"]["agent_tool_name"]
    r = R.resolve_operand(opspec, "unlock_discoverable_agent_tool", "agent_tool_name", d, msgs, BANK)
    assert r["status"] == "ok", r
    print("PASS operator_grounded_ok")


def test_u3_direct_dispatch_noop():
    """U3: operator_resolution != discoverable = no-op(retail direct-dispatch)."""
    opspec = {"kind": "operator", "arg": "agent_tool_name", "name_pattern": "[a-z_]+_[0-9]{4}"}
    d = {"agent_tool_name": "whatever_0000"}
    r = R.resolve_operand(opspec, "x", "agent_tool_name", d, [], {})
    assert r["status"] == "ok", "direct-dispatch operator는 no-op이어야(U3)"
    print("PASS u3_direct_dispatch_noop")


# ── (ii-b) ★전이: 같은 resolve_write가 retail·banking 둘 다 실행 ──
def test_transfer_same_dispatcher():
    """엔진 무수정·A2 스왑만: resolve_write가 retail(membership)·banking(operator) 둘 다 처리."""
    # retail: 오바인딩 deny
    rmsgs = [Msg("tool", ORD_9672333), Msg("tool", ORD_8528674)]
    rr = R.resolve_write("modify_pending_order_items",
                         {"order_id": "#W8528674", "item_ids": ["1684786391"]}, rmsgs, RETAIL)
    assert rr["status"] == "deny" and rr["arg"] == "item_ids", rr
    # banking: 도구명 날조 deny — 같은 함수·다른 A2
    bmsgs = [Msg("tool", json.dumps({"available_tools": ["update_transaction_rewards_3847"]}))]
    br = R.resolve_write("unlock_discoverable_agent_tool",
                         {"agent_tool_name": "fake_tool_0000"}, bmsgs, BANK)
    assert br["status"] == "deny" and br["arg"] == "agent_tool_name", br
    # spec 없는 도메인 = 우아한 강등
    er = R.resolve_write("some_tool", {"x": 1}, [], {})
    assert er["status"] == "ok"
    print("PASS transfer_same_dispatcher (★엔진 무수정·retail+banking A2-스왑·[[11]])")


if __name__ == "__main__":
    test_equiv_membership()
    test_equiv_provenance()
    test_provenance_user_first_ok()
    test_operator_fabrication_deny()
    test_operator_grounded_ok()
    test_u3_direct_dispatch_noop()
    test_transfer_same_dispatcher()
    print("ALL PASS (7/7) - 통일 디스패처 동치+operator+전이 오프라인")
