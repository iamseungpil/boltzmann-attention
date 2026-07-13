# -*- coding: utf-8 -*-
"""Unit tests for TOOLERR 도구-에러 라우팅 (T2_TOOLERR·사용자 지시 2026-07-13).

순수 함수: classify_tool_error / _trailing_tool_errors / _transfer_tools.
로직=일반(엔진)·분류정보=A2. retail 실측 에러("not found or available" 등) 재현.
"""
import sys, os, json, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("T2_GATE_KINDS", "auth,confirm")

from t2_gate_patch import classify_tool_error, _trailing_tool_errors, _transfer_tools


class Msg:
    def __init__(self, role, content=None, error=False, id=None, tool_calls=None):
        self.role, self.content, self.error = role, content, error
        self.id, self.tool_calls = id, tool_calls


class TC:
    def __init__(self, name, arguments, id):
        self.name, self.arguments, self.id, self.requestor = name, arguments, id, "assistant"


A2 = json.load(open(os.path.join(HERE, "a2", "retail.gate.json"), encoding="utf-8"))


def _retail_recover_msgs():
    # t52형: 틀린 변형 id로 exchange → "not found or available" 에러
    tc = TC("exchange_delivered_order_items",
            {"order_id": "#W1", "item_ids": ["a"], "new_item_ids": ["7195021808"]}, "c1")
    return [Msg("user", "exchange it"),
            Msg("assistant", None, tool_calls=[tc]),
            Msg("tool", "Error: New item 7195021808 not found or available", error=True, id="c1")]


def test_classify_recover():
    r = classify_tool_error(_retail_recover_msgs(), A2)
    assert r is not None, "retail variant-not-found must classify"
    sp, tool, fargs = r
    assert sp["class"] == "recover" and tool == "exchange_delivered_order_items", (sp, tool)
    assert fargs.get("new_item_ids") == ["7195021808"], fargs
    print("PASS test_classify_recover")


def test_trailing_only():
    # 에러 뒤에 user 발화가 오면 = 이미 넘어감 → 미분류
    msgs = _retail_recover_msgs() + [Msg("user", "ok never mind, do something else")]
    assert classify_tool_error(msgs, A2) is None, "user 발화가 최신이면 미발화"
    print("PASS test_trailing_only")


def test_no_error_none():
    tc = TC("get_order_details", {"order_id": "#W1"}, "c1")
    msgs = [Msg("assistant", None, tool_calls=[tc]),
            Msg("tool", json.dumps({"order_id": "#W1", "status": "pending"}), error=False, id="c1")]
    assert classify_tool_error(msgs, A2) is None, "성공 결과 = 미발화"
    print("PASS test_no_error_none")


def test_unmatched_pattern_none():
    tc = TC("cancel_pending_order", {"order_id": "#W1"}, "c1")
    msgs = [Msg("assistant", None, tool_calls=[tc]),
            Msg("tool", "Error: some unrelated failure xyz", error=True, id="c1")]
    assert classify_tool_error(msgs, A2) is None, "A2 패턴 미매칭 = 침묵(안전)"
    print("PASS test_unmatched_pattern_none")


def test_transfer_tools_from_notice_gate():
    # banking A2: notice 게이트 applies_to = transfer 도구 도출
    bank = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
    tt = _transfer_tools(bank)
    assert "transfer_to_human_agents" in tt, tt
    print("PASS test_transfer_tools_from_notice_gate (A2-도출·엔진 리터럴 0)")


def test_transfer_tools_empty_a2():
    assert _transfer_tools({}) == set() and _transfer_tools(None) == set()
    print("PASS test_transfer_tools_empty_a2")


def test_abstain_class_synthetic():
    # abstain spec를 가진 합성 A2(예: KB 검색 없음)
    a2 = {"tool_error_specs": [
        {"applies_to": ["KB_search"], "match": "no information|not available",
         "class": "abstain", "hint": "The knowledge base has no entry."}]}
    tc = TC("KB_search", {"query": "travel card"}, "c1")
    msgs = [Msg("assistant", None, tool_calls=[tc]),
            Msg("tool", "Error: no information found", error=True, id="c1")]
    r = classify_tool_error(msgs, a2)
    assert r and r[0]["class"] == "abstain", r
    print("PASS test_abstain_class_synthetic")


if __name__ == "__main__":
    test_classify_recover()
    test_trailing_only()
    test_no_error_none()
    test_unmatched_pattern_none()
    test_transfer_tools_from_notice_gate()
    test_transfer_tools_empty_a2()
    test_abstain_class_synthetic()
    print("ALL PASS (7/7)")
