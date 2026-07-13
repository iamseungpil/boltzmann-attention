# -*- coding: utf-8 -*-
"""Unit tests for v3.2 CONSISTENCY 가드 (L10 멤버십 t35형 · G-noop t71형).

오프라인·순수 함수만 (membership_violation / noop_write / _record_for / _ids_at_path).
실측값 재현: t35(1684786391 ∈ #W9672333·∉ #W8528674) · t71(#W5782623 주소 == 요청 주소).
Δspurious 가드: 미조회 침묵 · 정상 write 무발화 · min_match.
"""
import sys, os, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("T2_GATE_KINDS", "auth,confirm")

from t2_gate_patch import membership_violation, noop_write, _record_for, _ids_at_path


class Msg:
    def __init__(self, role, content, error=False):
        self.role, self.content, self.error = role, content, error


SPEC = {"entity_key": "order_id", "items_key": "item_ids",
        "items_id_path": ["items", "item_id"], "detail_reader": "get_order_details"}

# t35 실측 형상: 랩탑 1684786391은 #W9672333(pending)에 있고 #W8528674에는 없음
ORD_9672333 = json.dumps({"order_id": "#W9672333", "status": "pending",
                          "items": [{"item_id": "1684786391", "name": "Laptop"}]})
ORD_8528674 = json.dumps({"order_id": "#W8528674", "status": "delivered",
                          "items": [{"item_id": "6704763132", "name": "Bluetooth Speaker"},
                                    {"item_id": "4716977452", "name": "Bluetooth Speaker"}]})
MSGS_T35 = [Msg("tool", ORD_9672333), Msg("tool", ORD_8528674)]


def test_t35_membership_reject_with_hint():
    d = {"order_id": "#W8528674", "item_ids": ["1684786391"],
         "new_item_ids": ["5052031638"], "payment_method_id": "paypal_7664977"}
    mv = membership_violation(d, SPEC, MSGS_T35)
    assert mv is not None, "t35 mis-binding must be flagged"
    bad, oid, hint = mv
    assert bad == ["1684786391"] and oid == "#W8528674" and hint == "#W9672333", mv
    print("PASS test_t35_membership_reject_with_hint")


def test_membership_ok_silent():
    d = {"order_id": "#W9672333", "item_ids": ["1684786391"]}
    assert membership_violation(d, SPEC, MSGS_T35) is None
    print("PASS test_membership_ok_silent")


def test_membership_unfetched_silent():
    d = {"order_id": "#W0000001", "item_ids": ["1684786391"]}   # 상세 미조회 주문
    assert membership_violation(d, SPEC, MSGS_T35) is None, "unfetched container must be silent"
    print("PASS test_membership_unfetched_silent")


# t71 실측 형상: #W5782623 주소가 이미 Charlotte 기본주소
ORD_5782623 = json.dumps({"order_id": "#W5782623", "status": "pending",
                          "address": {"address1": "159 Hickory Lane", "address2": "Suite 995",
                                      "city": "Charlotte", "state": "NC",
                                      "country": "USA", "zip": "28245"},
                          "items": [{"item_id": "2492465580", "name": "Backpack"}]})
MSGS_T71 = [Msg("tool", ORD_5782623)]


def test_t71_noop_reject():
    d = {"order_id": "#W5782623", "address1": "159 Hickory Lane", "address2": "Suite 995",
         "city": "Charlotte", "state": "NC", "country": "USA", "zip": "28245"}
    assert noop_write(d, SPEC, MSGS_T71) is True, "all-equal write must flag as noop"
    print("PASS test_t71_noop_reject")


def test_t41_real_change_silent():
    """t41형 정당 수정(443→445 diff)은 무발화."""
    rec = json.dumps({"order_id": "#W4082615",
                      "address": {"address1": "443 Maple Drive", "address2": "Suite 394",
                                  "city": "Fort Worth", "state": "TX",
                                  "country": "USA", "zip": "76165"}})
    d = {"order_id": "#W4082615", "address1": "445 Maple Drive", "address2": "Suite 394",
         "city": "Fort Worth", "state": "TX", "country": "USA", "zip": "76165"}
    assert noop_write(d, SPEC, [Msg("tool", rec)]) is False
    print("PASS test_t41_real_change_silent")


def test_noop_min_match_guard():
    """매칭 필드 < min_match(희소)면 무발화 — cancel(order_id, reason)류."""
    d = {"order_id": "#W5782623", "reason": "no longer needed"}
    assert noop_write(d, SPEC, MSGS_T71) is False
    print("PASS test_noop_min_match_guard")


def test_ids_at_path_dict_container():
    """variants형 dict-keyed 컨테이너도 지원."""
    rec = {"product_id": "p1", "variants": {"111": {"item_id": "111"}, "222": {"item_id": "222"}}}
    assert _ids_at_path(rec, ["variants", "item_id"]) == {"111", "222"}
    print("PASS test_ids_at_path_dict_container")


def test_record_for_latest_wins():
    old = json.dumps({"order_id": "#W1", "status": "pending", "items": []})
    new = json.dumps({"order_id": "#W1", "status": "delivered", "items": []})
    rec = _record_for([Msg("tool", old), Msg("tool", new)], "order_id", "#W1")
    assert rec["status"] == "delivered", "latest tool output must win"
    print("PASS test_record_for_latest_wins")


if __name__ == "__main__":
    test_t35_membership_reject_with_hint()
    test_membership_ok_silent()
    test_membership_unfetched_silent()
    test_t71_noop_reject()
    test_t41_real_change_silent()
    test_noop_min_match_guard()
    test_ids_at_path_dict_container()
    test_record_for_latest_wins()
    print("ALL PASS (8/8)")
