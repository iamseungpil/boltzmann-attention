# -*- coding: utf-8 -*-
"""Unit tests for L4-tie (v3.2·t64·A1_V3_PROBE_FORENSIC §1).

t64 실측값 재현: predicate(방수∧avail∧가격≤502.28) 동률 {6700049080($466.75), 6117189161($481.50)}
→ tie_break(min_price) = 6700049080 = gold. 보수 가드: 미선언·가격결손·단일은 None.
"""
import sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from t2_formalize_exec import tie_break, _floor_ok

# t64 실측 variant 값 (genv3_probe idx19 tool 출력서)
RECORDS = [
    ("6700049080", {"item_id": "6700049080", "price": 466.75,
                    "options": {"resolution": "4K", "waterproof": "yes"}, "available": True}),
    ("6117189161", {"item_id": "6117189161", "price": 481.50,
                    "options": {"resolution": "4K", "waterproof": "yes"}, "available": True}),
    ("5925362855", {"item_id": "5925362855", "price": 503.51,
                    "options": {"resolution": "1080p", "waterproof": "yes"}, "available": True}),
]
TIE = ["6700049080", "6117189161"]


def test_t64_tie_min_price():
    assert tie_break(TIE, RECORDS, "min_price") == "6700049080", "gold = cheapest of tie"
    print("PASS test_t64_tie_min_price")


def test_undeclared_mode_none():
    assert tie_break(TIE, RECORDS, None) is None
    assert tie_break(TIE, RECORDS, "max_price") is None, "MENU 밖 모드 = 미지원(온톨로지)"
    print("PASS test_undeclared_mode_none")


def test_single_id_none():
    assert tie_break(["6700049080"], RECORDS, "min_price") is None
    print("PASS test_single_id_none")


def test_missing_price_none():
    recs = [("a1", {"item_id": "a1", "options": {}}),
            ("a2", {"item_id": "a2", "price": 10.0, "options": {}})]
    assert tie_break(["a1", "a2"], recs, "min_price") is None, "가격 결손 후보 = 보수 None"
    print("PASS test_missing_price_none")


def test_floor_keep_precedes_tie():
    """cur가 동률 집합 안이면 keep(정답 미파괴)이 tie보다 먼저 — t64서 모델이 481.50을
    골랐어도 predicate-만족이라 keep(치환 없음). tie는 many(cur∉집합)에서만."""
    fg = _floor_ok({"ids": TIE, "why": "tie"}, "6117189161")
    assert fg["status"] == "keep", fg
    fg2 = _floor_ok({"ids": TIE, "why": "tie"}, "9999999999")
    assert fg2["status"] == "many", fg2
    print("PASS test_floor_keep_precedes_tie")


if __name__ == "__main__":
    test_t64_tie_min_price()
    test_undeclared_mode_none()
    test_single_id_none()
    test_missing_price_none()
    test_floor_keep_precedes_tie()
    print("ALL PASS (5/5)")
