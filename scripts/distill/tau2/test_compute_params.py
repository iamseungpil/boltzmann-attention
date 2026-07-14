# -*- coding: utf-8 -*-
"""compute_ops gold-blind 유닛 (2026-07-14·§8-1). A2 저작 op-스펙이 *정책대로* 계산하는지 검증.
★gold-blind: 기대값은 정책문서(doc_036/031/032)서 유도·궤적 gold action 값 안 봄([[11]] 순환방지).
재현율(gold 대조)은 §8-2 별도(저작 검증 아님)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_compute import apply_op

A2 = json.load(open(os.path.join(os.path.dirname(__file__), "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
CO = A2["compute_ops"]


def run(spec, params, now="11/14/2025"):
    return apply_op(spec, {"params": params, "now": now})


def test_liability():
    """doc_036/031: ≤2 business days→$50 · ≤60 days→$500 · 이후→전액(disputed_amount)."""
    spec = CO["file_debit_card_transaction_dispute"]["customer_max_liability_amount"]
    assert run(spec, {"issue_noticed_date": "11/13/2025", "disputed_amount": 300}) == 50
    assert run(spec, {"issue_noticed_date": "11/10/2025", "disputed_amount": 300}) == 500
    assert run(spec, {"issue_noticed_date": "08/01/2025", "disputed_amount": 300}) == 300
    assert run(spec, {"disputed_amount": 300}) is None            # 날짜없음→abstain


def test_provisional_debit():
    """doc_032: ALL(timely ≤60d ∧ category∈{5종} ∧ written_statement)."""
    spec = CO["file_debit_card_transaction_dispute"]["provisional_credit_eligible"]
    base = {"issue_noticed_date": "11/10/2025", "dispute_category": "unauthorized_transaction", "written_statement_provided": True}
    assert run(spec, base) is True
    assert run(spec, {**base, "dispute_category": "goods_services_not_received"}) is False  # category 밖
    assert run(spec, {**base, "written_statement_provided": False}) is False                # written no
    assert run(spec, {**base, "issue_noticed_date": "01/01/2025"}) is False                 # 늦음(>60d)
    assert run(spec, {"dispute_category": "duplicate_charge", "written_statement_provided": True}) is None  # 날짜없음→abstain


def test_all():
    n = 0
    for fn in (test_liability, test_provisional_debit):
        fn(); n += 1
        print("  PASS %s" % fn.__name__)
    print("compute_ops gold-blind 유닛 %d/%d PASS" % (n, 2))


if __name__ == "__main__":
    test_all()
