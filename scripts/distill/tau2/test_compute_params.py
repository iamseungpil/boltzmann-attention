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
    """doc_036/031: liability = min(disputed_amount, tier_cap). tier=tx→disc(§8-2 proxy).
    ≤30일(proxy)→$50 · ≤60일→$500 · 이후→전액. §8-2 재현 89.4%(proxy)."""
    spec = CO["file_debit_card_transaction_dispute"]["customer_max_liability_amount"]
    # tx→disc 1일(≤30)·amount>50 → min(300,50)=50
    assert run(spec, {"transaction_date": "11/13/2025", "discovery_date": "11/14/2025", "disputed_amount": 300}) == 50
    # tx→disc 41일(≤60)·amount 625 → min(625,500)=500
    assert run(spec, {"transaction_date": "10/04/2025", "discovery_date": "11/14/2025", "disputed_amount": 625}) == 500
    # tx→disc 40일(≤60)·amount 412.88 → min(412.88,500)=412.88
    assert run(spec, {"transaction_date": "10/05/2025", "discovery_date": "11/14/2025", "disputed_amount": 412.88}) == 412.88
    # tx→disc >60일 → 전액
    assert run(spec, {"transaction_date": "01/01/2025", "discovery_date": "11/14/2025", "disputed_amount": 300}) == 300
    assert run(spec, {"disputed_amount": 300}) is None            # 날짜없음→abstain


def test_bool_expr_engine():
    """bool_expr 엔진 op 3값논리(A2 무관·§8-3서 provisional A2스펙은 net−4로 드롭·op는 일반유지)."""
    spec = {"op": "bool_expr", "all": [
        {"ref": "params.a", "eq": True},
        {"expr": {"op": "days_between", "a": "params.d1", "b": "params.d2"}, "<=": 60},
        {"ref": "params.c", "in": ["x", "y"]}]}
    assert run(spec, {"a": True, "d1": "11/01/2025", "d2": "11/10/2025", "c": "x"}) is True
    assert run(spec, {"a": False, "d1": "11/01/2025", "d2": "11/10/2025", "c": "x"}) is False
    assert run(spec, {"a": True, "d1": "01/01/2025", "d2": "11/10/2025", "c": "x"}) is False  # >60d
    assert run(spec, {"a": True, "c": "x"}) is None                                           # 날짜없음→abstain


class _Obj:
    def __init__(self, **k): self.__dict__.update(k)


def test_resolve_compute_params():
    """배선 스텁(§6·resolve_compute_params): dispute 호출의 틀린 liability를 결정론 compute로 silent-repair 감지."""
    import t2_resolve as _rz, json as _j
    nested = {"transaction_date": "11/13/2025", "discovery_date": "11/14/2025",
              "disputed_amount": 300, "customer_max_liability_amount": 100}   # agent=100(틀림)·gold=50
    tc = _Obj(name="call_discoverable_agent_tool",
              arguments={"agent_tool_name": "file_debit_card_transaction_dispute_6281",
                         "arguments": _j.dumps(nested)})
    am = _Obj(tool_calls=[tc])
    msgs = [_Obj(content="The current time is 2025-11-14 03:40:00 EST.")]
    reps = _rz.resolve_compute_params(am, msgs, A2)
    assert len(reps) == 1, reps
    assert reps[0]["param"] == "customer_max_liability_amount"
    assert str(reps[0]["computed"]) == "50" and str(reps[0]["old"]) == "100"
    # 에이전트가 이미 맞춘(50) 경우 → 미개입(repair 없음·Δspurious 회피)
    nested2 = dict(nested); nested2["customer_max_liability_amount"] = 50
    tc2 = _Obj(name="call_discoverable_agent_tool",
               arguments={"agent_tool_name": "file_debit_card_transaction_dispute_6281", "arguments": _j.dumps(nested2)})
    assert _rz.resolve_compute_params(_Obj(tool_calls=[tc2]), msgs, A2) == []


def test_all():
    n = 0
    for fn in (test_liability, test_bool_expr_engine, test_resolve_compute_params):
        fn(); n += 1
        print("  PASS %s" % fn.__name__)
    print("compute_ops gold-blind + 배선 유닛 %d/%d PASS" % (n, 3))


if __name__ == "__main__":
    test_all()
