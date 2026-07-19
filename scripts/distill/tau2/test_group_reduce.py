# -*- coding: utf-8 -*-
"""group_reduce 프리미티브 + 합성 단위테스트 (ACCOUNT_APY_OFFLOAD §4-2·P0).
엔진 산술만 검증(도메인 리터럴 0). 실제 gold 값 재현은 §4-0 census(KB 필요)로 별도."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_compute import apply_op  # noqa: E402


def _c(**kw):
    return dict(kw)


def t_group_reduce_basic():
    # checking boost=max1, card=max1, relationship=sum
    ctx = _c(boosts=[
        {"kind": "checking", "value": 0.75}, {"kind": "checking", "value": 0.10},
        {"kind": "card", "value": 0.50}, {"kind": "card", "value": 0.20},
        {"kind": "relationship", "value": 0.05}, {"kind": "relationship", "value": 0.05},
    ])
    spec = {"op": "group_reduce", "over": "boosts", "group_by": "kind", "value_field": "value",
            "reducers": {"checking": "max1", "card": "max1", "relationship": "sum"}}
    # 0.75(max) + 0.50(max) + 0.10(sum) = 1.35
    r = apply_op(spec, ctx)
    assert abs(r - 1.35) < 1e-9, r
    assert ctx.get("_gr_flags", []) == []
    print("t_group_reduce_basic OK", r)


def t_unknown_kind_flag():
    ctx = _c(boosts=[{"kind": "checking", "value": 0.75}, {"kind": "mystery", "value": 9.99}])
    spec = {"op": "group_reduce", "over": "boosts", "group_by": "kind", "value_field": "value",
            "reducers": {"checking": "max1"}, "unknown_policy": "flag"}
    r = apply_op(spec, ctx)
    assert abs(r - 0.75) < 1e-9, r                       # unknown 미합성
    assert ctx.get("_gr_flags") == ["mystery"], ctx.get("_gr_flags")  # 플래그·silent drop 아님
    print("t_unknown_kind_flag OK", r, ctx["_gr_flags"])


def t_argmax_composed_key():
    # 후보별 base + group_reduce(boosts) 로 argmax (§2a 표현트리)
    cands = [
        {"option": "Green", "base_apy": 3.0, "boosts": [{"kind": "checking", "value": 0.75}]},
        {"option": "SilverPlus", "base_apy": 3.0, "boosts": [
            {"kind": "checking", "value": 0.10}, {"kind": "card", "value": 1.50}]},
    ]
    key = {"op": "sum", "of": ["r.base_apy",
           {"op": "group_reduce", "over": "r.boosts", "group_by": "kind", "value_field": "value",
            "reducers": {"checking": "max1", "card": "max1"}}]}
    spec = {"op": "argmax", "over": "cands", "key": key, "return": "option"}
    r = apply_op(spec, _c(cands=cands))
    # Green=3.75, SilverPlus=3.0+0.10+1.50=4.60 → SilverPlus
    assert r == "SilverPlus", r
    print("t_argmax_composed_key OK", r)


def t_interest_delta_composition():
    # correct effective vs applied → principal * Δapy/100 * days/365
    ctx = _c(principal=8000.0, applied_apy=5.625,
             correct=[{"kind": "base", "value": 3.35}, {"kind": "checking", "value": 3.50}],
             p_start="2025-08-14", p_end="2025-11-14")
    eff = {"op": "group_reduce", "over": "correct", "group_by": "kind", "value_field": "value",
           "reducers": {"base": "sum", "checking": "max1"}}
    delta_apy = {"op": "diff", "a": eff, "b": "applied_apy"}
    days = {"op": "days_between", "a": "p_start", "b": "p_end"}
    amount = {"op": "multiply",
              "a": {"op": "multiply", "a": "principal", "b": {"op": "multiply", "a": delta_apy, "b": 0.01}},
              "b": {"op": "multiply", "a": days, "b": {"op": "const", "value": 1.0 / 365}}}
    eff_v = apply_op(eff, ctx)
    d_v = apply_op(delta_apy, ctx)
    days_v = apply_op(days, ctx)
    amt = apply_op(amount, ctx)
    # eff=6.85, delta=1.225, days=92, amount=8000*0.01225*92/365 ≈ 24.71
    assert abs(eff_v - 6.85) < 1e-9, eff_v
    assert abs(d_v - 1.225) < 1e-9, d_v
    assert days_v == 92, days_v
    assert abs(amt - (8000 * 0.01225 * 92 / 365)) < 1e-6, amt
    print("t_interest_delta_composition OK eff=%.3f delta=%.3f days=%d amt=%.2f" % (eff_v, d_v, days_v, amt))


def t_empty_and_missing():
    assert apply_op({"op": "group_reduce", "over": "nope", "group_by": "k",
                     "value_field": "v", "reducers": {}}, _c()) is None
    ctx = _c(boosts=[{"kind": "checking"}])              # value 없음 → skip
    r = apply_op({"op": "group_reduce", "over": "boosts", "group_by": "kind",
                  "value_field": "value", "reducers": {"checking": "max1"}}, ctx)
    assert r == 0.0, r
    print("t_empty_and_missing OK")


if __name__ == "__main__":
    t_group_reduce_basic()
    t_unknown_kind_flag()
    t_argmax_composed_key()
    t_interest_delta_composition()
    t_empty_and_missing()
    print("ALL PASS")


def t_across_min_023():
    # 023: 월별윈도우 합의 min ≥ threshold (개설 2024-11-15, 3거래/월 예시)
    txns=[
        {"date":"2024-11-20","amount":5000},{"date":"2024-12-01","amount":3000},  # win0: 8000
        {"date":"2024-12-20","amount":4000},{"date":"2025-01-05","amount":4000},  # win1: 8000
        {"date":"2025-01-20","amount":3000},{"date":"2025-02-01","amount":3000},  # win2: 6000 (<7500!)
    ]
    bucketed=apply_op({"op":"bucket_month_window","over":"txns","anchor":"open",
                       "date_field":"date","out_field":"win"},{"txns":txns,"open":"2024-11-15"})
    # min over windows
    mn=apply_op({"op":"group_reduce","over":"b","group_by":"win","value_field":"amount",
                 "default_reducer":"sum","across":"min"},{"b":bucketed})
    assert mn==6000.0, mn                 # win2 미달 → min=6000
    print("t_across_min_023 OK min_monthly=%.0f (qualifies=%s)"%(mn, mn>=7500))
    # 모든 월 충족 케이스
    txns2=[{"date":"2024-11-20","amount":8000},{"date":"2024-12-20","amount":9000},{"date":"2025-01-20","amount":7500}]
    b2=apply_op({"op":"bucket_month_window","over":"t","anchor":"o","date_field":"date","out_field":"win"},{"t":txns2,"o":"2024-11-15"})
    mn2=apply_op({"op":"group_reduce","over":"b","group_by":"win","value_field":"amount","default_reducer":"sum","across":"min"},{"b":b2})
    assert mn2==7500.0 and mn2>=7500, mn2
    print("t_across_min_023 qualify-case OK min=%.0f qualifies=%s"%(mn2,mn2>=7500))


def t_dg_sum_unchanged():
    # D+G APY: across 미지정=sum 유지(회귀 없음)
    ctx={"boosts":[{"kind":"checking","value":0.75},{"kind":"card","value":0.5},{"kind":"relationship","value":0.1}]}
    r=apply_op({"op":"group_reduce","over":"boosts","group_by":"kind","value_field":"value",
                "reducers":{"checking":"max1","card":"max1","relationship":"sum"}},ctx)
    assert abs(r-1.35)<1e-9, r
    print("t_dg_sum_unchanged OK", r)
