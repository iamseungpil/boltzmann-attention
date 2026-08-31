# -*- coding: utf-8 -*-
"""★§T-1a 호출자 우선 — 격리 서브는 **비어 있는 operand 만** 채운다.

근거(093 nightA 실측): 메인이 세 번 다 같은 성분(base 4.0 · checking 0.25 · relationship 0.025
= gold 의 **4.275**)을 보냈는데 `fetch_formalize` 서브가 회차마다 다른 행을 물어와
**8.975 → 2.775 → 8.975** 가 나왔고, 에이전트가 *"tool malfunction"* 으로 판정해 human 이관 →
gold 변이 2행 MISSING ⇒ **1.0 → 0.0**. 정본 계약은 *메인이 formalize · 엔진은 계산*([[10]]/[[52]]).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_scaffold_get as SG

CALLER = [{"kind": "base", "value": 4.0}, {"kind": "checking", "value": 0.25},
          {"kind": "relationship", "value": 0.025}]
SUB = [{"kind": "base", "value": 4.0}, {"kind": "base", "value": 2.5},
       {"kind": "card", "value": 2.2}, {"kind": "checking", "value": 0.25}]


def t_caller_value_survives():
    kept, skipped = SG.iso_split_injection({"components": CALLER}, {"components": SUB})
    assert kept == {} and skipped == ["components"], (kept, skipped)


def t_absent_key_is_filled():
    kept, skipped = SG.iso_split_injection({}, {"components": SUB})
    assert kept == {"components": SUB} and skipped == []


def t_empty_values_count_as_absent():
    for empty in (None, "", [], {}):
        kept, _ = SG.iso_split_injection({"components": empty}, {"components": SUB})
        assert kept == {"components": SUB}, empty


def t_zero_is_not_empty():
    """0 과 False 는 **값이다** — 비어 있음으로 취급하면 조용히 덮인다."""
    kept, skipped = SG.iso_split_injection({"principal": 0}, {"principal": 96000})
    assert kept == {} and skipped == ["principal"]


def t_partial_fill():
    ctx = {"principal": 96000}
    kept, skipped = SG.iso_split_injection(ctx, {"principal": 1, "actual_apy": 5.1})
    assert kept == {"actual_apy": 5.1} and skipped == ["principal"]


def t_off_restores_old_behaviour():
    kept, skipped = SG.iso_split_injection({"components": CALLER}, {"components": SUB},
                                           fill_only=False)
    assert kept == {"components": SUB} and skipped == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("t_")]
    for f in fns:
        f()
        print("ok %s" % f.__name__)
    print("PASS %d/%d" % (len(fns), len(fns)))
