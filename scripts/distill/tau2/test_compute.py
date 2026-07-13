# -*- coding: utf-8 -*-
"""t2_compute 일반 op 라이브러리 유닛 (2026-07-13·keystone·[[05]] 도메인일반)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_compute as C

FAILS = []
def ck(n, got, exp):
    ok = (got == exp) or (isinstance(got, float) and isinstance(exp, (int, float)) and abs(got - exp) < 1e-6)
    print(("PASS " if ok else "FAIL ") + n + ("" if ok else " | got=%r exp=%r" % (got, exp)))
    if not ok: FAILS.append(n)

ctx = {"params": {"disputed_amount": 100.0, "transaction_date": "11/05/2025",
                  "discovery_date": "11/06/2025", "expected": 250.0, "actual": 180.0},
       "records": [{"apy": 3.5, "id": "a"}, {"apy": 4.2, "id": "b"}, {"apy": 1.1, "id": "c"}]}

ck("const", C.apply_op({"op": "const", "value": 50}, ctx), 50)
ck("ref", C.apply_op({"op": "ref", "path": "params.disputed_amount"}, ctx), 100.0)
ck("min", C.apply_op({"op": "min", "of": ["params.disputed_amount", "50"]}, ctx), 50)
ck("max", C.apply_op({"op": "max", "of": ["params.expected", "params.actual"]}, ctx), 250.0)
ck("diff", C.apply_op({"op": "diff", "a": "params.expected", "b": "params.actual"}, ctx), 70.0)
ck("clamp", C.apply_op({"op": "clamp", "value": "params.disputed_amount", "max": "50"}, ctx), 50)
ck("days_between", C.apply_op({"op": "days_between", "a": "params.transaction_date",
                               "b": "params.discovery_date"}, ctx), 1)
ck("argmax", C.apply_op({"op": "argmax", "over": "records", "key": "apy", "return": "id"}, ctx), "b")
ck("argmin", C.apply_op({"op": "argmin", "over": "records", "key": "apy", "return": "id"}, ctx), "c")

# ★liability lookup_table: 명세서-신고 타이밍 → {≤2:50, ≤60:500, else:full}
LIAB = {"op": "lookup_table",
        "key": {"op": "days_between", "a": "params.transaction_date", "b": "params.discovery_date"},
        "table": [{"cmp": "<=", "thr": 2, "result": 50},
                  {"cmp": "<=", "thr": 60, "result": 500},
                  {"result": {"op": "ref", "path": "params.disputed_amount"}}]}
ck("liab_timely_50", C.apply_op(LIAB, ctx), 50)          # 1일 → 50
ctx2 = dict(ctx); ctx2["params"] = dict(ctx["params"], discovery_date="12/20/2025")  # 45일
ck("liab_mid_500", C.apply_op(LIAB, ctx2), 500)
ctx3 = dict(ctx); ctx3["params"] = dict(ctx["params"], discovery_date="03/20/2026")  # >60일
ck("liab_late_full", C.apply_op(LIAB, ctx3), 100.0)

# ★filter op (reference-filter keystone): criteria로 record 매칭 → id
RECS = [{"transaction_id": "btxn_1", "date": "11/05/2025", "description": "RHO-BANK ATM #4827 WITHDRAWAL", "type": "atm_withdrawal"},
        {"transaction_id": "btxn_2", "date": "11/06/2025", "description": "CITYFIT GYM MONTHLY", "type": "debit_card_purchase"},
        {"transaction_id": "btxn_3", "date": "11/06/2025", "description": "CITYFIT GYM MONTHLY", "type": "debit_card_purchase"}]
fctx = lambda crit: {"records": RECS, "criteria": crit}
FSPEC = {"op": "filter", "over": "records", "return": "transaction_id",
         "match": [{"field": "date", "eq": "criteria.date"},
                   {"field": "description", "contains": "criteria.merchant"},
                   {"field": "type", "eq": "criteria.type"}]}
# date로 유일(ATM)
ck("filter_date_unique", C.apply_op(FSPEC, fctx({"date": "11/05/2025"})), "btxn_1")
# merchant substring으로 유일(단 CityFit는 2건이라 date+merchant도 애매) → date+merchant
ck("filter_merchant", C.apply_op(FSPEC, fctx({"date": "11/05/2025", "merchant": "ATM"})), "btxn_1")
# 애매(CityFit 2건·date만) → on_ambiguous 기본 none
ck("filter_ambiguous_none", C.apply_op(FSPEC, fctx({"date": "11/06/2025", "merchant": "CITYFIT"})), None)
# on_ambiguous=first
FSPEC_F = dict(FSPEC, on_ambiguous="first")
ck("filter_ambiguous_first", C.apply_op(FSPEC_F, fctx({"date": "11/06/2025", "merchant": "CITYFIT"})), "btxn_2")
# 0 매칭 → None
ck("filter_nomatch", C.apply_op(FSPEC, fctx({"date": "12/25/2025"})), None)
# 부분기준(criteria 결측 필드는 조건 스킵)
ck("filter_partial_crit", C.apply_op(FSPEC, fctx({"merchant": "ATM"})), "btxn_1")

# 안전: 미지 op·결측 → None
ck("unknown_op", C.apply_op({"op": "frobnicate"}, ctx), None)
ck("missing_ref", C.apply_op({"op": "ref", "path": "params.nonexistent"}, ctx), None)

print("\n%d FAIL" % len(FAILS) if FAILS else "\nALL PASS (t2_compute 일반 op)")
sys.exit(1 if FAILS else 0)
