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

# 안전: 미지 op·결측 → None
ck("unknown_op", C.apply_op({"op": "frobnicate"}, ctx), None)
ck("missing_ref", C.apply_op({"op": "ref", "path": "params.nonexistent"}, ctx), None)

print("\n%d FAIL" % len(FAILS) if FAILS else "\nALL PASS (t2_compute 일반 op)")
sys.exit(1 if FAILS else 0)
