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

# ★C185(a) catalog_filter 3버킷 — 결측 필드는 제약을 통과시키지 않는다(W5 002 실측 결함).
#   구판: 제약 있음 ∧ 행에 그 사실 없음 → eligible로 통과 = 도구가 "충족" 단언(거짓 유발).
CAT = {"op": "catalog_filter", "table": [
    {"card": "HasHigh", "annual_fee": 200.0, "cashback": 10.0, "source": "doc a"},
    {"card": "HasLow", "annual_fee": 0.0, "cashback": 2.5, "source": "doc b"},
    {"card": "NoData", "annual_fee": 0.0, "source": "doc c"},          # cashback 미문서
    {"card": "InviteOnly", "cashback": 9.0, "invite_only": True, "source": "doc d"},
    {"card": "BizRow", "business": True, "cashback": 9.0, "source": "doc e"},
]}
_r = C.apply_op(CAT, {"min_cashback": 5, "business": False})
_names = lambda b: sorted(x["card"] for x in _r[b])
ck("catalog_eligible_documented_only", _names("eligible"), ["HasHigh"])
ck("catalog_excluded_violation", [(x["card"], "violates" in x["reason"]) for x in _r["excluded"]
                                  if x["card"] == "HasLow"], [("HasLow", True)])
ck("catalog_missing_field_not_eligible", "NoData" in _names("eligible"), False)
ck("catalog_missing_field_unverified", _names("unverified"), ["NoData"])
ck("catalog_unverified_names_field",
   "cashback" in _r["unverified"][0]["undocumented"][0], True)
ck("catalog_invite_only_excluded",
   any(x["card"] == "InviteOnly" and x["reason"] == "invitation-only" for x in _r["excluded"]), True)
ck("catalog_biz_segment_skipped",
   "BizRow" in (_names("eligible") + _names("excluded") + _names("unverified")), False)
# 제약 미지정 → 결측 필드도 미검증 아님(unverified는 '제약이 걸린' 결측만)
_r2 = C.apply_op(CAT, {"business": False})
ck("catalog_no_constraint_no_unverified", sorted(x["card"] for x in _r2["unverified"]), [])
ck("catalog_no_constraint_all_eligible",
   sorted(x["card"] for x in _r2["eligible"]), ["HasHigh", "HasLow", "NoData"])
# note가 unverified 의미를 설명해야(모델이 오해 없이 쓰도록)
ck("catalog_note_explains_unverified", "unverified" in _r["note"], True)

# ★C187(c)(d) 조건부 값 + 신규 제약(한도·구매보호). 003 실측: Silver fx=무구독 2.75/premium 0인데
#   구판이 무조건부 2.75만 보고 gold를 하드 배제했다.
CAT2 = {"op": "catalog_filter",
        "conditional_fields": {"fx_fee": {"alt": "fx_fee_with_premium", "when": "premium_subscriber"}},
        "table": [
            {"card": "CondCard", "fx_fee": 2.75, "fx_fee_with_premium": 0.0,
             "limit_max": 100000, "purchase_protection": True, "source": "doc s"},
            {"card": "FlatBad", "fx_fee": 2.75, "limit_max": 100000, "source": "doc f"},
            {"card": "NoProt", "fx_fee": 0.0, "purchase_protection": False, "source": "doc n"},
            {"card": "SmallLimit", "fx_fee": 0.0, "limit_max": 50000, "source": "doc l"},
        ]}
_c1 = C.apply_op(CAT2, {"max_fx_fee": 0, "premium_subscriber": True, "business": False})
ck("cond_true_uses_alt", sorted(x["card"] for x in _c1["eligible"]),
   ["CondCard", "NoProt", "SmallLimit"])
_c2 = C.apply_op(CAT2, {"max_fx_fee": 0, "premium_subscriber": False, "business": False})
ck("cond_false_uses_base", "CondCard" in [x["card"] for x in _c2["excluded"]], True)
_c3 = C.apply_op(CAT2, {"max_fx_fee": 0, "business": False})           # 조건 미지
ck("cond_unknown_is_unverified", [x["card"] for x in _c3["unverified"]], ["CondCard"])
ck("cond_unknown_cites_clause", "premium_subscriber" in _c3["unverified"][0]["undocumented"][0], True)
ck("cond_flat_still_excluded", "FlatBad" in [x["card"] for x in _c3["excluded"]], True)
# 신규 제약: 한도(ge)·구매보호
_c4 = C.apply_op(CAT2, {"min_credit_limit": 100000, "business": False})
ck("min_credit_limit_excludes", "SmallLimit" in [x["card"] for x in _c4["excluded"]], True)
ck("min_credit_limit_missing_unverified", "NoProt" in [x["card"] for x in _c4["unverified"]], True)
_c5 = C.apply_op(CAT2, {"needs_purchase_protection": True, "business": False})
ck("needs_prot_excludes_documented_no", "NoProt" in [x["card"] for x in _c5["excluded"]], True)
ck("needs_prot_missing_unverified", "FlatBad" in [x["card"] for x in _c5["unverified"]], True)
# 실제 A2 카탈로그 회귀(002/006 gold 보존)
import json as _json
_a2 = _json.load(open("a2/banking_knowledge.gate.json", encoding="utf-8"))
_sp = [t["op"] for t in _a2["scaffold_get_tools"] if t["name"] == "check_card_application_fit"][0]
ck("a2_002_platinum_only",
   [x["card"] for x in C.apply_op(_sp, {"min_cashback": 5, "business": False})["eligible"]],
   ["Platinum Rewards Card"])
ck("a2_006_ecocard_only",
   [x["card"] for x in C.apply_op(_sp, {"max_fx_fee": 1.5, "max_min_payment_pct": 1.5,
                                        "needs_virtual_card": True, "credit_score": 540,
                                        "business": False})["eligible"]], ["EcoCard"])
ck("a2_003_silver_eligible_with_premium",
   "Silver Rewards Card" in [x["card"] for x in C.apply_op(
       _sp, {"max_fx_fee": 0, "needs_purchase_protection": True, "min_credit_limit": 100000,
             "premium_subscriber": True, "credit_score": 720, "business": False})["eligible"]], True)

print("\n%d FAIL" % len(FAILS) if FAILS else "\nALL PASS (t2_compute 일반 op)")
sys.exit(1 if FAILS else 0)
