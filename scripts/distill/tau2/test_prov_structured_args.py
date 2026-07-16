# -*- coding: utf-8 -*-
"""PROV 거짓양성 회귀 테스트 (2026-07-16 라이브 사고).

사고: 에이전트가 `verify_identity(provided={"date_of_birth":"11/03/1990","phone_number":"312-555-0481"})`
      를 **올바르게** 호출했는데 PROV가 매번 "invented"로 반려 → 에이전트가 OTP/보안질문을 지어내고
      상담원 이관(bank_ctl_20260716_2215 sim0). 원인 2개:
        (1) `"id" in "provided"` = True  → 부분문자열 힌트 매칭이 `provided`를 식별자 인자로 오판
        (2) `_flatten`이 JSON **문자열**을 안 품 → JSON 덩어리 전체를 문맥서 찾다 실패
사용: py -3 test_prov_structured_args.py
"""
import json
import sys

sys.path.insert(0, ".")
from t2_gate_patch import _flatten, _hint_hit, DEFAULT_ARG_HINTS as H  # noqa: E402

fail = 0


def ck(desc, got, want):
    global fail
    ok = got == want
    fail += (not ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {desc}\n        got={got!r} want={want!r}")


# ── (1) 힌트 매칭: 오탐 제거하되 기존 참양성 유지
ck("'provided'는 식별자 아님 (id 부분문자열 오탐)", _hint_hit("provided", H), False)
ck("'record'는 식별자 아님", _hint_hit("record", H), False)
ck("'valid_until'는 식별자 아님", _hint_hit("valid_until", H), False)
ck("'transactions'는 식별자 아님", _hint_hit("transactions", H), False)
ck("★기존유지: 'user_id'", _hint_hit("user_id", H), True)
ck("★기존유지: 'order_id'", _hint_hit("order_id", H), True)
ck("★기존유지: 'item_ids'", _hint_hit("item_ids", H), True)
ck("★기존유지: 'address1' (C24 대상)", _hint_hit("address1", H), True)
ck("★기존유지: 'payment_method_id'", _hint_hit("payment_method_id", H), True)
ck("★기존유지: 'phone_number'", _hint_hit("phone_number", H), True)

# ── (2) _flatten: JSON 문자열 → leaf
v = json.dumps({"date_of_birth": "11/03/1990", "phone_number": "312-555-0481"})
ck("JSON 문자열이 leaf로 분해됨", sorted(_flatten(v)), ["11/03/1990", "312-555-0481"])
ck("JSON 배열 문자열도 분해", sorted(_flatten('["a1","b2"]')), ["a1", "b2"])
ck("평범한 문자열은 그대로", list(_flatten("txn_123")), ["txn_123"])
ck("JSON 아닌 중괄호 문자열도 그대로", list(_flatten("{not json}")), ["{not json}"])
ck("dict는 기존대로", sorted(_flatten({"a": "x1", "b": "y2"})), ["x1", "y2"])

# ── (3) 사고 재현: 그 호출이 이제 통과하는가
ctx = ("hi. full name: priya sharma, phone number: 312-555-0481, "
       "date of birth: 11/03/1990. i don't have my user id.").lower()
leaves = [str(x).strip() for x in _flatten(v)]
ck("★사고 재현: provided의 모든 leaf가 문맥에 실재", all(l.lower() in ctx for l in leaves), True)
ck("★사고 재현: 'provided'는 애초에 검사 대상 아님", _hint_hit("provided", H), False)

print()
print("ALL PASS" if not fail else f"{fail} FAILED")
sys.exit(1 if fail else 0)
