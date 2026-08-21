# -*- coding: utf-8 -*-
"""일회성 — P5(2026-08-21·t7335 halfA 072): get_atm_fee_discrepancies 반환문 완결-인상 제거
+ 미검사 축 문면 명시. 3사본(specific/gate/split-core) 동기.

근거(T7335_NT1_FORENSIC_HALFA_2026_08_21.md task_072 절): 구 반환문 "ONE fee_refund credit ...
across all identified fee discrepancies of THIS account" 이 **완결 인상**을 줘 모델의 보완
검사(누락 fee_rebate 스캔)를 억제 — rebate 5/6 패턴이 [27] 문맥에 실재했는데 [38] 에서 $12.00
그대로 write(gold 대비 $2.00 = 11/14 누락 rebate). 엔진은 구조상 부재-rebate 를 못 본다(A2 입력
스키마 = "ONE element per atm_fee line" — 존재 라인의 금액만 입력 우주).

수리 = **문면·선언만**([[62]] — 누락-rebate 검사 로직 신설 0·op/isolate 불변):
 ① 완결 인상 제거("across all identified ..." 삭제)
 ② 검사한 축 명시(도구 선언 도출: params.transactions "ONE element per atm_fee line" +
    op.select_discrepant actual_field=fee_amount → "전달된 atm_fee 라인의 금액만")
 ③ 검사 안 한 축 명시(fee_rebate 부재 — 기존 A3 선언 축자 "리베이트 축은 별도 fee_rebate
    라인이라 이 도구 범위 밖(모호점 6·11)" · _note_rebate_field 보류 기록을 문면으로 승격)
    + [[64]] fix-naming(forensic 072 처방 "최소 수정 대안" 축자: "check the account's rebate
    policy against the fee_rebate lines yourself")
정책 문구 출처 불변: doc_bank_accounts_bank_accounts_(general)_017 §2 "net correction" 축자 유지
(test_atm_fee_op REG 검정 유지·delta_total 미표기 유지).
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS = ["a2/banking_knowledge.specific.json", "a2/banking_knowledge.gate.json",
         "a2/split/banking_knowledge.core.json"]

RET = ("ATM fee lines whose charged amount does NOT match the documented fee schedule for this "
       "account level: {details}. SCOPE OF THIS CHECK - fee-line amounts only: each atm_fee line "
       "you passed in was compared against the documented fee schedule. This tool did NOT check "
       "whether any rebate is missing - i.e. whether the account's documented rebate policy "
       "promises a fee_rebate line that the transaction history does not show; check the "
       "account's rebate policy against the fee_rebate lines yourself before crediting. If "
       "corrections are owed, the credit policy requires ONE fee_refund credit for the net "
       "correction of THIS account (do not credit the same lines twice).")

NOTE_ADD = (" | 2026-08-21 P5(t7335 halfA 072 실측: 완결-인상 반환문이 모델의 보완 rebate 검사를 "
            "억제 — [27] 문맥에 rebate 5/6 패턴 실재·[38] $12.00 write·차액 $2.00=11/14 누락 "
            "rebate): ①'across all identified fee discrepancies' 완결 문구 제거 ②검사한 축을 "
            "문면 명시(도구 선언 도출: transactions 'ONE element per atm_fee line' + "
            "select_discrepant actual_field=fee_amount = 전달된 fee 라인의 금액만) ③검사 안 한 "
            "축(fee_rebate 부재)을 문면 명시 — 기존 선언 축자('리베이트 축은 별도 fee_rebate "
            "라인이라 이 도구 범위 밖(모호점 6·11)'·_note_rebate_field 보류)의 문면 승격 + "
            "[[64]] fix-naming(072 처방 축자 'check the account's rebate policy against the "
            "fee_rebate lines yourself'). 검사 로직 신설 0([[62]]·op/isolate 불변)·'net "
            "correction' 정책 축자(general_017 §2)·delta_total 미표기 유지. 효과는 미측정.")

entries = []
for rel in PATHS:
    p = os.path.join(HERE, rel)
    j = json.load(io.open(p, encoding="utf-8"))
    hit = None
    for t in j.get("scaffold_get_tools") or []:
        if t.get("name") == "get_atm_fee_discrepancies":
            hit = t
            break
    if hit is None:
        print("MISSING in %s" % rel)
        sys.exit(1)
    hit["return_template"] = RET
    if NOTE_ADD not in (hit.get("_note_") or ""):
        hit["_note_"] = (hit.get("_note_") or "") + NOTE_ADD
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(j, f, ensure_ascii=False, indent=1)
        f.write("\n")
    entries.append(hit)
    print("updated %s" % rel)

print("3사본 json-등가:", entries[0] == entries[1] == entries[2])
