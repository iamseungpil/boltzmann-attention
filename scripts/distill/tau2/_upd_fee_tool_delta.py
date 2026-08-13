# -*- coding: utf-8 -*-
"""일회성 — get_atm_fee_discrepancies 템플릿 갱신(FIX-5·차액 표면화) 3사본 프로그램 삽입.

근거: x288 A_DOCS 0/8(산술 결손) + t7274w 073 실측(id 수신 후에도 금액 오답·중복 크레딧).
문구 출처: doc_bank_accounts_bank_accounts_(general)_017 §2 Fee mischarges 축자
("apply a credit for the net correction across all identified fee discrepancies").
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS = ["a2/banking_knowledge.specific.json", "a2/banking_knowledge.gate.json",
         "a2/split/banking_knowledge.core.json"]

RET = ("ATM fee lines whose charged amount does NOT match the documented fee schedule for "
       "this account level: {details}. If any lines are listed, the credit policy requires "
       "ONE fee_refund credit for the net correction across all identified fee discrepancies "
       "of THIS account = ${delta_total:.2f} (do not credit the same lines twice).")
DETAIL = "{id} (charged ${actual:.2f}, documented fee ${expected:.2f}, difference ${delta:.2f})"
NOTE_ADD = (" | 2026-08-13 FIX-5(t7274w 073 실측: id-만 반환하니 모델이 차액 아닌 값을 크레딧 "
            "$24.50/$15/$5 + chk_3 중복 크레딧): {details}(라인별 charged/documented/difference)와 "
            "{delta_total}(net 합)을 표면화 — x288 A_DOCS 0/8 이 잰 산술 결손 범위 내·엔진이 이미 "
            "계산한 값의 뺄셈·합만([[62]] 최소). 문구 출처 = doc_bank_accounts_bank_accounts_"
            "(general)_017 §2 Fee mischarges 'apply a credit for the net correction across all "
            "identified fee discrepancies' 축자. 픽·페어링·network 분류·credit 실행은 모델 몫 불변.")

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
    hit["detail_item_template"] = DETAIL
    if NOTE_ADD not in (hit.get("_note_") or ""):
        hit["_note_"] = (hit.get("_note_") or "") + NOTE_ADD
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(j, f, ensure_ascii=False, indent=1)
        f.write("\n")
    entries.append(hit)
    print("updated %s" % rel)

print("3사본 json-등가:", entries[0] == entries[1] == entries[2])
