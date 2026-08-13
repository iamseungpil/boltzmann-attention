# -*- coding: utf-8 -*-
"""일회성 — fee 도구 transactions param 문면에 x301 C_DUP 승리 축자 추가 + op dup_field (3사본).

격리 근거: x301_fee_formalize_probe.py 4셀 n=8 — A_CUR rho 0/8·dup 0/8 / B_WARN 0/8·0/8
(추상 경고는 역효과: 정답까지 파괴) / **C_DUP rho 8/8·dup 8/8** / D_NEG(무관 경고 통제).
출시 문면 = C_DUP 팔에서 측정한 WARN+DUP 블록 **축자**([[03b]] 측정한 문구 = 출시할 문구).
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS = ["a2/banking_knowledge.specific.json", "a2/banking_knowledge.gate.json",
         "a2/split/banking_knowledge.core.json"]

# x301 C_DUP 팔 축자(WARN + DUP·probe 원문 그대로)
ADD = (" IMPORTANT: the fee line's own description (e.g. 'NON-RHO ATM FEE') is exactly what "
       "you are auditing - it may be wrong. Determine 'network' ONLY from the paired "
       "withdrawal's description: a RHO-BANK machine is 'rho' (a fee on a RHO-BANK withdrawal "
       "is itself the error you are looking for), a machine outside the U.S. is 'foreign', any "
       "other bank's machine is 'non_rho'. Also: if TWO fee lines belong to the SAME "
       "withdrawal, include both, and add \"duplicate_of\": \"<the other fee line's "
       "transaction_id>\" on the second one.")
NOTE_ADD = (" | 2026-08-14 x301 판정(4셀 n=8): A_CUR rho 0/8(자기-참조 함정: fee 라벨을 network "
            "근거로 오용 — 8/8 결정론 재현·073 이 4시행 내내 8/11 에 멈춘 원인)·B_WARN 0/8"
            "(추상 경고는 역효과: 정답까지 파괴 — [[63]] 더하기-지시 무효의 재실증)·"
            "**C_DUP rho 8/8·dup 8/8**(구체 페어링 작업 지시가 주의를 인출 행으로 끌어 network "
            "부수 정답) → 이 param 문면 = C_DUP 축자. op dup_field='duplicate_of' = 모델 선언의 "
            "산술 귀결(기대 0)만 엔진이 소비([[59]] 무결·x288 산술 계열·[[62]] ③).")

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
    base = hit["params"]["transactions"]
    if ADD not in base:
        hit["params"]["transactions"] = base + ADD
    hit["op"]["dup_field"] = "duplicate_of"
    if NOTE_ADD not in (hit.get("_note_") or ""):
        hit["_note_"] = (hit.get("_note_") or "") + NOTE_ADD
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(j, f, ensure_ascii=False, indent=1)
        f.write("\n")
    entries.append(hit)
    print("updated %s" % rel)

print("3사본 json-등가:", entries[0] == entries[1] == entries[2])
