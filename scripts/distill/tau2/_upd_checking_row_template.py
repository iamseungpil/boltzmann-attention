# -*- coding: utf-8 -*-
"""일회성 — (B2) row_template 컴팩트 렌더 추가(x291b 형식 포렌식·3사본).

근거: x291b strict 11/16(사전등록 ≥12 미달) — 미스 = repr 소음이 문서-재계산 장황 유발
(토큰캡 3)·rebate 조항 견인(2). C_CALC 8/8 의 컴팩트 라인 형식 동형으로 교체.
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS = ["a2/banking_knowledge.specific.json", "a2/banking_knowledge.gate.json",
         "a2/split/banking_knowledge.core.json"]

ROW_T = ("- {account_class}: out-of-network ATM total ${out_of_network_total_usd:.2f} | "
         "foreign ATM total ${foreign_atm_total_usd:.2f} ({source})")
NC_T = "(not computable from the documented schedule: {names})"
RET = ("Documented ATM fee totals per personal checking account class for the stated usage "
       "(monthly free allowances deducted; out-of-network and foreign totals are SEPARATE "
       "columns - how the two fee types combine for one withdrawal abroad is not documented, "
       "so weigh both columns for foreign out-of-network usage):\n{result}\nThese totals "
       "already cover every personal checking class's documented Rho-Bank ATM fees for the "
       "stated usage - they are computed, not estimates. Third-party ATM operator surcharges "
       "and operator-fee rebate programs are OUTSIDE these totals. This tool does not pick a "
       "class: compare the totals yourself, verify the remaining candidate's eligibility and "
       "non-ATM terms in its cited source docs, and confirm with the customer. If no rows "
       "appear, a numeric parameter was missing or unreadable - call again with months, "
       "withdrawals_per_month and withdrawal_amount as plain numbers.")
NOTE_ADD = (" | 2026-08-13 x291b 형식 포렌식(strict 11/16·repr 소음→문서-재계산 장황·rebate "
            "견인): row_template 컴팩트 라인 렌더(C_CALC 8/8 동형 형식)로 교체 — 값·행 순서 "
            "불변·엔진=치환만. 재측정 게이트 = strict ≥12/16.")

entries = []
for rel in PATHS:
    p = os.path.join(HERE, rel)
    j = json.load(io.open(p, encoding="utf-8"))
    hit = None
    for t in j.get("scaffold_get_tools") or []:
        if t.get("name") == "get_checking_atm_fee_totals":
            hit = t
            break
    if hit is None:
        print("MISSING in %s" % rel)
        sys.exit(1)
    hit["op"]["row_template"] = ROW_T
    hit["op"]["not_computable_note"] = NC_T
    hit["return_template"] = RET
    if NOTE_ADD not in (hit.get("_note_") or ""):
        hit["_note_"] = (hit.get("_note_") or "") + NOTE_ADD
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(j, f, ensure_ascii=False, indent=1)
        f.write("\n")
    entries.append(hit)
    print("updated %s" % rel)

print("3사본 json-등가:", entries[0] == entries[1] == entries[2])
