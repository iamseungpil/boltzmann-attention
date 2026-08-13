# -*- coding: utf-8 -*-
"""일회성 — FIX-10: get_atm_fee_discrepancies 에 fetch_formalize isolate 선언 (3사본).

근거 사슬: x301 C_DUP 8/8(스펙이 형식화 문맥에 실재하면 rho·dup 전부 정답 — 그 문맥이 곧
서브 문맥과 동형) ↔ t7282 라이브(같은 문면이 도구 스키마에만 있으면 도달 0회·A_CUR 오답
그대로) ↔ x301b(재생성-경계 계기로는 라이브 검증 불가·3회째 계기 실패) ⇒ 문면-도달 경로
대신 **서브 배치**(기존 fetch_formalize 기전·§2b 105/105·x275 선례: 라이브 0/8 ↔ 격리 8/8).
판단은 LLM(서브)·엔진은 운반만([[59]] 무결)·실패 시 메인 인자 폴백(거동 보존).
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS = ["a2/banking_knowledge.specific.json", "a2/banking_knowledge.gate.json",
         "a2/split/banking_knowledge.core.json"]

ISO = {
    "mode": "fetch_formalize",
    "ref_params": ["account_id", "account_class"],
    "getter_tools": ["call_discoverable_agent_tool"],
    "operand_keys": ["transactions"],
    "max_rounds": 3,
    "instructions": (
        "You are a records-extraction sub-task. For the checking account in REFERENCE: "
        "(1) call call_discoverable_agent_tool with agent_tool_name="
        "'get_bank_account_transactions_9173' and that account_id to read the account's "
        "transaction records. (2) Build the transactions array with ONE element per atm_fee "
        "line of THIS account. Each element MUST have ONLY these fields, copied from the "
        "records: transaction_id (string - the fee line's id), fee_amount (number - the fee "
        "line's amount as a POSITIVE number, no $), withdrawal_amount (number, POSITIVE - the "
        "cash dispensed by the atm_withdrawal this fee belongs to: the adjacent withdrawal "
        "record with the same account and date), network (string, exactly one of 'rho' | "
        "'non_rho' | 'foreign'). IMPORTANT: the fee line's own description (e.g. 'NON-RHO ATM "
        "FEE') is exactly what you are auditing - it may be wrong. Determine 'network' ONLY "
        "from the paired withdrawal's description: a RHO-BANK machine is 'rho' (a fee on a "
        "RHO-BANK withdrawal is itself the error you are looking for), a machine outside the "
        "U.S. is 'foreign', any other bank's machine is 'non_rho'. Also: if TWO fee lines "
        "belong to the SAME withdrawal, include both, and add \"duplicate_of\": \"<the other "
        "fee line's transaction_id>\" on the second one. Do NOT compute fees - just copy raw "
        "values."),
    "answer_format": ("Reply with exactly one JSON object and nothing else: "
                      "{\"transactions\": [ {\"transaction_id\": \"...\", \"fee_amount\": 0.0, "
                      "\"withdrawal_amount\": 0.0, \"network\": \"...\"} ]}"),
    "_note": ("2026-08-14 FIX-10. x301 4셀 n=8: A_CUR 0/8·B_WARN 0/8(역효과)·C_DUP rho 8/8·"
              "dup 8/8·D_NEG 0/8 — 지시문 핵심 블록 = C_DUP 측정 축자([[03b]]). t7282 라이브: "
              "같은 문면이 스키마에만 있으면 도달 0회(인라인 형식화 = A_CUR 동형 오답·"
              "'paired withdrawal' 문맥 등장 0). x301b(재생성-경계) 계기 실패 → 배치 채널 확정. "
              "기전 = 기존 fetch_formalize(§2b 105/105·get_interest_correction 선례)·판단 = "
              "LLM 서브·엔진 운반만·실패 시 메인 인자 폴백."),
}
PARAM_ACCT = ("string (required) - the checking account ID being audited (e.g. from the "
              "accounts listing); used to look up its transaction records.")

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
    hit["isolate"] = json.loads(json.dumps(ISO))
    if "account_id" not in hit["params"]:
        # 순서 보존: account_class 앞에 account_id 삽입
        newp = {}
        for k, v in hit["params"].items():
            if k == "account_class":
                newp["account_id"] = PARAM_ACCT
            newp[k] = v
        hit["params"] = newp
    hit["description"] = hit["description"].replace(
        "call this ONCE PER ACCOUNT, passing the account's level (class)",
        "call this ONCE PER ACCOUNT, passing the account's id and its level (class)")
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(j, f, ensure_ascii=False, indent=1)
        f.write("\n")
    entries.append(hit)
    print("updated %s" % rel)

print("3사본 json-등가:", entries[0] == entries[1] == entries[2])
