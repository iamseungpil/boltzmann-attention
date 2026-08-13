# -*- coding: utf-8 -*-
"""t7285(nt=4) 073 credit-write 착수 축 전수 포렌식 ([[08]]).

산술 축은 t7284 에서 닫혔다(도구가 9.50/9.00/1.50 = gold 산출). 남은 질문은 **write 착수**:
- credit 계열 도구(apply_checking_account_credit_5829 / apply_statement_credit_8472)가
  unlock/call 로 한 번이라도 등장하는가
- claimprov(CLAIM-PROVENANCE) 문구가 언제 어떤 형태로 나갔는가
- 종료 직전 손님이 본 마지막 assistant 본문은 무엇인가

사용: py -3 bank_t7285_credit_forensic.py [tag ...]
"""
import gzip
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

B = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
CREDIT = ("apply_checking_account_credit", "apply_statement_credit", "update_transaction_rewards")
TAGS = sys.argv[1:] or ["bank_t7285_b_20260814g"]


def calls(m):
    return [tc.get("function", {}).get("name") or tc.get("name")
            for tc in (m.get("tool_calls") or [])]


for tag in TAGS:
    with gzip.open(os.path.join(B, tag + "_results.json.gz"), "rt", encoding="utf-8") as f:
        d = json.load(f)
    print("#" * 78)
    print("#", tag)
    for s in d.get("simulations", []):
        tid = s.get("task_id")
        rw = (s.get("reward_info") or {}).get("reward")
        term = s.get("termination_reason")
        msgs = s.get("messages") or []
        seq, credit_hits, claim_notes, last_txt = [], [], [], ""
        for i, m in enumerate(msgs):
            role = m.get("role")
            content = m.get("content") or ""
            for n in calls(m):
                if not n:
                    continue
                seq.append((i, n))
                if any(c in n for c in CREDIT):
                    credit_hits.append((i, n))
            # unlock 인자로 등장하는 credit 도구명도 착수 시도로 센다
            if role == "assistant":
                for tc in (m.get("tool_calls") or []):
                    a = json.dumps(tc.get("function", {}).get("arguments", ""), ensure_ascii=False)
                    for c in CREDIT:
                        if c in a:
                            credit_hits.append((i, "ARG:" + c))
                if isinstance(content, str) and content.strip():
                    last_txt = content.strip()
            if isinstance(content, str) and "CLAIM-PROVENANCE" in content:
                claim_notes.append((i, role, " ".join(content.split())[:150]))
        print("-" * 78)
        print(f"{tid}  reward={rw}  term={term}  msgs={len(msgs)}  calls={len(seq)}")
        print("  credit 계열 등장:", credit_hits or "**0건**")
        for i, role, t in claim_notes:
            print(f"  [{i}] claimprov({role}): {t}")
        print("  마지막 assistant 본문:", " ".join(last_txt.split())[:220])
