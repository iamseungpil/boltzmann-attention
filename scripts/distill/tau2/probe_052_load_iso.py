#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""052 부하-격리 프로브 (2026-07-22·사용자 지시 "052도 격리하라").

질문: 052의 last_approved_cli_submitted_date 오형식화('none' vs gold 2025-09-15)가
**부하/gather-순서**인가 **능력(formalize)**인가?
- p_traj(라이브·rall12 재스모크): 'none'(오답) — 단 check_cli 호출 시점에 CLI history 미독(순서 의심).
- p_iso = 이 프로브: **정보-완비 격리**(계좌+CLI history[approved 001·PENDING 2,3]+payment+time을 명시 제공)
  에서 last_approved formalize를 n회 샘플 → 2025-09-15 정답률.
판정: p_iso 높음 → 부하/gather-순서(라이브서 CLI history 선독 안 함)→scaffold(READLOOP/순서게이트).
      p_iso 낮음(none 반복) → 능력(approved 필터 formalize learn 잔여·LEARN §2.1).
측정 규율(등대): gold 2025-09-15가 컨텍스트에 실재(선검사)·유일 approved.
비용: 무료(로컬 vLLM). Run: python probe_052_load_iso.py --n 8
"""
import argparse
import json
import re

TIME = "The current time is 2025-11-14 03:40:00 EST."
ACCOUNT = """Found 1 record(s) in 'credit_card_accounts':
1. Record ID: cc_5e4c1a83b0_bronze
   account_id: cc_5e4c1a83b0_bronze
   card_type: Bronze Rewards Card
   date_of_account_open: 05/10/2023
   current_balance: $1,500.00
   credit_limit: $4,000.00
   account_status: ACTIVE"""
CLI_HISTORY = """Credit limit increase history for account cc_5e4c1a83b0_bronze:
Found 3 record(s):
1. request_id: cli_5e4c1a83b0_bronze_001
   status: approved
   submitted_date: 2025-09-15
   previous_limit: $3,000.00  new_limit: $4,000.00
2. request_id: cli_e33db0778663
   requested_increase_amount: 1000
   submitted_at: 11/14/2025
   status: PENDING
3. request_id: cli_cb2cc23854d1
   requested_increase_amount: 400
   submitted_at: 11/14/2025
   status: PENDING"""
PAYMENT = """Payment history for account 'cc_5e4c1a83b0_bronze' (last 6 months):
Consecutive on-time payments: 6"""

INFO = "\n\n".join(["TOOL OUTPUT (current time):\n" + TIME,
                    "TOOL OUTPUT (credit card account):\n" + ACCOUNT,
                    "TOOL OUTPUT (CLI request history):\n" + CLI_HISTORY,
                    "TOOL OUTPUT (payment history):\n" + PAYMENT])

# check_cli의 last_approved param 설명(A2·문구교정판)을 그대로 사용
LAST_APPROVED_DESC = ("Look at the CLI request history tool output. Among the requests whose status "
                      "is 'approved', copy the most recent submitted_date (as shown, e.g. 2025-09-15). "
                      "The request the customer just submitted in THIS conversation has status "
                      "'pending' (not 'approved'), so it is automatically excluded - do not confuse "
                      "it with a prior approval. Pass 'none' ONLY if NO request in the history has "
                      "status 'approved'.")

PROMPT = """You are a bank agent deciding a credit limit increase on the customer's Bronze Rewards \
Card. Below is the information gathered (tool outputs).

=== CASE FILE ===
{info}
=== END CASE FILE ===

Determine ONE input for the eligibility check:
- last_approved_cli_submitted_date: {desc}

Reply with exactly one JSON object and nothing else:
{{"last_approved_cli_submitted_date": ""}}"""


def last_json(text):
    out = None
    for m in re.finditer(r"\{[^{}]*\}", text or "", re.S):
        try:
            out = json.loads(m.group(0))
        except Exception:
            pass
    return out


def score(j):
    if not isinstance(j, dict):
        return {"parse": False}
    v = str(j.get("last_approved_cli_submitted_date", "")).strip().lower()
    return {"parse": True,
            "approved_ok": ("2025-09-15" in v or "09/15/2025" in v),
            "said_none": (v in ("none", "", "null")),
            "raw": v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--base", default="http://localhost:8141/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    a = ap.parse_args()
    presence = {"2025-09-15": "2025-09-15" in INFO, "approved": "approved" in INFO}
    print("[probe052] ctx_chars=%d presence=%s" % (len(INFO), presence))
    from openai import OpenAI
    cl = OpenAI(base_url=a.base, api_key="dummy")
    rows = []
    for i in range(a.n + 1):
        temp = 0.0 if i == 0 else 0.7
        r = cl.chat.completions.create(model=a.model, temperature=temp, max_tokens=120,
                                       messages=[{"role": "user",
                                                  "content": PROMPT.format(info=INFO, desc=LAST_APPROVED_DESC)}])
        sc = score(last_json(r.choices[0].message.content))
        sc["temp"] = temp
        rows.append(sc)
        print("[%d] t=%.1f approved_ok=%s none=%s raw=%r"
              % (i, temp, sc.get("approved_ok"), sc.get("said_none"), sc.get("raw")), flush=True)
    n = len(rows)
    print("\n== p_iso 집계 (n=%d·정보완비 격리) ==" % n)
    print("  approved_ok(=2025-09-15): %d/%d   (p_traj 라이브=0·모델 'none')"
          % (sum(1 for r in rows if r.get("approved_ok")), n))
    print("  said_none(오답): %d/%d" % (sum(1 for r in rows if r.get("said_none")), n))
    print("판정: approved_ok 높음 → 부하/gather-순서(scaffold) / none 반복 → formalize 능력(learn)")


if __name__ == "__main__":
    main()
