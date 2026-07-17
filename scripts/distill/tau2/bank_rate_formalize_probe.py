#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""★rate-formalize 측정 (무료·2026-07-18) — §PROD-2 재설계의 성패를 가른다.

질문: base 32B가 **KB rate 문서 + 원거래 + 계정개설일**을 받아 각 거래의 **최종 배율(rate)**을
  정확히 formalize하나? (§PROD-2 ③ 분담선: base_rate·promo는 LLM formalize·날짜산술은 엔진)
정답(task_026·gold÷amount 확정·개설일 02/13/2025 6개월=08/13):
  txn ...403 Business Silver/Travel 03/22 → 20 (10%×2 프로모)
  txn ...410 Business Silver/Travel 08/25 → 10 (프로모 만료)
  txn ...411 Business Silver/Software 09/15 → 10
  txn ...506 Silver/Software        05/18 → 4
판정: n샘플 중 4/4 정확히 낸 비율. 높으면 분담선 이동=승리. 낮으면 [[45]] 부하(scale)로 재분류.
⚠️엔진 리터럴 0(측정 코드다·[[05]] 대상 아님). rate는 **모델이** 낸다 — 우리가 안 알려준다.

Run: python3 bank_rate_formalize_probe.py --base http://localhost:8141/v1 --n 8
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bank_fab_probes import post  # noqa: E402

GOLD = {"txn_a8f1c2d3e403": 20, "txn_a8f1c2d3e410": 10,
        "txn_a8f1c2d3e411": 10, "txn_b7e2d4c5f506": 4}

TXNS = [
    {"transaction_id": "txn_a8f1c2d3e403", "card": "Business Silver Rewards Card", "category": "Travel",
     "merchant": "JetBlue Airways", "amount": 315.00, "date": "03/22/2025"},
    {"transaction_id": "txn_a8f1c2d3e410", "card": "Business Silver Rewards Card", "category": "Travel",
     "merchant": "Southwest Airlines", "amount": 380.00, "date": "08/25/2025"},
    {"transaction_id": "txn_a8f1c2d3e411", "card": "Business Silver Rewards Card", "category": "Software",
     "merchant": "Atlassian", "amount": 149.99, "date": "09/15/2025"},
    {"transaction_id": "txn_b7e2d4c5f506", "card": "Silver Rewards Card", "category": "Software",
     "merchant": "Adobe", "amount": 255.00, "date": "05/18/2025"},
]
ACCOUNT_OPEN = "02/13/2025"


def build_prompt(kb):
    docs = "\n\n".join(f"### {v['title']}\n{v['content']}" for k, v in kb.items() if not k.startswith("_"))
    txn_lines = "\n".join(
        f"  {t['transaction_id']}: card={t['card']}, category={t['category']}, "
        f"amount=${t['amount']}, date={t['date']}" for t in TXNS)
    return (
        "You are computing cash-back reward rates. Below are the bank's reward-rate policy documents, "
        "the customer's account opening date, and their transactions.\n\n"
        "=== POLICY DOCUMENTS ===\n" + docs + "\n\n"
        f"=== ACCOUNT ===\nAccount opened: {ACCOUNT_OPEN}\n\n"
        "=== TRANSACTIONS ===\n" + txn_lines + "\n\n"
        "For EACH transaction, determine the reward RATE MULTIPLIER that applies (as a plain number: "
        "e.g. a 10% rate is 10, a 4% rate is 4, and if a 2x promo applies multiply it — e.g. 10% doubled is 20). "
        "Reason from the policy documents about the base rate for the card+category, and whether the "
        "double-cash-back promo applies (check the account opening date and the promo's timing rule). "
        "Reply with EXACTLY one JSON object and nothing else, mapping each transaction_id to its final "
        "multiplier number:\n"
        '{"txn_a8f1c2d3e403": <n>, "txn_a8f1c2d3e410": <n>, "txn_a8f1c2d3e411": <n>, "txn_b7e2d4c5f506": <n>}')


def parse(text):
    for chunk in (text or "").split("{")[::-1]:
        try:
            j = json.loads("{" + chunk.split("}")[0] + "}")
            if isinstance(j, dict) and any(k in j for k in GOLD):
                return {k: j.get(k) for k in GOLD}
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8141/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--kb", default="/home/woori/scratch/kb_rate_docs.json")
    a = ap.parse_args()
    kb = json.load(open(a.kb, encoding="utf-8"))
    prompt = build_prompt(kb)
    print("프롬프트 %d자 · 정답 %s\n" % (len(prompt), GOLD))

    allcorrect = 0
    percell = {k: 0 for k in GOLD}
    for i in range(a.n):
        try:
            r = post(a.base, {"model": a.model, "temperature": a.temp, "max_tokens": 1500,
                              "messages": [{"role": "user", "content": prompt}]}, timeout=300)
            out = parse(r["choices"][0]["message"].get("content"))
        except Exception as e:
            print("  [%d] ERR %r" % (i, str(e)[:60]))
            continue
        if out is None:
            print("  [%d] 파싱실패" % i)
            continue
        hits = {k: (float(out[k]) == GOLD[k]) if out.get(k) is not None else False for k in GOLD}
        for k, v in hits.items():
            percell[k] += v
        ok = all(hits.values())
        allcorrect += ok
        print("  [%d] %s  %s" % (i, "✓ALL" if ok else "부분", {k: out.get(k) for k in GOLD}))

    print("\n" + "=" * 60)
    print("★4/4 전부 정확 = %d/%d = %.0f%%" % (allcorrect, a.n, 100 * allcorrect / max(a.n, 1)))
    print("★셀별 정확율:", {k: f"{percell[k]}/{a.n}" for k in GOLD})
    print("판정: 높으면 §PROD-2 분담선 이동=승리 · 낮으면 rate-formalize가 32B에 [[45]] 부하")


if __name__ == "__main__":
    main()
