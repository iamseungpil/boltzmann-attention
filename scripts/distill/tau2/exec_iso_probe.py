#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""execution 격리 프로브 (2026-07-24·C141 잔여=execution의 성격 격리).
apply_flag(수수료 waiver) 미실행이 무시(won't)인가 조건추론 실패(can't-reason)인가.
- E0_baseline: doc_003+요청+계좌 → "다음 도구?" (라이브 미실행 재현 확인)
- E_reason: 조건 안 떠먹이고 "이 손님 fee-waiver 자격? 어느 도구?" (추론 검정)
- E_act: 결론+도구명 떠먹이고 "다음 도구 나열" (실행 검정·떠먹여도 하나)
판정: reason실패+act성공→추론갭(LEARN) / reason성공+act성공(그런데 라이브 실패)→salience·execution-in-context /
      reason성공+act실패→순수 무시(FORCE).
Run: seka python exec_iso_probe.py [--provider openrouter --model ...] --n 4
"""
import argparse
import json
import os
import sys

import requests

ART = json.load(open("/tmp/exec043.json"))
DOC = ART["doc003"]
ACCT = ("Account record: Platinum Rewards Card cc_224959b99e_plat, "
        "date_of_account_open: 11/10/2022 (today is in 2025, so open for over 2 years), "
        "current_balance $0.00 (already paid), identity verified.")
REASON = "Customer's stated reason for closing: the annual fee is too expensive."

PROMPTS = {
    "E0_baseline": (
        "BANK CLOSURE/RETENTION POLICY:\n" + DOC + "\n\n" + ACCT + "\n" + REASON
        + "\n\nThe customer wants to close the card. List, in order, the tool calls you make next "
          "to handle this correctly per policy. Name each tool."),
    "E_reason": (
        "BANK CLOSURE/RETENTION POLICY:\n" + DOC + "\n\n" + ACCT + "\n" + REASON
        + "\n\nQUESTION: Per the retention policy, is THIS customer eligible for the annual-fee "
          "waiver retention offer? Answer YES or NO, explain in one line why, and name the exact "
          "tool that implements it."),
    "E_act": (
        "BANK CLOSURE/RETENTION POLICY:\n" + DOC + "\n\n" + ACCT + "\n" + REASON
        + "\n\nGIVEN: This customer QUALIFIES for the annual-fee waiver (annual-fee reason + over "
          "2 years tenure), and the tool that applies it is apply_credit_card_account_flag. "
          "List, in order, the tool calls you make next per policy. Name each tool."),
}


def call(base, model, prompt, temp, headers):
    r = requests.post(base + "/chat/completions", json={
        "model": model, "messages": [
            {"role": "system", "content": "You are a precise banking assistant."},
            {"role": "user", "content": prompt}],
        "temperature": temp, "max_tokens": 1500}, headers=headers, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def scores(txt):
    t = txt.lower()
    applies = ("apply_credit_card_account_flag" in t or "apply the flag" in t
               or "annual_fee_waived" in t or ("waiver" in t and "apply" in t))
    yes = "yes" in t[:200] and "not eligible" not in t[:200] and "no," not in t[:200]
    return applies, yes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--provider", default="vllm", choices=["vllm", "openrouter"])
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    base, headers = a.base, {}
    if a.provider == "openrouter":
        base = "https://openrouter.ai/api/v1"
        headers = {"Authorization": "Bearer " + os.environ.get("OPENROUTER_API_KEY", "")}
    print("MODEL=%s" % a.model)
    for cond in ("E0_baseline", "E_reason", "E_act"):
        applies_ct = yes_ct = tot = 0
        for i in range(1 + a.n):
            t = 0.0 if i == 0 else 0.7
            try:
                ap_, yes = scores(call(base, a.model, PROMPTS[cond], t, headers))
            except Exception as e:
                print("  %s run%d ERR %r" % (cond, i, e)); continue
            applies_ct += ap_; yes_ct += yes; tot += 1
        print("== %-11s applies_flag %d/%d  says_eligible %d/%d"
              % (cond, applies_ct, tot, yes_ct, tot))


if __name__ == "__main__":
    main()
