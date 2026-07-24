#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""position-controlled plan 프로브 (2026-07-24·사용자 질문 "salience=lost-in-the-middle인가").
동일 filler(traj)에 절차(fullkb)를 위치만 바꿔 삽입 → plan 완전성 대조. lost-in-the-middle vs
거리감쇠 vs presence 격리:
  P_start(절차 맨앞)·P_middle(filler 중간)·P_end(질문 직전)·P_absent(절차 없음=baseline).
  중간 최악·끝 최선 → lost-in-the-middle / 단조(먼과거 나쁨) → 거리감쇠 / 위치무관 → presence.
Run: seka python position_probe.py [--provider openrouter --model ...] --n 4
"""
import argparse
import json
import os
import re
import sys

import requests

ART = json.load(open("/tmp/plan043.json"))
FILLER = ART["traj"]                    # 긴 중립 문맥(실제 궤적)
PROC = ART["fullkb"]                    # 절차(doc_001/002/003/016)

REQUIRED = [
    ("closure_reason_history", ["closure reason history", "closure_reason_history", "reason history"]),
    ("pending_replacement", ["pending replacement", "replacement order", "pending_replacement"]),
    ("dispute_history", ["dispute history", "dispute_history"]),
    ("pay_balance", ["pay off", "pay the balance", "pay_credit_card", "pay the outstanding", "pay the $75", "outstanding balance"]),
    ("close_account", ["close the account", "close_credit_card", "close the card", "closing the account"]),
    ("log_reason", ["log the closure", "log_credit_card_closure", "record the closure reason", "log the reason", "closure reason"]),
    ("apply_flag", ["apply the flag", "apply_credit_card_account_flag", "account flag", "apply a flag", "annual_fee_waived", "fee waiver"]),
]
ASK = ("Using the bank policy and the conversation above, list in order EVERY step and tool you must "
       "perform to FULLY complete the customer's request to close their card (include all reads AND "
       "writes required by policy). Number each step.")


def score(txt):
    t = txt.lower()
    return sum(1 for _, kws in REQUIRED if any(k in t for k in kws))


def build(pos):
    hdr = "=== CONVERSATION / CONTEXT ===\n"
    proc = "\n\n=== BANK CLOSURE POLICY (procedure documents) ===\n" + PROC + "\n\n"
    if pos == "P_absent":
        body = hdr + FILLER
    elif pos == "P_start":
        body = proc + hdr + FILLER
    elif pos == "P_end":
        body = hdr + FILLER + proc
    else:  # P_middle
        h = len(FILLER) // 2
        body = hdr + FILLER[:h] + proc + FILLER[h:]
    return [{"role": "system", "content": "You are a precise banking assistant."},
            {"role": "user", "content": body + "\n\n" + ASK}]


def call(base, model, msgs, temp, headers):
    r = requests.post(base + "/chat/completions", json={
        "model": model, "messages": msgs, "temperature": temp, "max_tokens": 2000},
        headers=headers, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


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
    print("MODEL=%s | positions: start/middle/end/absent (same length except absent)" % a.model)
    for pos in ("P_absent", "P_start", "P_middle", "P_end"):
        msgs = build(pos)
        agg = []
        for i in range(1 + a.n):
            t = 0.0 if i == 0 else 0.7
            try:
                agg.append(score(call(base, a.model, msgs, t, headers)))
            except Exception as e:
                print("%s run%d ERROR %r" % (pos, i, e))
        if agg:
            print("== %-9s MEAN %.1f/7 (min %d max %d)  runs=%s"
                  % (pos, sum(agg) / len(agg), min(agg), max(agg), agg))


if __name__ == "__main__":
    main()
