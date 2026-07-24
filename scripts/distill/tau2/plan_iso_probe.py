#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plan-격리 프로브 (2026-07-24·[[10]] plan=LLM 원칙 검정·C124 격리논리 확장).
질문: 에이전트가 close-사슬의 완전한 plan을 (A)최소문맥서 만드나 vs (B)전체궤적서 drift하나?
- A_minimal: 손님요청+계좌+closure 프로토콜 → "완주에 필요한 모든 단계 나열"
- B_fulltraj: 실제 궤적(rall23 043 drift 케이스) + 같은 질문
채점=필수 7단계 중 몇 개 언급(키워드 lenient). A 완전·B 열세 → drift(실행문제)·A도 열세 → plan능력 결손.
Run(remote): seka python plan_iso_probe.py --base http://localhost:8140/v1 --n 6
"""
import argparse
import json
import re
import sys

import requests

ART = json.load(open("/tmp/plan043.json"))

REQUIRED = [
    ("closure_reason_history", ["closure reason history", "closure_reason_history", "reason history"]),
    ("pending_replacement", ["pending replacement", "replacement order", "pending_replacement"]),
    ("dispute_history", ["dispute history", "dispute_history"]),
    ("pay_balance", ["pay off", "pay the balance", "pay_credit_card", "pay the outstanding", "pay the $75"]),
    ("close_account", ["close the account", "close_credit_card", "close the card", "closing the account"]),
    ("log_reason", ["log the closure", "log_credit_card_closure", "record the closure reason",
                    "log the reason", "closure reason"]),
    ("apply_flag", ["apply the flag", "apply_credit_card_account_flag", "account flag", "apply a flag"]),
]

ASK = ("List, in order, EVERY step and tool you must perform to FULLY and correctly complete the "
       "customer's request (do not stop early — include all reads AND writes required by policy). "
       "Number each step.")


def score(txt):
    t = txt.lower()
    hit = [name for name, kws in REQUIRED if any(k in t for k in kws)]
    return len(hit), hit


def build(variant):
    if variant == "A_minimal":
        u = ("CUSTOMER: " + ART["request"] + "\n\nACCOUNT RECORD:\n" + ART.get("account", "")
             + "\n\nCLOSURE POLICY:\n" + ART.get("kb", "") + "\n\n" + ASK)
    else:
        u = ("Below is an in-progress bank-agent conversation transcript.\n\n" + ART["traj"]
             + "\n\n" + ASK)
    return [{"role": "system", "content": "You are a precise banking assistant."},
            {"role": "user", "content": u}]


def call(base, model, msgs, temp):
    r = requests.post(base + "/chat/completions", json={
        "model": model, "messages": msgs, "temperature": temp, "max_tokens": 700}, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    print("required steps: %d — %s" % (len(REQUIRED), [r[0] for r in REQUIRED]))
    for variant in ("A_minimal", "B_fulltraj"):
        msgs = build(variant)
        agg = []
        for i in range(1 + a.n):
            t = 0.0 if i == 0 else 0.7
            try:
                sc, hit = score(call(a.base, a.model, msgs, t))
            except Exception as e:
                print("%s run%d ERROR %r" % (variant, i, e))
                continue
            agg.append(sc)
            miss = [n for n, _ in REQUIRED if n not in hit]
            print("%s run%d: %d/7  missing=%s" % (variant, i, sc, miss))
        if agg:
            print("== %s MEAN %.1f/7 (min %d max %d)" % (variant, sum(agg) / len(agg), min(agg), max(agg)))


if __name__ == "__main__":
    main()
