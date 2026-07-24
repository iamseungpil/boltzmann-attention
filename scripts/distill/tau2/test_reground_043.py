#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BRANCH-REGROUND 직접 검정 (2026-07-24·사용자 "compact plan 재환기가 희석 회피" 주장).
오염된 043 맥락에 분기점 리마인더를 넣어 apply_flag 살아나나:
  R_none    : 오염맥락만 (baseline·자발 apply_flag 기대 0)
  R_rawdoc  : 오염맥락 + doc_003 전문(~4.5K·문서 재투입·C143 계열)
  R_compact : 오염맥락 + compact plan 진척 한 줄 (사용자 주장·희석 없음)
R_compact 성공·R_none 실패 → 사용자 설계 직접 확증. C136(종료·막연 실패)과 대조=분기점·compact가 핵심.
Run(remote): seka python test_reground_043.py --base http://localhost:8140/v1 --n 4
"""
import argparse
import gzip
import json
import os
import re
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
SIMR = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
DOCDIR = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"

# compact plan(사용자가 초반 PLAN-CAPTURE로 얻었을 목록·진척 대조 = 결정론 산출물 모사)
COMPACT = ("[PLAN — required steps for this closure, with progress]\n"
           "  done: verify_identity, pay_credit_card_from_checking, get_closure_reason_history, "
           "log_credit_card_closure_reason, get_pending_replacement_orders, close_credit_card_account\n"
           "  REMAINING (do these before finalizing): get_user_dispute_history, "
           "apply_credit_card_account_flag")
# ★rationale 실은 compact(PLAN-CAPTURE가 깨끗맥락서 도출 시 자연 산출·조건단계 근거 포함)
COMPACT_RAT = ("[PLAN — required steps for this closure, with progress]\n"
               "  done: verify_identity, pay, get_closure_reason_history, log_closure_reason, "
               "get_pending_replacement_orders, close_credit_card_account\n"
               "  REMAINING (do before finalizing):\n"
               "   - get_user_dispute_history (eligibility check: no open disputes)\n"
               "   - apply_credit_card_account_flag  (annual-fee-waiver retention offer: this "
               "customer's reason is the annual fee and tenure is over 2 years, so per the "
               "retention policy you must offer the fee waiver by applying this flag)")

ASK = "What is the very next tool call you make? Name the tool."


def load043():
    d = json.load(gzip.open(os.path.join(SIMR, "bank_rall25a_20260724.results.json.gz")))
    for s in d.get("simulations", []):
        if str(s.get("task_id")) == "task_043":
            return s.get("messages", [])
    raise SystemExit("043 not found")


def polluted(msgs, upto_close=True):
    """오염 맥락 = 전체 궤적(에이전트 tool_call/에러/retry 포함). 분기점=close 직전까지."""
    lines = []
    for m in msgs:
        role, c = m.get("role"), str(m.get("content") or "")
        for tc in (m.get("tool_calls") or []):
            nm = tc.get("name")
            # 분기점: 첫 close 시도 직전에서 컷(아직 finalize 안 함)
            if upto_close and "close_credit_card_account" in str(tc.get("arguments", "")):
                return "\n".join(lines)[:32000]
            lines.append("[%s->tool] %s %s" % (role, nm, str(tc.get("arguments"))[:180]))
        if c.strip():
            lines.append("[%s] %s" % (role, c.strip()[:400]))
    return "\n".join(lines)[:32000]


def full_doc(did):
    p = os.path.join(DOCDIR, did + ".json")
    if os.path.exists(p):
        d = json.load(open(p))
        return (d.get("content") or d.get("text") or "")[:4600]
    return ""


def call(base, model, ctx, temp, headers):
    r = requests.post(base + "/chat/completions", json={
        "model": model, "messages": [
            {"role": "system", "content": "You are a precise banking assistant continuing the conversation."},
            {"role": "user", "content": ctx}],
        "temperature": temp, "max_tokens": 800}, headers=headers, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def hits(txt):
    t = txt.lower()
    fl = ("apply_credit_card_account_flag" in t or "annual_fee_waived" in t
          or ("apply" in t and "flag" in t))
    dh = "get_user_dispute_history" in t or "dispute history" in t
    return fl, dh


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
    poll = polluted(load043())
    doc003 = full_doc("doc_credit_cards_credit_card_account_logistics_003")
    conds = {
        "R_none": poll + "\n\n" + ASK,
        "R_rawdoc": poll + "\n\n[RETENTION POLICY]\n" + doc003 + "\n\n" + ASK,
        "R_compact": poll + "\n\n" + COMPACT + "\n\n" + ASK,
        "R_compact_rat": poll + "\n\n" + COMPACT_RAT + "\n\n" + ASK,
    }
    print("MODEL=%s | polluted ctx=%d chars" % (a.model, len(poll)))
    for label, ctx in conds.items():
        fl = dh = tot = 0
        for i in range(1 + a.n):
            t = 0.0 if i == 0 else 0.7
            try:
                f, d = hits(call(base, a.model, ctx, t, headers))
                fl += f; dh += d; tot += 1
            except Exception as e:
                print("  %s run%d ERR %r" % (label, i, e))
        print("== %-10s apply_flag %d/%d  dispute_history %d/%d  (ctx=%d)"
              % (label, fl, tot, dh, tot, len(ctx)))


if __name__ == "__main__":
    main()
