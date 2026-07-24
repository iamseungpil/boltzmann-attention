#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""isolated-planning 서브콜 오프라인 검증 (2026-07-24·C142 처방).
가설: 계획 서브콜을 **깨끗한 맥락**(user 요청 + retrieved 정책 tool-output)으로 주면 apply_flag를
내지만, **full transcript**(오염)로 주면 놓친다. rall25 043 실궤적으로 대조.
clean 맥락 = user 발화 ∪ KB/정책 tool 출력(에이전트 자기 행동·에러·retry 제외).
Run(remote): seka python test_isoplan_043.py --base http://localhost:8140/v1 --n 4
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

ASK = ("List, in order, EVERY tool you must call to FULLY complete the customer's request per bank "
       "policy (include every read AND write the policy requires, e.g. the retention protocol). "
       "Name each tool.")


def load043():
    d = json.load(gzip.open(os.path.join(SIMR, "bank_rall25a_20260724.results.json.gz")))
    for s in d.get("simulations", []):
        if str(s.get("task_id")) == "task_043":
            return s.get("messages", [])
    raise SystemExit("043 not found")


def clean_context(msgs):
    """깨끗한 계획 맥락: user 발화 + KB/정책 tool 출력만(에이전트 tool_call/에러/retry 제외).
    정책 출력 = KB_search 결과(문서 내용). 에이전트 자기 행동의 오염 제거 = C142 처방."""
    parts = []
    for m in msgs:
        role = m.get("role")
        c = str(m.get("content") or "")
        if role == "user" and c.strip():
            parts.append("[CUSTOMER] " + c.strip())
        if role == "tool" and not m.get("error"):
            # 정책/문서 출력만(계좌 레코드·KB 문서). 에러·짧은 확인문은 제외.
            if ("## " in c or "Step " in c or "Eligibility" in c or "policy" in c.lower()
                    or "Record ID" in c) and len(c) > 200:
                parts.append("[POLICY/RECORD] " + c.strip()[:2500])
    return "\n\n".join(parts)


def full_context(msgs):
    lines = []
    for m in msgs:
        role, c = m.get("role"), str(m.get("content") or "")
        for tc in (m.get("tool_calls") or []):
            lines.append("[%s->tool] %s %s" % (role, tc.get("name"), str(tc.get("arguments"))[:200]))
        if c.strip():
            lines.append("[%s] %s" % (role, c.strip()[:400]))
    return "\n".join(lines)[:60000]


def call(base, model, ctx, temp, headers):
    r = requests.post(base + "/chat/completions", json={
        "model": model, "messages": [
            {"role": "system", "content": "You are a precise banking assistant."},
            {"role": "user", "content": ctx + "\n\n" + ASK}],
        "temperature": temp, "max_tokens": 1500}, headers=headers, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def has_flag(txt):
    t = txt.lower()
    return ("apply_credit_card_account_flag" in t or "annual_fee_waived" in t
            or ("apply" in t and "flag" in t) or ("fee waiver" in t and "apply" in t))


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
    msgs = load043()
    cc, fc = clean_context(msgs), full_context(msgs)
    print("MODEL=%s | clean_ctx=%d chars  full_ctx=%d chars  (apply_flag in clean-ctx source: %s)"
          % (a.model, len(cc), len(fc), "apply_credit_card_account_flag" in cc.lower()
             or "annual fee" in cc.lower()))
    for label, ctx in (("CLEAN", cc), ("FULL", fc)):
        hit = 0; tot = 0
        for i in range(1 + a.n):
            t = 0.0 if i == 0 else 0.7
            try:
                hit += has_flag(call(base, a.model, ctx, t, headers)); tot += 1
            except Exception as e:
                print("  %s run%d ERR %r" % (label, i, e))
        print("== %-5s plans apply_flag: %d/%d" % (label, hit, tot))


if __name__ == "__main__":
    main()
