#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PLAN-CAPTURE make-or-break (2026-07-24·C143 §5a·설계 EPLAN_COMPACT_PLAN_REGROUND).
가설: 실궤적서 **요청-매칭 상위 절차문서만**(focused) 자동선택해 계획 서브콜 주면 apply_flag 포함
plan 나온다(C142 E0 재현·C143 35K-전부 0/5와 대조). = focused 자동구성이 배선 가능한가.
비교조건:
  F_all   : 모든 정책 tool출력(35K·C143 재현·기대 실패)
  F_topk  : 요청-매칭 상위 K 문서만(focused 자동선택·기대 성공)
  F_gold  : doc_003만(수동 오라클·C142 E0 상한)
Run(remote): seka python test_plancapture_043.py --base http://localhost:8140/v1 --n 4
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

ASK = ("List, in order, EVERY tool you must call to FULLY complete the customer's request per bank "
       "policy — include every read AND write the policy requires. Name each tool.")


def load043():
    d = json.load(gzip.open(os.path.join(SIMR, "bank_rall25a_20260724.results.json.gz")))
    for s in d.get("simulations", []):
        if str(s.get("task_id")) == "task_043":
            return s.get("messages", [])
    raise SystemExit("043 not found")


def user_text(msgs):
    return "\n".join(str(m.get("content") or "") for m in msgs if m.get("role") == "user")


def retrieved_docs(msgs):
    """궤적서 retrieved 정책문서 조각들: (doc_id, content) — KB tool출력서 파싱."""
    docs = {}
    for m in msgs:
        if m.get("role") != "tool" or m.get("error"):
            continue
        c = str(m.get("content") or "")
        # KB search 출력: 'ID: doc_xxx  ... Content: ...' 블록들
        for mm in re.finditer(r"ID:\s*(doc_[a-z_()0-9]+).*?Content:\s*(.*?)(?=(?:\d+\.\s+[A-Z])|ID:\s*doc_|$)",
                              c, re.S):
            did, body = mm.group(1), mm.group(2).strip()
            if len(body) > len(docs.get(did, "")):
                docs[did] = body[:2500]
    return docs


def full_doc(did):
    p = os.path.join(DOCDIR, did + ".json")
    if os.path.exists(p):
        d = json.load(open(p))
        return (d.get("content") or d.get("text") or "")[:3000]
    return ""


def topk_docs(utext, docs, k=3):
    """요청-매칭 상위 K: 요청 키워드(닫기/해지/취소류 + 명사)와 문서 겹침 점수. 엔진 일반·도메인 리터럴 0."""
    uwords = set(re.findall(r"[a-zA-Z]{4,}", utext.lower()))
    scored = []
    for did, body in docs.items():
        bw = set(re.findall(r"[a-zA-Z]{4,}", body.lower()))
        scored.append((len(uwords & bw), did))
    scored.sort(reverse=True)
    return [did for _, did in scored[:k]]


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
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    base, headers = a.base, {}
    if a.provider == "openrouter":
        base = "https://openrouter.ai/api/v1"
        headers = {"Authorization": "Bearer " + os.environ.get("OPENROUTER_API_KEY", "")}
    msgs = load043()
    ut = user_text(msgs)
    docs = retrieved_docs(msgs)
    topk = topk_docs(ut, docs, a.k)
    gold_id = next((d for d in docs if "logistics_003" in d), None)
    print("MODEL=%s | retrieved docs=%d  topk(%d)=%s  gold003=%s"
          % (a.model, len(docs), a.k, [d.split("_")[-1] for d in topk], bool(gold_id)))

    def ctx_of(ids, use_full=False):
        parts = ["[CUSTOMER] " + ut]
        for did in ids:
            body = full_doc(did) if use_full else docs.get(did, "")
            parts.append("[POLICY %s]\n%s" % (did, body))
        return "\n\n".join(parts)

    conds = {
        "F_all": ctx_of(list(docs.keys())),
        "F_topk": ctx_of(topk, use_full=True),
        "F_gold": ctx_of([gold_id], use_full=True) if gold_id else None,
    }
    for label, ctx in conds.items():
        if ctx is None:
            print("== %-7s (skip)" % label); continue
        hit = tot = 0
        for i in range(1 + a.n):
            t = 0.0 if i == 0 else 0.7
            try:
                hit += has_flag(call(base, a.model, ctx, t, headers)); tot += 1
            except Exception as e:
                print("  %s run%d ERR %r" % (label, i, e))
        print("== %-7s apply_flag %d/%d  (ctx=%d chars)" % (label, hit, tot, len(ctx)))


if __name__ == "__main__":
    main()
