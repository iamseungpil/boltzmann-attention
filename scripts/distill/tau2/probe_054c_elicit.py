#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""054c 순서-elicitation sweep (2026-07-22·사용자 지시 "여러 방법으로 규칙→순서 연결 되는가").

배경: C1(규칙 상식)=9/9 wait·C2(3요청 동시순서·힌트)=0/9 CLI-먼저. 사용자 통찰:
C2 실패가 "규칙-적용 무능"인지 "3개 동시정렬 부하"인지 세분화 — pairwise·greedy·chain 등으로 분해하면
규칙 지식을 순서로 연결할 수 있지 않나?
방법 sweep(어느 하나라도 CLI-먼저 끌어내면 = 부하/활성화·scaffold / 전부 실패 = 규칙-적용 추론 learn):
- M1 pair_norule    : dispute vs CLI, 규칙 미제공(순수 상식 2택)
- M2 pair_rule      : dispute vs CLI, 규칙 명시(모델이 C1서 아는 것·chain 1홉)
- M3 greedy_rule    : 3작업 중 "지금 첫 번째 하나"만 규칙과 함께
- M4 chain          : 단계 분해(dispute후 상태→CLI가능?→순서)
- M5 consequence    : "dispute 먼저 후 CLI 요청하면 승인되나?"(인과 방향·규칙 미제공)
판정: M2~M4 중 CLI-먼저 높음 → 규칙 상기+분해로 닫힘(scaffold·activation) / 전부 낮음 → 적용추론 learn.
주의: 규칙 제공은 모델 자신이 C1서 아는 상식의 명시(진단)·라이브 A2이식과 구분([[03b]]).
비용: 무료(로컬 vLLM). Run: python probe_054c_elicit.py --n 8
"""
import argparse
import json
import re

RULE = ("BANK RULE: while an account has a pending fraud dispute OR a replacement card that has "
        "been ordered but not yet delivered/activated, a credit limit increase on that account "
        "CANNOT be approved.")

M1 = ("A customer wants both: (A) to file a dispute for a fraudulent charge, and (B) a credit limit "
      "increase - on the same credit card account. Which should you do FIRST? Reply with exactly one "
      'JSON object: {"first": "dispute" | "credit_limit_increase", "reason": "<one sentence>"}')

M2 = (RULE + "\n\nThe customer wants both (A) to file a dispute for a fraudulent charge and (B) a "
      "credit limit increase on the same account. Which should you do FIRST to avoid one blocking "
      'the other? Reply with exactly one JSON object: '
      '{"first": "dispute" | "credit_limit_increase", "reason": "<one sentence>"}')

M3 = (RULE + "\n\nThe customer asks for three things on one account: file a fraud dispute, order a "
      "replacement card, and get a credit limit increase. You will do them one at a time. Which "
      'ONE should you do FIRST? Reply with exactly one JSON object: '
      '{"first": "dispute" | "replacement" | "credit_limit_increase", "reason": "<one sentence>"}')

M4 = ("Reason step by step about a credit card account. (a) If you file a fraud dispute on a "
      "transaction, what status does that create on the account? (b) Given that status, can a "
      "credit limit increase be approved on that account right now? (c) Therefore, between filing "
      "the dispute and doing the credit limit increase, which must come FIRST? Reply with exactly "
      'one JSON object: {"a": "<...>", "b": "<yes/no>", "c": "dispute" | "credit_limit_increase"}')

M5 = ("A customer files a fraud dispute on their card, and THEN (while that dispute is still "
      "pending) asks for a credit limit increase on the same account. Can the credit limit increase "
      'be approved? Reply with exactly one JSON object: {"approved": "yes" | "no", "reason": "<one sentence>"}')


def last_json(text):
    out = None
    for m in re.finditer(r"\{.*\}", text or "", re.S):
        try:
            out = json.loads(m.group(0))
        except Exception:
            pass
    return out


def cli_first(j, key="first"):
    if not isinstance(j, dict):
        return None
    v = str(j.get(key, "")).lower()
    return "credit" in v or "limit" in v or "cli" in v


def m5_blocked(j):
    if not isinstance(j, dict):
        return None
    return "no" in str(j.get("approved", "")).lower()


def run(cl, model, prompt, n, label, judge, key="first"):
    hits, parsed = 0, 0
    for i in range(n + 1):
        temp = 0.0 if i == 0 else 0.7
        r = cl.chat.completions.create(model=model, temperature=temp, max_tokens=220,
                                       messages=[{"role": "user", "content": prompt}])
        j = last_json(r.choices[0].message.content)
        v = judge(j, key) if key else judge(j)
        if j is not None:
            parsed += 1
        if v:
            hits += 1
        detail = (j or {}).get(key) if (isinstance(j, dict) and key) else (j or {}).get("approved") if isinstance(j, dict) else None
        print("  [%s %d] t=%.1f -> %s (CLI-first/blocked=%s)" % (label, i, temp, detail, v), flush=True)
    print("== %s: %d/%d (parsed %d)" % (label, hits, n + 1, parsed), flush=True)
    return hits, n + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--base", default="http://localhost:8141/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    a = ap.parse_args()
    from openai import OpenAI
    cl = OpenAI(base_url=a.base, api_key="dummy")
    res = {}
    print("=== M1 pair_norule (dispute vs CLI·규칙 미제공) ===")
    res["M1"] = run(cl, a.model, M1, a.n, "M1", cli_first)
    print("=== M2 pair_rule (dispute vs CLI·규칙 명시) ===")
    res["M2"] = run(cl, a.model, M2, a.n, "M2", cli_first)
    print("=== M3 greedy_rule (3작업 첫선택·규칙 명시) ===")
    res["M3"] = run(cl, a.model, M3, a.n, "M3", cli_first)
    print("=== M4 chain (단계분해·(c)=CLI?) ===")
    res["M4"] = run(cl, a.model, M4, a.n, "M4", cli_first, key="c")
    print("=== M5 consequence (dispute먼저→CLI승인? no=규칙적용) ===")
    res["M5"] = run(cl, a.model, M5, a.n, "M5", m5_blocked, key=None)
    print("\n== 집계 (CLI-먼저 or blocked 인지) ==")
    for k in ("M1", "M2", "M3", "M4", "M5"):
        h, n = res[k]
        print("  %s: %d/%d" % (k, h, n))
    print("판정: M2~M4 높음 → 규칙상기+분해로 순서 도출(scaffold) / 전부 낮음 → 적용추론 learn(crossover)")


if __name__ == "__main__":
    main()
