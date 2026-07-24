#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BRANCH-REGROUND make-or-break (엔진판·multi-step·2026-07-24·C144 후속).
test_reground_043.py(하드코딩 doc·single-next ASK)의 상위:
  · 리마인더를 **엔진**(t2_eplan_patch.branch_reground_reminder)이 생성 — 정책문서를
    transcript서 추출(하드코딩 아님) → 라이브 배선과 동형(도메인 리터럴 0 실증).
  · **multi-step ASK**("남은 도구 전부 순서대로 나열") — single-next 아티팩트(R_both가
    하나만 고르던 C144 문제) 제거 → apply_flag AND dispute 둘 다 나오나 직접 측정.
조건(ablation):
  R_none     : 오염맥락만(baseline)
  R_chain    : + 기존 chain_reminder(compact 이름만·정책문서 X·C136 동형)
  R_reground : + branch_reground_reminder(엔진·read=이름/write=실제 정책문서)
기대(C144): apply_flag = R_none≈0 < R_chain(불충분) < R_reground(문서 회복)·dispute는 이름만으로도.
Run(remote): seka python test_reground_live_043.py --base http://localhost:8140/v1 --n 5
"""
import argparse
import gzip
import json
import os
import re
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SIMR = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")

import t2_eplan_patch as ep  # 엔진(같은 디렉토리)

ASK_MULTI = ("List, in order, EVERY remaining tool call you must still make before you "
             "finalize or close this account. For each, name the exact tool. Do not skip any "
             "required step.")


def load043():
    d = json.load(gzip.open(os.path.join(SIMR, "bank_rall25a_20260724.results.json.gz")))
    for s in d.get("simulations", []):
        if str(s.get("task_id")) == "task_043":
            return s.get("messages", [])
    raise SystemExit("043 not found")


def cut_before_close(msgs):
    """분기점 = 첫 close 시도(디스패처 unwrap) 직전까지 메시지 컷."""
    out = []
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            if "close_credit_card_account" in str(tc.get("arguments", "")):
                return out
        out.append(m)
    return out


def render(msgs, max_chars=32000):
    """오염 맥락 텍스트(test_reground_043.polluted 동형·에이전트 tool_call/에러 포함)."""
    lines = []
    for m in msgs:
        role, c = m.get("role"), str(m.get("content") or "")
        for tc in (m.get("tool_calls") or []):
            lines.append("[%s->tool] %s %s"
                         % (role, tc.get("name"), str(tc.get("arguments"))[:180]))
        if c.strip():
            lines.append("[%s] %s" % (role, c.strip()[:400]))
    return "\n".join(lines)[:max_chars]


def call(base, model, ctx, temp, headers):
    r = requests.post(base + "/chat/completions", json={
        "model": model, "messages": [
            {"role": "system",
             "content": "You are a precise banking assistant continuing the conversation."},
            {"role": "user", "content": ctx}],
        "temperature": temp, "max_tokens": 1200}, headers=headers, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def hits(txt):
    t = txt.lower()
    fl = ("apply_credit_card_account_flag" in t or "annual_fee_waived" in t
          or ("apply" in t and "flag" in t and "fee" in t))
    dh = "get_user_dispute_history" in t or "dispute history" in t
    return fl, dh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--provider", default="vllm", choices=["vllm", "openrouter"])
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    base, headers = a.base, {}
    if a.provider == "openrouter":
        base = "https://openrouter.ai/api/v1"
        headers = {"Authorization": "Bearer " + os.environ.get("OPENROUTER_API_KEY", "")}

    full = load043()
    cut = cut_before_close(full)
    spec = ep.load_eplan_spec("banking_knowledge")
    chain = ep.chain_gap(cut, spec)
    print("branch cut: %d msgs (of %d) | chain missing_reads=%s missing_writes=%s"
          % (len(cut), len(full), chain and chain["missing_reads"],
             chain and chain["missing_writes"]))
    r_chain = ep.chain_reminder(chain)
    r_reg = ep.branch_reground_reminder(chain, cut, spec)
    # 엔진이 실제 정책문서를 붙였는지 확증(도메인 리터럴 0 실증)
    print("reground reminder len=%d | has POLICY block=%s | names apply_flag policy=%s"
          % (len(r_reg), "[POLICY" in r_reg,
             "annual_fee_waived" in r_reg or "Retention Protocol" in r_reg))

    poll = render(cut)
    conds = {
        "R_none":     poll + "\n\n" + ASK_MULTI,
        "R_chain":    poll + "\n\n" + r_chain + "\n\n" + ASK_MULTI,
        "R_reground": poll + "\n\n" + r_reg + "\n\n" + ASK_MULTI,
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
        print("== %-11s apply_flag %d/%d  dispute_history %d/%d  (ctx=%d)"
              % (label, fl, tot, dh, tot, len(ctx)))


if __name__ == "__main__":
    main()
