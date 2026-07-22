#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""038 처방-선택 부하 프로브 (2026-07-22·사용자 지시 "038도 부하문제일 수 있다·격리하라").

질문: 038의 처방-오선택(사기청구 dispute 요청에 apply_statement_credit 오선택)이
**부하(다중 요청·긴 대화가 판단을 흐림)**인가 **능력(단일 요청도 오선택)**인가?
- p_traj(라이브·rall11) = statement_credit 오선택(dispute 미착수·gold=file_dispute)
- 3개 부하 수준에서 "이 상황에 맞는 도구는?"을 n회 샘플 → file_dispute 선택률.
  L0(최소·격리): KB 처방 규칙 + 사기청구 한 줄 → 도구 선택
  L1(단일요청): dispute 요청 1건만의 대화 컨텍스트
  L2(라이브·다중): 038 실제 컨텍스트(처방 결정 시점까지·다중 요청·portal 혼란 포함)
판정: L0/L1 높고 L2 낮음 → 부하(격리/controller로 닫힘·사용자 가설 지지).
      전 수준 낮음 → 능력(처방 formalize learn 잔여·LEARN §2.1).
비용: 무료(로컬 vLLM). Run: python probe_038_prescription_load.py --n 8
"""
import argparse
import json
import re

RES = "/home/woori/scratch/tau2-bench/data/simulations/bank_rall11a_20260722/results.json"

# KB 처방 규칙 요약(doc_credit_cards_(general)_014 dispute·doc_017 statement_credit·logistics closure)
KB = """KNOWLEDGE BASE (prescription rules):
- To formally dispute a credit card transaction (unauthorized/fraudulent charge, merchant issue,
  billing error), use the tool file_credit_card_transaction_dispute.
- apply_statement_credit is ONLY for goodwill adjustments, promotional credits, fee reversals, or
  other account credits - NOT for disputing a charge.
- close_credit_card_account permanently closes the account (destructive; only when the customer
  explicitly wants to close AND eligibility is met).
- order_replacement_credit_card issues a new card number (for lost/stolen/fraud/damaged)."""

L0 = KB + """

SITUATION: The customer reports an Amazon charge for $89.99 that they did not make at all (they
call it unauthorized/fraudulent) and want it disputed.

Which single tool is the correct FIRST action to handle this request? Reply with exactly one JSON
object and nothing else: {"tool": "<tool_name>"}"""


def build_live_context():
    d = json.load(open(RES))
    s = next(x for x in d["simulations"] if x["task_id"] == "task_038")
    ms = s["messages"]
    # cut = 첫 apply_statement_credit/file_dispute/unlock 직전(처방 결정 시점)
    cut = None
    for i, m in enumerate(ms):
        for t in (m.get("tool_calls") or []):
            nm = t.get("name", "")
            arg = json.dumps(t.get("arguments", {}))
            if "apply_statement_credit" in arg or "file_credit_card_transaction_dispute" in arg:
                cut = i
                break
        if cut is not None:
            break
    parts = []
    for m in ms[:cut]:
        c = (m.get("content") or "").strip()
        if m.get("role") == "user" and c and not c.startswith("###"):
            parts.append("CUSTOMER: " + c)
        elif m.get("role") == "tool":
            parts.append("TOOL OUTPUT:\n" + c[:400])
    return "\n\n".join(parts), cut


L2_TMPL = KB + """

Below is the conversation so far.
=== CASE FILE ===
{info}
=== END CASE FILE ===

The customer wants the Amazon charge(s) disputed. Which single tool is the correct FIRST action?
Reply with exactly one JSON object and nothing else: {{"tool": "<tool_name>"}}"""


def last_json(text):
    out = None
    for m in re.finditer(r"\{[^{}]*\}", text or "", re.S):
        try:
            out = json.loads(m.group(0))
        except Exception:
            pass
    return out


def is_dispute(j):
    if not isinstance(j, dict):
        return None
    t = str(j.get("tool", "")).lower()
    if "file_credit_card_transaction_dispute" in t or "file_dispute" in t or t == "file_credit_card_transaction_dispute":
        return True
    return False


def run_condition(cl, model, prompt, n, label):
    hits, parsed = 0, 0
    for i in range(n):
        temp = 0.0 if i == 0 else 0.7
        r = cl.chat.completions.create(model=model, temperature=temp, max_tokens=120,
                                       messages=[{"role": "user", "content": prompt}])
        j = last_json(r.choices[0].message.content)
        d = is_dispute(j)
        if j is not None:
            parsed += 1
        if d:
            hits += 1
        print("  [%s %d] t=%.1f -> %s (dispute=%s)" % (label, i, temp,
              (j or {}).get("tool") if isinstance(j, dict) else None, d), flush=True)
    print("== %s: file_dispute %d/%d (parsed %d)" % (label, hits, n, parsed), flush=True)
    return hits, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--base", default="http://localhost:8141/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    a = ap.parse_args()
    from openai import OpenAI
    cl = OpenAI(base_url=a.base, api_key="dummy")
    info, cut = build_live_context()
    print("[probe038] live cut msg=%d ctx_chars=%d" % (cut, len(info)))
    print("\n=== L0 (최소·격리: KB규칙+사기청구 한줄) ===")
    run_condition(cl, a.model, L0, a.n, "L0")
    print("\n=== L2 (라이브·다중요청 컨텍스트) ===")
    run_condition(cl, a.model, L2_TMPL.format(info=info), a.n, "L2")
    print("\n판정: L0 높고 L2 낮음 → 부하 / 둘 다 낮음 → 능력(처방 formalize)")


if __name__ == "__main__":
    main()
