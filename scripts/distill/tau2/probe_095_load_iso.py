#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""095 부하-격리 프로브 (2026-07-20·등대 §1.4 load=p_iso−p_traj·사용자 지시).

질문: 095의 slot-formalize 실패(96000/1개월/0.46875/6.0 vs gold 8000/full-yr/5.625/6.85)가
**부하(load)**인가 **능력(INFER-calibration)**인가?
- p_traj(라이브) = 0/1 (e2e9 실측·4슬롯 전부 오답)
- p_iso = 이 프로브: **정보-맞춘 격리**(그 시점까지의 사용자 발화+도구 출력 전부·assistant 자기생성만 제거)
  에서 같은 slot-formalize를 n회 샘플 → 정답률.
판정: p_iso 높음 → 부하 → isolate 배선으로 닫힘(learn 불요). p_iso 낮음 → 능력 → learn 축 유지.
측정 규율(등대): 정보-빈약 프로브 금지 — gold 값들이 문맥에 실재하는지 선검사·미실재면 부하판정 무효.
비용: 무료(로컬 vLLM만·user-sim 0). Run: python probe_095_load_iso.py --n 8
"""
import argparse
import datetime
import json
import re
import sys

RES = "/home/woori/scratch/tau2-bench/data/simulations/bank_e2e9_a_20260720/results.json"
GOLD = {"principal": 8000.0, "actual_apy": 5.625, "expected_apy": 6.85, "amount": 98.00}


def build_context():
    d = json.load(open(RES))
    s = next(x for x in d["simulations"] if x["task_id"] == "task_095")
    ms = s["messages"]
    cut = next(i for i, m in enumerate(ms)
               for t in (m.get("tool_calls") or []) if t.get("name") == "get_interest_correction")
    parts = []
    for m in ms[:cut]:
        c = (m.get("content") or "").strip()
        if m.get("role") == "user" and c and not c.startswith("###"):
            parts.append("CUSTOMER: " + c)
        elif m.get("role") == "tool":
            parts.append("TOOL OUTPUT:\n" + c)
    return "\n\n".join(parts), cut


PROMPT = """You are a bank agent resolving a savings-interest discrepancy case. Below is ALL the \
information gathered so far in the conversation (customer statements and tool outputs), in order.

=== CASE FILE ===
{info}
=== END CASE FILE ===

Based ONLY on the case file, determine the values for the interest correction:
- principal: the principal balance of THE savings account in question (number, no $)
- period_start, period_end (MM/DD/YYYY): the period over which the WRONG APY was actually applied
- expected_apy: the correct APY under the stacking policy (number, percentage points)
- actual_apy: the APY that was actually applied to the account (number, percentage points)
- amount_difference: principal * (expected_apy-actual_apy)/100 * days/365, rounded to 2 decimals

Reply with exactly one JSON object and nothing else:
{{"principal": 0, "period_start": "", "period_end": "", "expected_apy": 0, "actual_apy": 0, "amount_difference": 0}}"""


def last_json(text):
    out = None
    for m in re.finditer(r"\{[^{}]*\}", text or "", re.S):
        try:
            out = json.loads(m.group(0))
        except Exception:
            pass
    return out


def days(a, b):
    def p(x):
        for f in ("%m/%d/%Y", "%Y-%m-%d"):
            try:
                return datetime.datetime.strptime(str(x).strip(), f)
            except Exception:
                pass
    da, db = p(a), p(b)
    return (db - da).days if da and db else None


def score(j):
    if not isinstance(j, dict):
        return {"parse": False}
    def num(k):
        try:
            return float(str(j.get(k)).replace(",", "").replace("$", ""))
        except Exception:
            return None
    pr, ea, aa, am = num("principal"), num("expected_apy"), num("actual_apy"), num("amount_difference")
    dd = days(j.get("period_start"), j.get("period_end"))
    return {"parse": True,
            "principal_ok": pr == GOLD["principal"],
            "expected_ok": ea is not None and abs(ea - GOLD["expected_apy"]) < 0.01,
            "actual_ok": aa is not None and abs(aa - GOLD["actual_apy"]) < 0.01,
            "period_full_year": dd is not None and dd >= 300,
            "amount_ok": am is not None and abs(am - GOLD["amount"]) <= 3.0,
            "raw": {"principal": pr, "expected": ea, "actual": aa, "days": dd, "amount": am}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--base", default="http://localhost:8141/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    a = ap.parse_args()
    info, cut = build_context()
    # ★정보-실재 선검사(등대 측정 규율): gold 값이 문맥에 없으면 '부하' 판정 무효(정보량 차이일 뿐).
    presence = {"8000": "8,000" in info or "8000" in info,
                "5.625": "5.625" in info,
                "checking_boost_3.5": "3.5" in info,
                "base_3.35_or_5.5": ("3.35" in info) or ("5.5" in info)}
    print("[probe] cut msg=%d ctx_chars=%d presence=%s" % (cut, len(info), presence))
    from openai import OpenAI
    cl = OpenAI(base_url=a.base, api_key="dummy")
    rows = []
    for i in range(a.n + 1):
        temp = 0.0 if i == 0 else 0.7
        r = cl.chat.completions.create(model=a.model, temperature=temp, max_tokens=400,
                                       messages=[{"role": "user", "content": PROMPT.format(info=info)}])
        sc = score(last_json(r.choices[0].message.content))
        sc["temp"] = temp
        rows.append(sc)
        print("[%d] t=%.1f %s" % (i, temp, {k: v for k, v in sc.items() if k != "raw"}), flush=True)
        if sc.get("parse"):
            print("    raw:", sc["raw"], flush=True)
    keys = ("principal_ok", "expected_ok", "actual_ok", "period_full_year", "amount_ok")
    print("\n== p_iso 집계 (n=%d) ==" % len(rows))
    for k in keys:
        print("  %s: %d/%d" % (k, sum(1 for r in rows if r.get(k)), len(rows)))
    full = sum(1 for r in rows if all(r.get(k) for k in keys))
    print("  ALL-slots: %d/%d   (p_traj 라이브=0/1·4슬롯 전오답)" % (full, len(rows)))


if __name__ == "__main__":
    main()
