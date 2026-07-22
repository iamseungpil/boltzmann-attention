#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""097 부하-격리 프로브 (2026-07-22·사용자 지시 "097=calc/부하 아닌가·격리하라").

질문: 097의 값-실패(principal 95000 vs 실제잔액 100000·Δapy 오형식화)가 **부하**인가 **능력**인가?
- p_traj(라이브·rall11) = principal 95000(오답)·apply 4계좌중 1개(95 vs gold 12)
- p_iso = 이 프로브: **정보-맞춘 격리**(get_all_accounts까지의 사용자발화+도구출력·assistant 자기생성 제거)
  에서 silver 계좌 interest-correction 입력(principal 등)을 n회 formalize → 정답률.
판정: p_iso 높음 → 부하 → SG_ISOLATE(fetch_formalize·subagent) 배선으로 닫힘(사용자 가설 지지·learn 불요).
      p_iso 낮음 → 능력 → formalize learn 잔여(LEARN §2.1).
측정 규율(등대): gold 값이 문맥에 실재하는지 선검사·미실재면 부하판정 무효.
비용: 무료(로컬 vLLM). Run: python probe_097_load_iso.py --n 8
"""
import argparse
import json
import re
import sys

RES = "/home/woori/scratch/tau2-bench/data/simulations/bank_rall11a_20260722/results.json"
# gold: silver 계좌 interest correction. principal=현재잔액 100000. amount 220.84(월할).
GOLD = {"principal": 100000.0, "amount": 220.84}


def build_context():
    d = json.load(open(RES))
    s = next(x for x in d["simulations"] if x["task_id"] == "task_097")
    ms = s["messages"]
    # cut = 첫 get_interest_correction 호출 직전(그 시점까지 정보 확보 완료)
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


PROMPT = """You are a bank agent resolving a savings-interest discrepancy case for the customer's \
SILVER account (account_id sav_mc80w7k3x9_silver). Below is ALL the information gathered so far in \
the conversation (customer statements and tool outputs), in order.

=== CASE FILE ===
{info}
=== END CASE FILE ===

Based ONLY on the case file, determine the interest-correction inputs for the SILVER account:
- principal: the principal balance of the SILVER savings account (number, no $ or commas)
- expected_apy: the correct APY under the stacking policy (percentage points)
- actual_apy: the APY actually applied to the account (percentage points)

Reply with exactly one JSON object and nothing else:
{{"principal": 0, "expected_apy": 0, "actual_apy": 0}}"""


def last_json(text):
    out = None
    for m in re.finditer(r"\{[^{}]*\}", text or "", re.S):
        try:
            out = json.loads(m.group(0))
        except Exception:
            pass
    return out


def score(j):
    if not isinstance(j, dict):
        return {"parse": False}
    def num(k):
        try:
            return float(str(j.get(k)).replace(",", "").replace("$", ""))
        except Exception:
            return None
    pr = num("principal")
    return {"parse": True,
            "principal_ok": pr == GOLD["principal"],
            "raw": {"principal": pr, "expected": num("expected_apy"), "actual": num("actual_apy")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--base", default="http://localhost:8141/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    a = ap.parse_args()
    info, cut = build_context()
    presence = {"100000": "100000" in info or "100,000" in info,
                "silver_id": "sav_mc80w7k3x9_silver" in info}
    print("[probe097] cut msg=%d ctx_chars=%d presence=%s" % (cut, len(info), presence))
    from openai import OpenAI
    cl = OpenAI(base_url=a.base, api_key="dummy")
    rows = []
    for i in range(a.n + 1):
        temp = 0.0 if i == 0 else 0.7
        r = cl.chat.completions.create(model=a.model, temperature=temp, max_tokens=300,
                                       messages=[{"role": "user", "content": PROMPT.format(info=info)}])
        sc = score(last_json(r.choices[0].message.content))
        sc["temp"] = temp
        rows.append(sc)
        print("[%d] t=%.1f %s" % (i, temp, {k: v for k, v in sc.items() if k != "raw"}), flush=True)
        if sc.get("parse"):
            print("    raw:", sc["raw"], flush=True)
    n = len(rows)
    print("\n== p_iso 집계 (n=%d) ==" % n)
    print("  principal_ok(=100000·잔액 복사): %d/%d   (p_traj 라이브=0·모델 95000)"
          % (sum(1 for r in rows if r.get("principal_ok")), n))


if __name__ == "__main__":
    main()
