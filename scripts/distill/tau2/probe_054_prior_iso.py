#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""054 prior-격리 프로브 (2026-07-22·사용자 지시 "054도 격리하라").

질문: 054의 잔여(approve의 env 히든룰 "pending dispute/replacement면 CLI 거부"·KB/도구/에러 전무)가
**진짜 learn/scale 경계(발견불가·prior 부재)**인가, 아니면 038처럼 **활성화 실패(상식-prior는 있으나
자유생성서 미발현)=scaffold(질의 게이트)**인가?
- p_traj(라이브·rall11): dispute 먼저 접수→CLI approve env거부→실패. 규칙 인지 흔적 없음.
- 이 프로브: 규칙 자체를 **명시 질문**해 모델이 상식으로 아는지 본다(격리·규칙은 여전히 미제공).
판정:
  C1(규칙 상식) 'wait/no' 높음 → **prior 있음 → 활성화 실패 → scaffold**(054도 learn 아님·crossover 뒤집힘).
  C1 낮음(yes-approve/모름) → **prior 없음 → 진짜 learn/scale 경계**(crossover 확정 강화).
  C2(순서)는 보조: CLI-먼저 선택률.
측정 규율: 규칙을 프롬프트에 심지 말 것(그러면 038형 KB-제공이 됨). 순수 상식 질문만.
비용: 무료(로컬 vLLM). Run: python probe_054_prior_iso.py --n 8
"""
import argparse
import json
import re

# C1: 규칙을 상식으로 아는가 (규칙 미제공·순수 은행-실무 상식 질문)
C1 = """A customer has an ACTIVE credit card account in good standing. Right now there is a PENDING \
fraud dispute open on one of their transactions, and a replacement card has been ordered for that \
account (shipped, not yet delivered/activated). The customer asks to increase the credit limit on \
that same account.

Under standard banking practice, should a credit limit increase be APPROVED right now, or should it \
WAIT until the dispute and replacement are resolved? Reply with exactly one JSON object and nothing \
else: {"decision": "approve_now" | "wait", "reason": "<one sentence>"}"""

# C2: 세 요청 순서 (규칙 힌트 없이·순서 최적화만)
C2 = """In a single call, a customer asks you to do THREE things on their credit card account:
(1) file a dispute for a fraudulent charge, (2) order a replacement card, (3) increase the credit \
limit. Some of these actions can put the account into a state that blocks other actions from \
completing. In what ORDER should you perform them to minimize the chance that any single request \
fails? Reply with exactly one JSON object and nothing else: \
{"order": ["...", "...", "..."], "reason": "<one sentence>"}"""


def last_json(text):
    out = None
    for m in re.finditer(r"\{.*\}", text or "", re.S):
        try:
            out = json.loads(m.group(0))
        except Exception:
            pass
    return out


def score_c1(j):
    if not isinstance(j, dict):
        return {"parse": False}
    dec = str(j.get("decision", "")).strip().lower()
    return {"parse": True, "wait": "wait" in dec, "approve_now": "approve" in dec,
            "raw": dec, "reason": str(j.get("reason", ""))[:120]}


def score_c2(j):
    if not isinstance(j, dict):
        return {"parse": False}
    order = j.get("order") or []
    order = [str(x).lower() for x in order] if isinstance(order, list) else []
    # CLI가 dispute·replacement보다 먼저인가
    def idx(kw):
        for i, o in enumerate(order):
            if kw in o:
                return i
        return 99
    cli, disp, repl = idx("limit"), idx("dispute"), idx("replacement")
    return {"parse": True, "cli_first": cli < disp and cli < repl,
            "order": order, "reason": str(j.get("reason", ""))[:120]}


def run(cl, model, prompt, scorer, n, label):
    rows = []
    for i in range(n + 1):
        temp = 0.0 if i == 0 else 0.7
        r = cl.chat.completions.create(model=model, temperature=temp, max_tokens=200,
                                       messages=[{"role": "user", "content": prompt}])
        sc = scorer(last_json(r.choices[0].message.content))
        rows.append(sc)
        print("  [%s %d] t=%.1f %s" % (label, i, temp,
              {k: v for k, v in sc.items() if k not in ("reason", "order", "raw")}), flush=True)
        if sc.get("reason"):
            print("       reason:", sc["reason"], flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--base", default="http://localhost:8141/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    a = ap.parse_args()
    from openai import OpenAI
    cl = OpenAI(base_url=a.base, api_key="dummy")
    print("=== C1: 규칙 상식(pending dispute/replacement → CLI approve? ·규칙 미제공) ===")
    r1 = run(cl, a.model, C1, score_c1, a.n, "C1")
    print("=== C2: 세 요청 순서(CLI-먼저 아는가) ===")
    r2 = run(cl, a.model, C2, score_c2, a.n, "C2")
    n1, n2 = len(r1), len(r2)
    print("\n== 집계 ==")
    print("  C1 wait(규칙 상식有): %d/%d   approve_now(모름): %d/%d"
          % (sum(1 for r in r1 if r.get("wait")), n1, sum(1 for r in r1 if r.get("approve_now")), n1))
    print("  C2 cli_first: %d/%d" % (sum(1 for r in r2 if r.get("cli_first")), n2))
    print("판정: C1 wait 높음 → prior有→활성화실패→scaffold / C1 낮음 → prior無→learn/scale 경계(crossover 확정)")


if __name__ == "__main__":
    main()
