#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""★rate 측정 v2 — **곱셈까지 도구로** (무료·2026-07-18·`RATE_SUBAGENT_DESIGN`).

v1(promo_active만·날짜만 offload) 결과: F2/F3 통과(도구호출·신뢰)이나 **모델이 최종 곱셈서 새 실수**
  — 411 base 10→4 오독·506 프로모 없는데 ×2. ⇒ **곱셈도 빼라**(사용자). 분담선 한 칸 더:
  모델 = **텍스트 해석만**(base_rate 이름값 + 프로모 파라미터) · 엔진 = 날짜판정 **+ 곱셈** = 최종 배율 반환.
도구 = compute_reward_rate(base_rate_pct, has_promo, promo_mult, account_open, txn_date, window_months,
                           promo_start, promo_end) → 최종 배율(숫자). ★모델은 숫자를 곱하지 않는다.
모델의 유일 임무 = 각 거래에 이 도구를 **올바른 인자로** 호출하는 것. 반환값이 곧 정답.
정답: 403→20 · 410→10 · 411→10 · 506→4.

Run: python3 bank_rate_toolcall2_probe.py --base http://localhost:8141/v1 --n 8 [--force]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bank_fab_probes import post  # noqa: E402
from bank_rate_formalize_probe import GOLD, build_prompt, parse  # noqa: E402
from bank_rate_toolcall_probe import _d, _add_months  # noqa: E402

TOOLS = [{
    "type": "function",
    "function": {
        "name": "compute_reward_rate",
        "description": ("Deterministically compute the final reward-rate multiplier for a transaction. "
                        "You provide the base rate you read from the policy (as a percent number, e.g. 10 "
                        "for 10%), and the promo parameters; this tool decides whether the promo applies "
                        "(dates) and returns the FINAL multiplier. Do NOT multiply yourself — call this for "
                        "every transaction and use its returned number."),
        "parameters": {
            "type": "object",
            "properties": {
                "base_rate_pct": {"type": "number", "description": "base reward rate as a percent number, from policy"},
                "has_promo": {"type": "boolean", "description": "does a limited-time promo exist for this card?"},
                "promo_mult": {"type": "number", "description": "promo multiplier, e.g. 2 for double (ignored if has_promo false)"},
                "account_open": {"type": "string", "description": "account opening date MM/DD/YYYY"},
                "txn_date": {"type": "string", "description": "transaction date MM/DD/YYYY"},
                "window_months": {"type": "integer", "description": "promo window length in months"},
                "promo_start": {"type": "string", "description": "promo period start MM/DD/YYYY"},
                "promo_end": {"type": "string", "description": "promo period end MM/DD/YYYY"},
            },
            "required": ["base_rate_pct", "has_promo", "account_open", "txn_date"],
        },
    },
}]


def compute_reward_rate(base_rate_pct, has_promo=False, promo_mult=1, account_open=None, txn_date=None,
                        window_months=6, promo_start=None, promo_end=None, **_):
    """★결정론 엔진(우리가 담을 부분·곱셈 포함). 모델은 base_rate·has_promo만 준다; 곱셈은 안 한다."""
    base = float(base_rate_pct)
    if not has_promo:
        return base
    ao, td = _d(account_open), _d(txn_date)
    if ao is None or td is None:
        return base
    elig = True
    if promo_start and promo_end:
        ps, pe = _d(promo_start), _d(promo_end)
        elig = bool(ps and pe and ps <= ao <= pe)
    active = ao <= td <= _add_months(ao, int(window_months or 6))
    return base * float(promo_mult or 1) if (elig and active) else base


def run_one(base, model, temp, force, prompt):
    msgs = [{"role": "user", "content": prompt +
             "\n\nFor EVERY transaction you MUST call compute_reward_rate with the base rate you read from "
             "the policy and the promo parameters; do NOT compute or multiply the final rate yourself. "
             "The tool returns the final multiplier. After calling it for all transactions, give the final "
             "JSON mapping each transaction_id to the tool's returned number."}]
    calls = 0
    for _ in range(8):
        payload = {"model": model, "messages": msgs, "tools": TOOLS,
                   "temperature": temp, "max_tokens": 1500, "n": 1}
        if force and calls == 0:
            payload["tool_choice"] = "required"
        r = post(base, payload, timeout=300)
        m = r["choices"][0]["message"]
        tcs = m.get("tool_calls") or []
        if not tcs:
            return parse(m.get("content")), calls
        msgs.append({"role": "assistant", "content": m.get("content"), "tool_calls": tcs})
        for tc in tcs:
            calls += 1
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:
                args = {}
            res = compute_reward_rate(**args) if tc["function"]["name"] == "compute_reward_rate" else "unknown"
            msgs.append({"role": "tool", "tool_call_id": tc.get("id", "c"), "content": json.dumps(res)})
    return None, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8141/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--kb", default="/home/woori/scratch/kb_rate_docs.json")
    a = ap.parse_args()
    kb = json.load(open(a.kb, encoding="utf-8"))
    prompt = build_prompt(kb)
    print("★곱셈까지 도구 · force=%s · 정답 %s\n" % (a.force, GOLD))
    allc = 0
    percell = {k: 0 for k in GOLD}
    called = 0
    for i in range(a.n):
        try:
            out, calls = run_one(a.base, a.model, a.temp, a.force, prompt)
        except Exception as e:
            print("  [%d] ERR %r" % (i, str(e)[:70]))
            continue
        called += (calls > 0)
        if out is None:
            print("  [%d] 파싱실패 (tool %d회)" % (i, calls))
            continue
        hits = {k: (float(out[k]) == GOLD[k]) if out.get(k) is not None else False for k in GOLD}
        for k, v in hits.items():
            percell[k] += v
        ok = all(hits.values())
        allc += ok
        print("  [%d] %s tool%d회  %s" % (i, "✓ALL" if ok else "부분", calls,
                                          {k: out.get(k) for k in GOLD}))
    print("\n" + "=" * 60)
    print("★4/4 정확 = %d/%d = %.0f%%  (v1 날짜만 도구=부분·formalize-only=12%%)" % (allc, a.n, 100 * allc / max(a.n, 1)))
    print("★도구 호출 샘플 = %d/%d" % (called, a.n))
    print("★셀별:", {k: f"{percell[k]}/{a.n}" for k in GOLD})


if __name__ == "__main__":
    main()
