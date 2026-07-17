#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""★rate-formalize + 날짜도구 function-calling 측정 (무료·2026-07-18·`RATE_SUBAGENT_DESIGN §6-2`).

가설(§0 측정): base_rate 100% · 날짜 만료산술 12%. ⇒ 날짜산술만 `promo_active` 도구로 빼면 4/4?
측정 = F2(모델이 도구를 부르나) · F3(도구 bool을 믿나).
  도구 = promo_active(account_open, txn_date, window_months, promo_start, promo_end) → bool  [결정론·엔진몫]
  모델 = base_rate·프로모 파라미터 formalize + 도구호출 + 결과로 최종 rate.
정답 배율: 403→20 · 410→10 · 411→10 · 506→4 (개설 02/13·6mo=08/13).
⚠️엔진 리터럴 0(측정 코드). 도구는 결정론 정답을 반환(우리가 담을 부분의 mock)·rate는 모델이 낸다.

Run: python3 bank_rate_toolcall_probe.py --base http://localhost:8141/v1 --n 8 [--force]
"""
import argparse
import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bank_fab_probes import post  # noqa: E402
from bank_rate_formalize_probe import TXNS, GOLD, ACCOUNT_OPEN, build_prompt, parse  # noqa: E402

TOOLS = [{
    "type": "function",
    "function": {
        "name": "promo_active",
        "description": ("Deterministically decide whether a limited-time promo multiplier applies to a "
                        "transaction. Returns true only if the account was opened within the promo period "
                        "AND the transaction date is within `window_months` of the account opening date. "
                        "Call this instead of computing the dates yourself."),
        "parameters": {
            "type": "object",
            "properties": {
                "account_open": {"type": "string", "description": "account opening date MM/DD/YYYY"},
                "txn_date": {"type": "string", "description": "transaction date MM/DD/YYYY"},
                "window_months": {"type": "integer", "description": "promo window length in months"},
                "promo_start": {"type": "string", "description": "promo period start MM/DD/YYYY"},
                "promo_end": {"type": "string", "description": "promo period end MM/DD/YYYY"},
            },
            "required": ["account_open", "txn_date", "window_months"],
        },
    },
}]


def _d(x):
    for f in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(x).split()[0], f)
        except Exception:
            pass
    return None


def _add_months(d, m):
    y, mo = d.year + (d.month - 1 + m) // 12, (d.month - 1 + m) % 12 + 1
    from calendar import monthrange
    return d.replace(year=y, month=mo, day=min(d.day, monthrange(y, mo)[1]))


def promo_active(account_open, txn_date, window_months, promo_start=None, promo_end=None, **_):
    """★결정론 엔진(우리가 담을 부분의 정답 구현). 모델이 이걸 부르고 결과를 믿는지가 측정 대상."""
    ao, td = _d(account_open), _d(txn_date)
    if ao is None or td is None:
        return False
    elig = True
    if promo_start and promo_end:
        ps, pe = _d(promo_start), _d(promo_end)
        elig = bool(ps and pe and ps <= ao <= pe)
    active = ao <= td <= _add_months(ao, int(window_months))
    return bool(elig and active)


def run_one(base, model, temp, force, prompt):
    msgs = [{"role": "user", "content": prompt +
             "\n\nWhen a promo may apply, you MUST call the promo_active tool to decide the date part; "
             "do not compute the dates yourself. Then give the final JSON of multipliers."}]
    calls = 0
    for _ in range(6):                      # 도구 왕복 상한
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
            res = promo_active(**args) if tc["function"]["name"] == "promo_active" else "unknown tool"
            msgs.append({"role": "tool", "tool_call_id": tc.get("id", "c"),
                         "content": json.dumps(res)})
    return None, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8141/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--force", action="store_true", help="첫 턴 tool_choice=required (레버 A)")
    ap.add_argument("--kb", default="/home/woori/scratch/kb_rate_docs.json")
    a = ap.parse_args()
    kb = json.load(open(a.kb, encoding="utf-8"))
    prompt = build_prompt(kb)
    print("★날짜도구 제공 · force=%s · 정답 %s\n" % (a.force, GOLD))
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
            print("  [%d] 파싱실패 (tool호출 %d회)" % (i, calls))
            continue
        hits = {k: (float(out[k]) == GOLD[k]) if out.get(k) is not None else False for k in GOLD}
        for k, v in hits.items():
            percell[k] += v
        ok = all(hits.values())
        allc += ok
        print("  [%d] %s tool%d회  %s" % (i, "✓ALL" if ok else "부분", calls,
                                          {k: out.get(k) for k in GOLD}))
    print("\n" + "=" * 60)
    print("★4/4 정확 = %d/%d = %.0f%%  (formalize-only 기준선 12%%)" % (allc, a.n, 100 * allc / max(a.n, 1)))
    print("★도구 호출한 샘플 = %d/%d (F2)" % (called, a.n))
    print("★셀별:", {k: f"{percell[k]}/{a.n}" for k in GOLD}, " ←410·411이 오르면 F3 통과")


if __name__ == "__main__":
    main()
