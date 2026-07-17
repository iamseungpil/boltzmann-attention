#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""★레버 A를 **실측 give-missing 실패 5건 그 지점**에서 시험 (무료·로컬 vLLM·2026-07-18 사용자 지시).

대상 = nt=20 실측 give-missing(producer는 불렀는데 `give_discoverable_user_tool` 0회·reward 0):
  dreq2 trial 3·7·8·16 + ctl2 trial 3.
방법 = `_ap_regen`과 **동형** 재생: 궤적 전체(사임 포함) + FOLLOWUP 피드백(A2 원문) → 재샘플.
  arm A/B = 같은 입력·`tool_choice` 유/무 **단일변수**. 접두당 n=4(사용자 지시 nt=4).
판정 = 재샘플이 ①`give_discoverable_user_tool` 냈나(=레버 목적) ②빈손(산문)인가 ③다른 도구인가.
⚠️이건 결정점 재생이지 라이브가 아니다 — give 이후 사용자 실행·coverage는 안 본다(그건 라이브 arm).

Run(리모트): python3 bank_givemiss_replay.py --base http://localhost:8141/v1 --n 4
"""
import argparse
import gzip
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SIMDIR = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")

from bank_fab_probes import to_openai, load_tools, post, AGENT_INSTRUCTION  # noqa: E402

TARGETS = [("bank_dreq2_nt20_20260718", 3), ("bank_dreq2_nt20_20260718", 7),
           ("bank_dreq2_nt20_20260718", 8), ("bank_dreq2_nt20_20260718", 16),
           ("bank_ctl2_nt20_20260718", 3)]


def load_sim(tag, trial):
    with gzip.open(os.path.join(SIMDIR, f"{tag}.results.json.gz"), "rt", encoding="utf-8") as f:
        sims = json.load(f)["simulations"]
    return next(s for s in sims if s["trial"] == trial)


def classify(msg):
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return "빈손(산문)"
    names = [t["function"]["name"] for t in tcs]
    if "give_discoverable_user_tool" in names:
        return "★give"
    return "다른도구:" + names[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8141/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=4000)
    a = ap.parse_args()

    tools, policy, a2 = load_tools()
    fb = [d for d in a2["scaffold_get_tools"] if d["name"] == "get_reward_discrepancies"][0]["follow_up"]["feedback"]
    sysmsg = [{"role": "system", "content": AGENT_INSTRUCTION + "\n\n<policy>\n" + policy + "\n</policy>"}]

    grand = {"none": Counter(), "required": Counter()}
    for tag, trial in TARGETS:
        s = load_sim(tag, trial)
        conv = sysmsg + to_openai(s["messages"]) + [{"role": "user", "content": fb}]
        label = f"{tag[5:10]} t{trial:02d}"
        row = {}
        for arm, tc in (("none", None), ("required", "required")):
            c = Counter()
            for _ in range(a.n):
                payload = {"model": a.model, "messages": conv, "tools": tools,
                           "temperature": a.temp, "max_tokens": a.max_tokens, "n": 1}
                if tc:
                    payload["tool_choice"] = tc
                try:
                    r = post(a.base, payload, timeout=420)
                    c[classify(r["choices"][0]["message"])] += 1
                except Exception as e:
                    c["ERR:" + repr(e)[:40]] += 1
            row[arm] = c
            grand[arm].update(c)
        print(f"{label}  없음={dict(row['none'])}  |  required={dict(row['required'])}", flush=True)

    print("\n" + "=" * 64)
    for arm in ("none", "required"):
        g = grand[arm]
        tot = sum(g.values())
        print(f"★합계 [{arm:8s}] give={g.get('★give', 0)}/{tot}  빈손={g.get('빈손(산문)', 0)}  "
              f"기타={ {k: v for k, v in g.items() if k not in ('★give', '빈손(산문)')} }")
    print("판정: required가 give를 유의하게 올리고 '다른도구'로 새지 않으면 레버 A가 이 5건을 구한다.")


if __name__ == "__main__":
    main()
