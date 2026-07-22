#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""054 time_verified 부하-격리 프로브 (2026-07-22·사용자 지시 "격리 필요하면 격리부터").

질문: 054 log_verification의 time_verified 오선택('2023-11-14 15:20' vs gold get_current_time
'2025-11-14 03:40:00 EST')이 **부하**인가 **능력**인가? (2023-11-14·15:20 모두 컨텍스트에 실재
=여러 실재값 중 오선택=097 principal 동형·WRITE_ARG_GROUND는 통과=격리로만 판별).
- p_traj(라이브·rall12)=오답(2023-11-14 15:20).
- p_iso=정보-맞춘 격리(log_verification 직전까지·assistant 자기생성 제거)에서 time_verified formalize n회.
판정: p_iso 높음→부하→서브분리/SG_ISOLATE로 닫힘(사용자 서브분리 제안 지지)·낮음→능력(formalize learn).
비용: 무료(로컬 vLLM). Run: python probe_054_time_iso.py --n 8
"""
import argparse
import json
import re

RES = "/home/woori/scratch/tau2-bench/data/simulations/bank_rall12b_20260722/results.json"
GOLD_TIME = "2025-11-14 03:40:00 EST"


def build_context():
    d = json.load(open(RES))
    s = next(x for x in d["simulations"] if x["task_id"] == "task_054")
    ms = s["messages"]
    cut = next(i for i, m in enumerate(ms)
               for t in (m.get("tool_calls") or []) if t.get("name") == "log_verification")
    parts = []
    for m in ms[:cut]:
        c = (m.get("content") or "").strip()
        if m.get("role") == "user" and c and not c.startswith("###"):
            parts.append("CUSTOMER: " + c)
        elif m.get("role") == "tool":
            parts.append("TOOL OUTPUT:\n" + c[:400])
    return "\n\n".join(parts), cut


PROMPT = """You are a bank agent about to log an identity verification. Below is the information \
gathered so far (customer statements and tool outputs).

=== CASE FILE ===
{info}
=== END CASE FILE ===

Determine ONE field for the verification log:
- time_verified: the CURRENT time at which verification is happening, taken from the get_current_time \
tool output (format exactly as that tool returned it).

Reply with exactly one JSON object and nothing else:
{{"time_verified": ""}}"""


def last_json(text):
    out = None
    for m in re.finditer(r"\{[^{}]*\}", text or "", re.S):
        try:
            out = json.loads(m.group(0))
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--base", default="http://localhost:8141/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    a = ap.parse_args()
    info, cut = build_context()
    presence = {"gold_2025-11-14_03:40": "2025-11-14 03:40" in info,
                "distractor_2023-11-14": "2023-11-14" in info, "distractor_15:20": "15:20" in info}
    print("[probe054t] cut=%d ctx_chars=%d presence=%s" % (cut, len(info), presence))
    from openai import OpenAI
    cl = OpenAI(base_url=a.base, api_key="dummy")
    ok = 0
    for i in range(a.n + 1):
        temp = 0.0 if i == 0 else 0.7
        r = cl.chat.completions.create(model=a.model, temperature=temp, max_tokens=100,
                                       messages=[{"role": "user", "content": PROMPT.format(info=info)}])
        j = last_json(r.choices[0].message.content)
        v = str((j or {}).get("time_verified", ""))
        hit = "2025-11-14 03:40" in v
        if hit:
            ok += 1
        print("[%d] t=%.1f time=%r ok=%s" % (i, temp, v, hit), flush=True)
    print("\n== p_iso: time_verified=get_current_time(2025-11-14 03:40): %d/%d (p_traj 라이브=0) ==" % (ok, a.n + 1))
    print("판정: 높음→부하(서브분리/SG_ISOLATE)·낮음→능력(formalize)")


if __name__ == "__main__":
    main()
