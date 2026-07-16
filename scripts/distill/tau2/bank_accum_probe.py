#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""인자 누적(A2 설명 교정) A/B 격리 프로브 (무료·[[09]]·정보-맞춘 격리·등대 §1.4).

실패 지점: fup_nt4 sim1 msg[18] — 사용자가 phone([11])·dob([17])를 줬는데 에이전트가
`verify_identity(provided={name, dob})`로 phone을 버려 NOT_VERIFIED (§16.3 신규 유형: 턴 간 union 실패).

A/B (단일 변수 = provided 설명):
  OLD: "JSON object of the identity values the CUSTOMER gave you, using these keys where available: ..."
  NEW: "... ALL identity values ... at ANY point in this conversation — accumulate across turns ..."
측정: verify_identity emit 중 provided가 {phone_number AND date_of_birth} 포함(union-정답) 비율.
판정: NEW서 union이 높으면 A2 교정으로 닫힘(온톨로지 불요·사용자 기준). 안 높으면 온톨로지/WM 설계로.

Run (리모트·8141):
  python3 bank_accum_probe.py --base http://localhost:8141/v1 --n 20
"""
import argparse
import gzip
import json
import os
import sys
import urllib.request
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results",
                   "bank_fup_20260716_nt4.results.json.gz")
A2P = os.path.join(HERE, "a2", "banking_knowledge.gate.json")

OLD_DESC = ("JSON object of the identity values the CUSTOMER gave you, using these keys where "
            "available: date_of_birth, address, phone_number, email.")

AGENT_INSTRUCTION = (
    "You are a customer service agent that helps the user according to the <policy> provided below.\n"
    "In each turn you can either:\n- Send a message to the user.\n- Make a tool call.\n"
    "You cannot do both at the same time.\n\n"
    "Try to be helpful and always follow the policy. Always make sure you generate valid JSON only."
)


def post(base, payload):
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def load_tools(provided_desc):
    """env 도구 + A2 주입 도구 — 라이브와 동일 스키마·provided 설명만 파라미터화."""
    from tau2.registry import registry
    import t2_scaffold_get as sg
    from tau2.environment.tool import Tool
    env = registry.get_env_constructor("banking_knowledge")()
    a2 = json.load(open(A2P, encoding="utf-8"))
    tools = [t.openai_schema for t in env.get_tools()]
    for d in a2["scaffold_get_tools"]:
        d2 = json.loads(json.dumps(d))
        if d2["name"] == "verify_identity":
            d2["params"]["provided"] = provided_desc
        tools.append(sg._build_tool(Tool, d2).openai_schema)
    return tools, env.get_policy()


def to_openai(msgs):
    out = []
    for m in msgs:
        role, content = m.get("role"), m.get("content")
        tcs = m.get("tool_calls") or []
        if role == "assistant":
            e = {"role": "assistant", "content": content or None}
            if tcs:
                e["tool_calls"] = [{"id": tc.get("id") or f"c{i}", "type": "function",
                                    "function": {"name": tc.get("name"),
                                                 "arguments": json.dumps(tc.get("arguments") or {})}}
                                   for i, tc in enumerate(tcs)]
            out.append(e)
        elif role == "user":
            if not tcs and content:
                out.append({"role": "user", "content": content})
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("id") or "c0", "content": content or ""})
    return out


def classify(name, args):
    if name != "verify_identity":
        return f"other:{name}" if name else "text"
    p = args.get("provided") if isinstance(args, dict) else None
    if isinstance(p, str):
        try:
            p = json.loads(p)
        except Exception:
            return "verify:unparsable"
    keys = set(p or {})
    has_phone = "phone_number" in keys
    has_dob = "date_of_birth" in keys
    if has_phone and has_dob:
        return "★verify:UNION(phone+dob)"
    if has_dob and not has_phone:
        return "verify:dob-only(직전턴만)"
    if has_phone and not has_dob:
        return "verify:phone-only"
    return f"verify:{sorted(keys)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8141/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--temp", type=float, default=0.7)
    a = ap.parse_args()

    with gzip.open(SIM, "rt", encoding="utf-8") as f:
        d = json.load(f)
    msgs = d["simulations"][1]["messages"]
    # 절단: dob를 준 사용자 발화([17])까지 포함 — 그 직후가 실패 결정점
    cut = None
    for i, m in enumerate(msgs):
        if (m.get("role") == "user" and isinstance(m.get("content"), str)
                and "11/03/1990" in m["content"] and not (m.get("tool_calls") or [])):
            cut = i
    assert cut is not None, "dob 발화 못 찾음"
    prefix = msgs[:cut + 1]
    print(f"접두 = msgs[0..{cut}] (phone은 그 이전 턴서 이미 제공됨)")

    new_desc = json.load(open(A2P, encoding="utf-8"))
    new_desc = [t for t in new_desc["scaffold_get_tools"] if t["name"] == "verify_identity"][0]["params"]["provided"]
    assert "accumulate" in new_desc, "A2가 신판이 아님"

    for tag, desc in (("OLD(교정 전)", OLD_DESC), ("NEW(누적 명시)", new_desc)):
        tools, policy = load_tools(desc)
        sysmsg = [{"role": "system", "content": AGENT_INSTRUCTION + "\n\n<policy>\n" + policy + "\n</policy>"}]
        conv = sysmsg + to_openai(prefix)
        cnt = Counter()
        for _ in range(a.n):
            try:
                r = post(a.base, {"model": a.model, "messages": conv, "tools": tools,
                                  "temperature": a.temp, "max_tokens": 700})
                m = r["choices"][0]["message"]
                tcs = m.get("tool_calls") or []
                if tcs:
                    f0 = tcs[0]["function"]
                    try:
                        args = json.loads(f0.get("arguments") or "{}")
                    except Exception:
                        args = {}
                    cnt[classify(f0["name"], args)] += 1
                else:
                    cnt["text"] += 1
            except Exception as e:
                cnt["__ERR__"] += 1
                print("  err:", repr(e)[:120], file=sys.stderr)
        print(f"\n===== {tag} (n={a.n}, T={a.temp}) =====")
        for k, v in cnt.most_common():
            print(f"  {v:3d}  {k}")
        union = sum(v for k, v in cnt.items() if k.startswith("★"))
        print(f"  ★union율: {union}/{a.n} = {100*union/a.n:.0f}%")


if __name__ == "__main__":
    main()
