#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""052 approve verdict-grounding 라우팅 격리 (2026-07-22·greenlight_iso 동형·[[08]]/[[05]]).
rall14 052 궤적을 approve 호출 직후에서 얼려, 새 게이트의 block 피드백을 주입하고 다음 턴 재샘플.
판정: 에이전트가 check_cli_eligibility(옳은 라우팅)로 가나 vs re-approve/deny/prose.
게이트 로직은 test_cli_eligibility(W14-17)서 확정 — 이건 에이전트 응답 격리.
Run: python3 bank_052_approve_iso.py --base http://localhost:8140/v1 --n 8
"""
import argparse, gzip, json, os, sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
SIMDIR = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
from bank_fab_probes import to_openai, load_tools, post, AGENT_INSTRUCTION  # noqa

def load_sim(tag, tid):
    with gzip.open(os.path.join(SIMDIR, f"{tag}.results.json.gz"), "rt", encoding="utf-8") as f:
        return next(s for s in json.load(f)["simulations"] if s["task_id"] == tid)

def approve_idx(msgs):
    for i, m in enumerate(msgs):
        for tc in (m.get("tool_calls") or []):
            a = tc.get("arguments") or (tc.get("function") or {}).get("arguments") or {}
            if isinstance(a, str):
                try: a = json.loads(a)
                except: a = {}
            if isinstance(a, dict) and str(a.get("agent_tool_name", "")).startswith("approve_credit_limit_increase"):
                return i
    raise RuntimeError("approve call not found")

def classify(msg):
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return "산문(no-call)"
    names = []
    for tc in tcs:
        nm = tc["function"]["name"]
        a = json.loads(tc["function"].get("arguments") or "{}")
        eff = a.get("agent_tool_name")
        names.append(eff or nm)
    if any(n == "check_cli_eligibility" for n in names):
        return "★check_cli_eligibility"
    if any(str(n).startswith("deny_credit_limit_increase") for n in names):
        return "deny"
    if any(str(n).startswith("approve_credit_limit_increase") for n in names):
        return "re-approve(재시도)"
    if any(str(n).startswith("unlock") and "check_cli" in str(n) for n in names):
        return "unlock:check_cli"
    return "다른도구:" + str(names[0])

def run(base, model, conv, tools, temp, n):
    c = Counter()
    for _ in range(n):
        try:
            r = post(base, {"model": model, "messages": conv, "tools": tools,
                            "temperature": temp, "max_tokens": 500, "n": 1}, timeout=420)
            c[classify(r["choices"][0]["message"])] += 1
        except Exception as e:
            c["ERR:" + repr(e)[:40]] += 1
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()

    tools, policy, a2 = load_tools()
    sysmsg = [{"role": "system", "content": AGENT_INSTRUCTION + "\n\n<policy>\n" + policy + "\n</policy>"}]
    s = load_sim("bank_rall14_20260722", "task_052")
    msgs = s["messages"]
    ai = approve_idx(msgs)
    ID = "cc_5e4c1a83b0_bronze"
    # 게이트 block 피드백 = 실제 A2 approve-verdict WEV feedback (shipped 문구)
    wev = [w for w in a2["write_evidence_specs"]
           if (w.get("applies_when") or {}).get("prefix") == "approve_credit_limit_increase"
           and "ELIGIBLE - all tier requirements" in (w.get("require_tokens") or [])][0]
    block_fb = wev["feedback"].replace("{id}", ID)
    orig_ok = None
    for m in msgs[ai + 1:]:
        if m.get("role") == "tool":
            orig_ok = m.get("content"); break

    frozen = sysmsg + to_openai(msgs[:ai + 1])   # up to & incl the approve call
    base_conv = frozen + [{"role": "tool", "tool_call_id": "c_ap", "content": orig_ok or "approved"}]
    gate_conv = frozen + [{"role": "tool", "tool_call_id": "c_ap", "content": block_fb}]
    print(f"frozen at approve idx={ai}; block_fb[:80]={block_fb[:80]!r}", flush=True)

    for name, conv in (("A_asis(approve-success)", base_conv), ("B_gate(block-fb)", gate_conv)):
        c0 = run(a.base, a.model, conv, tools, 0.0, 1)
        c7 = run(a.base, a.model, conv, tools, 0.7, a.n)
        print(f"\n[{name}] temp0={dict(c0)}\n         temp0.7(n={a.n})={dict(c7)}", flush=True)
    print("\n판정: B_gate가 check_cli_eligibility로 라우팅하면 → 게이트가 직접산수를 도구로 돌림 확정.", flush=True)

if __name__ == "__main__":
    main()
