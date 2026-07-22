#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""050 KB_search 반복-flail 격리 (2026-07-23·greenlight/approve_iso 동형·[[08]]).
rall15 050 [60-76]: 에이전트가 KB_search_bm25{"query":"get_pending_replacement_orders"}(도구名)를
8회 반복 → [DUPLICATE-READ] 무시·temp0 루프. 기존 dedup 피드백=대안 미제시.
격리: 2번째 동일 KB_search 직후에서 얼려, dedup 피드백 vs 강한 redirect 피드백 → 다음 턴.
판정: redirect가 다른 행동(plain-words 검색/unlock/call/proceed)을 유도하나.
Run: python3 bank_050_flail_iso.py --base http://localhost:8140/v1 --n 8
"""
import argparse, gzip, json, os, sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
SIMDIR = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
from bank_fab_probes import to_openai, load_tools, post, AGENT_INSTRUCTION  # noqa

DEDUP = ("[DUPLICATE-READ] This exact call (same tool, same arguments) was already executed earlier "
         "in this conversation; its full output is shown above and has not changed. Refer to that "
         "output instead of re-reading.")
# 강한 redirect (도메인-일반: 반복 탐지 + 대안 행동 명시·특정 도구名 하드코딩 없음)
REDIRECT = ("[REPEATED-CALL] You have already issued this EXACT tool call with these EXACT arguments "
            "earlier; it returns the same result and will not progress. Do NOT issue it again. Take a "
            "DIFFERENT action: if you are searching the knowledge base for a discoverable tool, search "
            "with PLAIN WORDS describing the step (not the tool's function name) — e.g. words a policy "
            "document would use; if the tool is already unlocked, call it directly; if you already have "
            "the information you need, proceed to the next step of the task.")

def load_sim(tag, tid):
    with gzip.open(os.path.join(SIMDIR, f"{tag}.results.json.gz"), "rt", encoding="utf-8") as f:
        return next(s for s in json.load(f)["simulations"] if s["task_id"] == tid)

def second_dup_kbsearch_idx(msgs):
    """index of the assistant msg making the 2nd identical KB_search (pending_replacement)."""
    seen = 0
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            nm = tc.get("name") or (tc.get("function") or {}).get("name")
            a = tc.get("arguments") or (tc.get("function") or {}).get("arguments") or {}
            if isinstance(a, str):
                try: a = json.loads(a)
                except: a = {}
            if nm == "KB_search_bm25" and "pending_replacement" in json.dumps(a):
                seen += 1
                if seen == 2:
                    return i
    raise RuntimeError("2nd KB_search not found")

def classify(msg):
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return "산문(no-call)"
    out = []
    for tc in tcs:
        nm = tc["function"]["name"]; a = json.loads(tc["function"].get("arguments") or "{}")
        eff = a.get("agent_tool_name")
        if nm == "KB_search_bm25":
            q = str(a.get("query", ""))
            out.append("KBsearch-SAME(name)" if "pending_replacement" in q or "get_" in q else f"KBsearch-PLAIN:{q[:24]}")
        elif nm == "shell":
            c = str(a.get("command", ""))
            out.append("shell-SAME" if "get_pending" in c else "shell-other")
        elif eff:
            out.append(("★unlock:" if "unlock" in nm else "★call:") + str(eff))
        else:
            out.append(nm)
    return " | ".join(out)

def run(base, model, conv, tools, temp, n):
    c = Counter()
    for _ in range(n):
        try:
            r = post(base, {"model": model, "messages": conv, "tools": tools,
                            "temperature": temp, "max_tokens": 400, "n": 1}, timeout=420)
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
    s = load_sim("bank_rall15_20260722", "task_050")
    msgs = s["messages"]
    idx = second_dup_kbsearch_idx(msgs)
    frozen = sysmsg + to_openai(msgs[:idx + 1])
    print(f"frozen at 2nd KB_search idx={idx}", flush=True)
    for name, fb in (("A_dedup", DEDUP), ("B_redirect", REDIRECT)):
        conv = frozen + [{"role": "tool", "tool_call_id": "c_kb", "content": fb}]
        c0 = run(a.base, a.model, conv, tools, 0.0, 1)
        c7 = run(a.base, a.model, conv, tools, 0.7, a.n)
        print(f"\n[{name}] temp0={dict(c0)}\n         temp0.7(n={a.n})={dict(c7)}", flush=True)
    print("\n판정: B_redirect가 SAME 반복을 깨고 PLAIN검색/unlock/call/proceed로 가면 → flail-break 유효.", flush=True)

if __name__ == "__main__":
    main()
