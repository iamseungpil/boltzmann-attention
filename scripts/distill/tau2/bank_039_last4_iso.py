#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""task_039 last-4 dispute-loop 격리 (2026-07-23·greenlight/approve_iso 동형·[[08]]).
루프: 손님이 last-4(1652)를 여러 번 확인해줬는데 에이전트가 "we need the last 4 digits" 반복·file 안 함.
격리: 손님이 1652 준 직후에서 얼려 → A(있는그대로)=재-verify 반복? B(값+file 지시)=file_dispute 재시도?
Run: python3 bank_039_last4_iso.py --base http://localhost:8140/v1 --n 8
"""
import argparse, gzip, json, os, sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
SIMDIR = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
from bank_fab_probes import to_openai, load_tools, post, AGENT_INSTRUCTION  # noqa

DIRECTIVE = ("[REMINDER] The customer has already retrieved and confirmed the last 4 digits of the Gold "
             "Rewards Card: 1652. You already have this value — do NOT ask for it again and do NOT re-run "
             "any lookup for it. File the dispute NOW: unlock file_credit_card_transaction_dispute and call "
             "it via call_discoverable_agent_tool with the transaction_id of the charge being disputed and "
             "card_last_4_digits='1652'. Then move on to the next disputed charge.")

def load_sim():
    with gzip.open(os.path.join(SIMDIR, "bank_eplansmoke_full_20260723.results.json.gz"), "rt", encoding="utf-8") as f:
        return next(s for s in json.load(f)["simulations"] if s["task_id"] == "task_039")

def freeze_idx(msgs):
    """FIRST user msg that gives 1652 AND asks to file (before frustration/transfer escalation).
    (last-1652 msg = 손님 transfer 요청 지점 → 오판. 첫 file-요청 지점서 얼려야 루프 테스트.)"""
    for i, m in enumerate(msgs):
        if m.get("role") != "user":
            continue
        c = str(m.get("content", "")).lower()
        if "1652" in c and ("file" in c or "proceed" in c) and "transfer" not in c:
            return i
    raise RuntimeError("no 1652+file user msg")

def classify(msg):
    tcs = msg.get("tool_calls") or []
    if not tcs:
        c = str(msg.get("content", "")).lower()
        if "last 4" in c or "last four" in c or "digits" in c:
            return "산문:last-4 재요청"
        return "산문:기타"
    names = []
    for tc in tcs:
        nm = tc["function"]["name"]
        a = json.loads(tc["function"].get("arguments") or "{}")
        eff = a.get("agent_tool_name")
        names.append(eff or nm)
    if any("file_credit_card_transaction_dispute" in str(n) for n in names):
        return "★file_dispute"
    if any(str(n).startswith("unlock") and "file_credit" in str(n) for n in names):
        return "unlock:file_dispute"
    if any("last_4" in str(n) or "give_discoverable_user" in str(n) for n in names):
        return "last-4 도구 재시도"
    return "다른도구:" + str(names[0])

def run(base, model, conv, tools, temp, n):
    c = Counter()
    for _ in range(n):
        try:
            r = post(base, {"model": model, "messages": conv, "tools": tools,
                            "temperature": temp, "max_tokens": 450, "n": 1}, timeout=420)
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
    s = load_sim()
    msgs = s["messages"]
    fi = freeze_idx(msgs)
    frozen = sysmsg + to_openai(msgs[:fi + 1])
    print(f"frozen at 1652-user idx={fi}", flush=True)
    base_conv = frozen
    dir_conv = frozen + [{"role": "user", "content": DIRECTIVE}]  # 지시=시스템 리마인더 대용(user 채널)
    for name, conv in (("A_asis", base_conv), ("B_directive", dir_conv)):
        c0 = run(a.base, a.model, conv, tools, 0.0, 1)
        c7 = run(a.base, a.model, conv, tools, 0.7, a.n)
        print(f"\n[{name}] temp0={dict(c0)}\n         temp0.7(n={a.n})={dict(c7)}", flush=True)
    print("\n판정: B_directive가 file_dispute(또는 unlock:file)로 가면 → '값 있으면 재-file' 지시가 루프 깬다.", flush=True)

if __name__ == "__main__":
    main()
