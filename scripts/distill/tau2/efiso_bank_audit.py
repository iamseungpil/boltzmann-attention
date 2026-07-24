#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E-F3-ISO Phase 1 — banking wrong-reference 소급 재감사 (2026-07-24·C124 프로토콜 일반화).
rall19/20/21의 file_dispute wrong/fabricated transaction_id 사례 전수에 대해
A_minimal(유저 발화+producer 목록) vs B_fullctx(결정 직전 flat transcript) 격리 프로브를 돌리고
{flip-to-load(전사/자기-정박) / capability / ambiguous}로 분류. 로컬 vLLM=무료.
Run(remote): seka python efiso_bank_audit.py --base http://localhost:8140/v1 --n 3
"""
import argparse
import gzip
import json
import os
import re
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
SIMR = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")

INSTANCES = [  # (gz, task, trial, label)
    ("bank_rall19_treat_20260723.results.json.gz", "task_039", 0, "r19t.039.0"),
    ("bank_rall20a_20260723.results.json.gz", "task_031", 0, "r20.031.0"),
    ("bank_rall20a_20260723.results.json.gz", "task_039", 0, "r20.039.0"),
    ("bank_rall21a_20260724.results.json.gz", "task_039", 0, "r21.039.0"),
]

ASK = ("Based on the customer's request and the transaction history above, list the transaction_id "
       "for EACH transaction the customer wants to dispute, in the customer's order. Answer with "
       "ONLY a JSON array of transaction_id strings, nothing else.")


def load_sim(gz, task, trial):
    d = json.load(gzip.open(os.path.join(SIMR, gz)))
    for s in d.get("simulations", []):
        if str(s.get("task_id")) == task and s.get("trial") == trial:
            return s
    raise SystemExit("sim not found %s %s.%s" % (gz, task, trial))


def gold_ids(sim):
    out = []
    for ac in (sim.get("reward_info") or {}).get("action_checks") or []:
        a = ac.get("action") or {}
        if "file_credit_card_transaction_dispute" in str(a.get("arguments", "")):
            m = re.search(r"txn_[0-9a-f]+", str(a.get("arguments")))
            if m:
                out.append(m.group())
    return out


def filed_ids(sim):
    out = []
    for msg in sim.get("messages", []):
        for tc in (msg.get("tool_calls") or []):
            aa = str(tc.get("arguments", ""))
            if ("file_credit_card_transaction_dispute" in aa
                    and tc.get("name") == "call_discoverable_agent_tool"):
                m = re.search(r"txn_[0-9a-f]+", aa)
                if m:
                    out.append(m.group())
    return out


def listing_of(sim):
    best = ""
    for m in sim.get("messages", []):
        c = str(m.get("content") or "")
        if m.get("role") == "tool" and "credit_card_transaction_history" in c and len(c) > len(best):
            best = c
    return best


def prefix_before_first_file(sim):
    msgs = sim.get("messages", [])
    for i, m in enumerate(msgs):
        for tc in (m.get("tool_calls") or []):
            if ("file_credit_card_transaction_dispute" in str(tc.get("arguments", ""))
                    and tc.get("name") == "call_discoverable_agent_tool"):
                return msgs[:i]
    return msgs


def flat(msgs):
    lines = []
    for m in msgs:
        r = m.get("role")
        for tc in (m.get("tool_calls") or []):
            lines.append("[%s TOOL_CALL] %s %s" % (r, tc.get("name"), str(tc.get("arguments"))[:400]))
        c = str(m.get("content") or "")
        if c:
            lines.append("[%s] %s" % (r, c))
    return "\n".join(lines)


def utext(sim):
    return "\n".join(str(m.get("content") or "") for m in sim.get("messages", [])
                     if m.get("role") == "user")


def call(base, model, content, temp):
    r = requests.post(base + "/chat/completions", json={
        "model": model, "temperature": temp, "max_tokens": 600,
        "messages": [{"role": "system", "content": "You are a precise banking assistant."},
                     {"role": "user", "content": content}]}, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    grand = []
    for gz, task, trial, label in INSTANCES:
        sim = load_sim(gz, task, trial)
        gold = gold_ids(sim)
        filed = filed_ids(sim)
        wrong = [f for f in dict.fromkeys(filed) if f not in gold]
        lst = listing_of(sim)
        fab = [w for w in wrong if w not in lst]
        print("== %s gold=%d filed=%d wrong=%s fabricated=%s"
              % (label, len(gold), len(filed), [w[-6:] for w in wrong], [w[-6:] for w in fab]))
        if not wrong:
            print("   (no wrong ids — skip)")
            continue
        pa = ("CUSTOMER MESSAGES:\n" + utext(sim)[:6000] + "\n\nTRANSACTION HISTORY:\n"
              + lst[:20000] + "\n\n" + ASK)
        pb = ("Below is a bank-agent conversation transcript (including tool outputs).\n\n"
              + flat(prefix_before_first_file(sim))[:60000] + "\n\n" + ASK)
        row = {"label": label, "wrong": wrong, "fab": fab, "A": [], "B": []}
        for tag, prompt, bucket in (("A", pa, row["A"]), ("B", pb, row["B"])):
            for i in range(1 + a.n):
                t = 0.0 if i == 0 else 0.7
                try:
                    txt = call(a.base, a.model, prompt, t)
                except Exception as e:
                    print("   %s run%d ERROR %r" % (tag, i, e))
                    continue
                ids = list(dict.fromkeys(re.findall(r"txn_[0-9a-f]+", txt)))
                gold_hit = len([g for g in gold if g in ids])
                wrong_rep = [w[-6:] for w in wrong if w in ids]
                bucket.append((gold_hit, wrong_rep))
                print("   %s run%d gold %d/%d wrong_reproduced=%s"
                      % (tag, i, gold_hit, len(gold), wrong_rep))
        grand.append(row)
    print("\n==== VERDICT PER INSTANCE ====")
    for row in grand:
        A, B = row["A"], row["B"]
        if not A or not B:
            print("%s: incomplete" % row["label"])
            continue
        a_clean = all(not w for _, w in A)
        a_gold = sum(g for g, _ in A) / len(A)
        b_rep = sum(1 for _, w in B if w) / len(B)
        v = ("FLIP-TO-LOAD" if a_clean and b_rep >= 0.5 else
             "LOAD-PARTIAL" if a_clean else "CAPABILITY/AMBIG")
        print("%s: A_gold=%.1f A_wrong_free=%s B_wrong_rate=%.2f -> %s"
              % (row["label"], a_gold, a_clean, b_rep, v))


if __name__ == "__main__":
    main()
