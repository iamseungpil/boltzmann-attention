#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""제외=완전(0%) vs 강등(기본율) 구분 프로브 (무료·2026-07-18 NIGHT+·`RATE_SUBAGENT §2h`).

라이브 발견: 서브가 "Software Exclusion: Microsoft/Coursera"를 보고 base_rate=0 냄. 그러나 정책 원문 =
  "do NOT earn the 10.0% bonus rate and **instead earn the standard 1.0% rate**" (강등·0% 아님).
반면 Bronze WeWork = "earn 0% cash back" (완전제외).
⇒ 정책이 두 종류를 **문장으로 구분**. 서브가 그 문장을 읽어 갈리나?

A/B:
  base = 현 프롬프트(제외면 0·quote)
  demote = ★"제외에 두 종류: (완전=0%) vs (보너스제외=기본율로 강등). 제외 문장을 끝까지 읽어 어느 쪽인지
           판단하고 base_rate를 그에 맞게 내라. 강등이면 기본율, 완전제외면 0." 명시

대상 = Business Silver(Software 강등) + Business Bronze(WeWork 완전). 둘 다 정확히 갈려야 성공.

Run: python3 bank_demote_probe.py --base http://localhost:8140/v1 --n 3
"""
import argparse
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
from bank_fab_probes import post  # noqa: E402
import bank_percard_probe as PC  # noqa: E402
import t2_scaffold_get as SG  # noqa: E402

BASE_TAIL = (
    "For EACH transaction give the base cash-back RATE (percent number; 0 if it earns nothing). "
    "If you set 0 because a document excludes this merchant/category, copy the exact excluding sentence "
    "into 'exclusion_quote'; otherwise leave it empty. Do NOT apply promos or multiply.")

DEMOTE_TAIL = (
    "For EACH transaction give the base cash-back RATE (percent number). ★Exclusions come in TWO kinds — "
    "read the excluding sentence to the END and decide which:\n"
    "  (1) FULL exclusion: the document says the merchant earns 0% / no cash back → base_rate=0.\n"
    "  (2) BONUS exclusion (demotion): the document says the merchant does NOT earn the bonus rate but "
    "'instead earns the standard/base rate' → base_rate = that standard/base rate (NOT 0).\n"
    "Only return 0 for a FULL exclusion. When you return 0, copy the exact 0%/no-cash-back sentence into "
    "'exclusion_quote'; otherwise leave it empty. Do NOT apply promos or multiply.")

PROMPT = (
    "You are a bank rewards specialist. Below are ALL policy documents for the {card}, then the customer's "
    "transactions.\n{tail}\n\n=== {card} — POLICY DOCUMENTS ===\n{docs}\n\n=== TRANSACTIONS ===\n{txns}\n\n"
    "Reply EXACTLY one JSON object mapping transaction_id to "
    '{{"base_rate": <n>, "exclusion_quote": "<exact sentence or empty>"}}:\n{schema}')


def run(base, model, temp, card, rows, docs, tail):
    docstr = "\n\n".join("### %s\n%s" % (d["title"], d["content"]) for d in docs)
    txns = "\n".join("  %s: merchant=%s, category=%s, amount=$%.2f"
                     % (r["transaction_id"], r["merchant"], r["category"], r["amount"]) for r in rows)
    schema = json.dumps({r["transaction_id"]: {"base_rate": "<n>", "exclusion_quote": "<s>"} for r in rows})
    prompt = PROMPT.format(card=card, tail=tail, docs=docstr, txns=txns, schema=schema)
    r = post(base, {"model": model, "temperature": temp, "max_tokens": 3000, "n": 1,
                    "messages": [{"role": "user", "content": prompt}]}, timeout=600)
    return SG._merge_json(r["choices"][0]["message"].get("content") or "", {x["transaction_id"] for x in rows})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.0)
    a = ap.parse_args()

    gold = {r["transaction_id"]: r for r in PC.build_gold()}
    bycard = defaultdict(list)
    for r in gold.values():
        bycard[r["card"]].append(r)
    targets = ["Business Silver Rewards Card", "Business Bronze Rewards Card"]
    print("★강등 vs 완전제외 구분 프로브 (temp=%s)\n" % a.temp)

    for arm, tail in [("base", BASE_TAIL), ("demote", DEMOTE_TAIL)]:
        print("=" * 60, "\nARM =", arm)
        for card in targets:
            rows = bycard[card]
            docs = PC.card_docs(card, "all")
            seen, uniq = set(), []
            for r in sorted(rows, key=lambda x: x["transaction_id"]):
                key = (r["category"], round(r["gold_pts"] / r["amount"], 1))
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(r)
            for i in range(a.n):
                out = run(a.base, a.model, a.temp, card, uniq, docs, tail)
                bad = []
                for r in uniq:
                    v = out.get(r["transaction_id"]) or {}
                    try:
                        br = float(v.get("base_rate"))
                    except Exception:
                        br = None
                    gr = r["gold_pts"] / r["amount"]
                    ok = br is not None and (abs(r["amount"] * br - r["gold_pts"]) <= 1
                                             or abs(r["amount"] * br * 2 - r["gold_pts"]) <= 1
                                             or (r["gold_pts"] == 0 and br == 0))
                    if not ok:
                        bad.append("%s(%s→%s gold%.1f)" % (r["merchant"][:12], r["category"], v.get("base_rate"), gr))
                print("   %s [%d] %s %s" % (card[:20], i, "✓" if not bad else "✗%d" % len(bad), " ".join(bad[:4])))
        print()


if __name__ == "__main__":
    main()
