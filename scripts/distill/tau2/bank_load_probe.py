#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""부하 축소 프로브 — 카드전체 vs 카테고리격리 vs 소배치 (무료·2026-07-18·`RATE_SUBAGENT §2h`).

라이브: Business Silver 서브가 **30거래**(Software 12 포함)를 한꺼번에 받음 → Microsoft/Coursera 강등 놓침.
가설([[45]]): 부하 줄이면 서브가 강등을 정확히 formalize. 엔진 교정·프롬프트 과교정 없이(둘 다 기각).

A/B/C (같은 거래·같은 문서·temp0·프롬프트 동일):
  full   = 카드 전체 거래 한 서브 (라이브 현재·부하 큼)
  cat    = 카테고리별 서브 분할 (Software 12거래만)
  batch  = 카테고리 + 소배치(N거래씩) (부하 최소)

대상 = Business Silver Software(강등 Microsoft/Coursera 포함). 강등 정확도 측정.

Run: python3 bank_load_probe.py --base http://localhost:8140/v1 --n 3 [--batch 4]
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

# ★프롬프트는 §2e 배포본 고정(demote 지시 없음·부하만 변수). 강등 여부는 서브가 문서서 스스로.
PROMPT = (
    "You are a bank rewards specialist. Below are ALL policy documents for the {card}, then the customer's "
    "transactions. For EACH transaction give the base cash-back RATE (percent number; 0 if it truly earns "
    "nothing). Do NOT return 0 just because a purchase is not a premium category — apply the base rate. "
    "Return 0 ONLY if a document explicitly excludes this merchant/category, and copy the excluding sentence "
    "into 'exclusion_quote'. Do NOT apply promos or multiply.\n\n"
    "=== {card} — POLICY DOCUMENTS ===\n{docs}\n\n=== TRANSACTIONS ===\n{txns}\n\n"
    "Reply EXACTLY one JSON object mapping transaction_id to "
    '{{"base_rate": <n>, "exclusion_quote": "<exact sentence or empty>"}}:\n{schema}')


def ask(base, model, temp, card, rows, docs):
    docstr = "\n\n".join("### %s\n%s" % (d["title"], d["content"]) for d in docs)
    txns = "\n".join("  %s: merchant=%s, category=%s, amount=$%.2f"
                     % (r["transaction_id"], r["merchant"], r["category"], r["amount"]) for r in rows)
    schema = json.dumps({r["transaction_id"]: {"base_rate": "<n>", "exclusion_quote": "<s>"} for r in rows})
    prompt = PROMPT.format(card=card, docs=docstr, txns=txns, schema=schema)
    r = post(base, {"model": model, "temperature": temp, "max_tokens": 3000, "n": 1,
                    "messages": [{"role": "user", "content": prompt}]}, timeout=600)
    return SG._merge_json(r["choices"][0]["message"].get("content") or "", {x["transaction_id"] for x in rows})


def judge(out, rows):
    bad = []
    for r in rows:
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
            bad.append("%s(%s→%s g%.1f)" % (r["merchant"][:10], r["category"], v.get("base_rate"), gr))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--card", default="Business Silver Rewards Card")
    a = ap.parse_args()

    gold = PC.build_gold()
    rows = [r for r in gold if r["card"] == a.card]
    # 고유 거래
    seen, uniq = set(), []
    for r in sorted(rows, key=lambda x: x["transaction_id"]):
        if r["transaction_id"] in seen:
            continue
        seen.add(r["transaction_id"])
        uniq.append(r)
    docs = PC.card_docs(a.card, "all")
    bycat = defaultdict(list)
    for r in uniq:
        bycat[r["category"]].append(r)
    print("★부하 프로브 · %s · 총 %d거래 · Software %d거래 (강등 Microsoft/Coursera 포함)\n"
          % (a.card, len(uniq), len(bycat.get("Software", []))))

    for arm in ["full", "cat", "batch"]:
        print("=" * 60, "\nARM =", arm)
        for i in range(a.n):
            # arm별 서브 호출 단위 구성
            if arm == "full":
                calls = [uniq]
            elif arm == "cat":
                calls = list(bycat.values())
            else:
                calls = []
                for cat, crows in bycat.items():
                    for j in range(0, len(crows), a.batch):
                        calls.append(crows[j:j + a.batch])
            allbad = []
            merged = {}
            for crows in calls:
                out = ask(a.base, a.model, a.temp, a.card, crows, docs)
                merged.update(out)
            allbad = judge(merged, uniq)
            # Software 강등만 따로
            sw = [r for r in uniq if r["category"] == "Software"]
            swbad = judge(merged, sw)
            print("   [%d] 서브호출 %d회 · 전체오독 %d · Software오독 %d %s"
                  % (i, len(calls), len(allbad), len(swbad), " ".join(swbad[:5])))
        print()


if __name__ == "__main__":
    main()
