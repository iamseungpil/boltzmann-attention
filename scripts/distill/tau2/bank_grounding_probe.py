#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""완성 재설계 통합 프로브 — 세 신호 기록 (무료·2026-07-18 NIGHT+·`RATE_SUBAGENT §2e`).

사용자 통찰: 1안(힌트)·2안(raw)이 **다른 답일 때 grounding 확인**. 단 불일치가 진짜예외 vs 과엄격을
항상 가르진 못함(힌트 미준수 시 둘 다 0-일치) → **세 신호를 다 기록해 진리표로 최선 규칙을 데이터로** 결정.

거래마다 기록:
  rate_raw   = 2안: 힌트 없이 서브 base_rate (과엄격이면 0)
  rate_hint  = 1안: 힌트 준 서브 base_rate (기본율 강조)
  quote      = grounding: 서브가 낸 exclusion 인용이 **그 카드 문서에 실재**하나(엔진 substring)
  default    = 카드 기본율(서브 formalize·KB서)
  gold       = 정답 rate
그리고 여러 결합규칙의 정확도를 **한 데이터셋서** 비교:
  R0 raw만 · R1 hint만 · R2 raw+무조건백필 · R3 ★불일치→grounding · R4 ★any0→grounding

Run: python3 bank_grounding_probe.py --base http://localhost:8140/v1 --n 3
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
from bank_fab_probes import post  # noqa: E402
import bank_percard_probe as PC  # noqa: E402
import t2_scaffold_get as SG  # noqa: E402


def _norm(s):
    return re.sub(r"[^a-z0-9%]+", " ", str(s).lower()).strip()


# 2안(raw): 힌트 없이 base_rate + exclusion 인용(grounding용)
PROMPT_RAW = (
    "You are a bank rewards specialist. Below are ALL policy documents for the {card}, then the customer's "
    "transactions. For EACH transaction give the base cash-back RATE (percent number; 0 if it earns nothing). "
    "If you set 0 because a document excludes this merchant/category, copy the exact excluding sentence into "
    "'exclusion_quote'; otherwise leave it empty. Do NOT apply promos or multiply.\n\n"
    "=== {card} — POLICY DOCUMENTS ===\n{docs}\n\n=== ACCOUNT ===\nCard: {card}\nAccount opened: {open}\n\n"
    "=== TRANSACTIONS ===\n{txns}\n\nReply EXACTLY one JSON object mapping transaction_id to "
    '{{"base_rate": <n>, "exclusion_quote": "<exact sentence or empty>"}}:\n{schema}')

# 1안(hint): 기본율 강조(값 안 알려줌·도메인일반)
PROMPT_HINT = (
    "You are a bank rewards specialist. Below are ALL policy documents for the {card}, then the customer's "
    "transactions. For EACH transaction give the base cash-back RATE (percent number).\n"
    "★Do NOT return 0 just because a purchase is not a premium/bonus category. Almost every card earns a "
    "BASE rate on all other purchases — apply it. Return 0 ONLY if a document explicitly excludes this "
    "specific merchant/category. Do NOT apply promos or multiply.\n\n"
    "=== {card} — POLICY DOCUMENTS ===\n{docs}\n\n=== ACCOUNT ===\nCard: {card}\nAccount opened: {open}\n\n"
    "=== TRANSACTIONS ===\n{txns}\n\nReply EXACTLY one JSON object mapping transaction_id to its "
    "base_rate number:\n{schema}")


def _ask(base, model, temp, prompt, ids):
    r = post(base, {"model": model, "temperature": temp, "max_tokens": 3000, "n": 1,
                    "messages": [{"role": "user", "content": prompt}]}, timeout=600)
    return SG._merge_json(r["choices"][0]["message"].get("content") or "", ids)


def _fnum(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--cards", default="")
    a = ap.parse_args()

    gold = PC.build_gold()
    bycard = defaultdict(list)
    for r in gold:
        bycard[r["card"]].append(r)
    cards = [c for c in sorted(bycard) if not a.cards or c in a.cards.split(",")]
    print("★grounding 통합 프로브 (세 신호·temp=%s) · 카드 %d\n" % (a.temp, len(cards)))

    RULES = ["R0_raw", "R1_hint", "R2_raw+fill", "R3_disagree→ground", "R4_any0→ground"]
    ok = {k: 0 for k in RULES}
    ntot = 0
    diag = defaultdict(int)   # grounding 동작 진단

    def hits(rate, r):
        return rate is not None and abs(r["amount"] * rate - r["gold_pts"]) <= 1

    def correct(rate, r):
        return hits(rate, r) or hits(rate * 2 if rate else None, r) or (r["gold_pts"] == 0 and rate == 0)

    for card in cards:
        rows = bycard[card]
        docs = PC.card_docs(card, "all")
        docnorm = _norm(" ".join((d.get("content") or "") for d in docs))
        default = PC.card_base_default(a.base, a.model, card, docs)
        seen, uniq = set(), []
        for r in sorted(rows, key=lambda x: x["transaction_id"]):
            key = (r["category"], round(r["gold_pts"] / r["amount"], 1))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        if not docs or not uniq:
            continue
        ids = {r["transaction_id"] for r in uniq}
        docstr = "\n\n".join("### %s\n%s" % (d["title"], d["content"]) for d in docs)
        txns = "\n".join("  %s: merchant=%s, category=%s, amount=$%.2f, date=%s"
                         % (r["transaction_id"], r["merchant"], r["category"], r["amount"], r["date"]) for r in uniq)
        sc_raw = json.dumps({r["transaction_id"]: {"base_rate": "<n>", "exclusion_quote": "<str>"} for r in uniq})
        sc_h = json.dumps({r["transaction_id"]: "<n>" for r in uniq})
        p_raw = PROMPT_RAW.format(card=card, docs=docstr, open=uniq[0]["account_open"], txns=txns, schema=sc_raw)
        p_h = PROMPT_HINT.format(card=card, docs=docstr, open=uniq[0]["account_open"], txns=txns, schema=sc_h)
        for i in range(a.n):
            try:
                raw = _ask(a.base, a.model, a.temp, p_raw, ids)
                hint = _ask(a.base, a.model, a.temp, p_h, ids)
            except Exception as e:
                print("### %s [%d] ERR %r" % (card, i, str(e)[:60]))
                continue
            if not raw or not hint:
                print("### %s [%d] 파싱실패" % (card, i))
                continue
            fails = defaultdict(list)
            for r in uniq:
                ntot += 1
                rr = _fnum((raw.get(r["transaction_id"]) or {}).get("base_rate"))
                rh = _fnum(hint.get(r["transaction_id"]))
                quote = _norm((raw.get(r["transaction_id"]) or {}).get("exclusion_quote") or "")
                grounded = len(quote) >= 8 and quote in docnorm
                # 결합규칙별 최종 rate
                cand = {}
                cand["R0_raw"] = rr
                cand["R1_hint"] = rh
                cand["R2_raw+fill"] = default if (rr == 0 and default is not None) else rr
                # R3: 불일치면 grounding — grounded 0이면 0, 아니면 nonzero(둘 중 0 아닌 것)
                if rr == rh:
                    cand["R3_disagree→ground"] = rr
                else:
                    if (rr == 0 or rh == 0) and default is not None:
                        cand["R3_disagree→ground"] = 0.0 if grounded else (rh if rh else rr)
                    else:
                        cand["R3_disagree→ground"] = rh   # 둘 다 nonzero 불일치 → 힌트
                # R4: 임의 0이면 grounding
                if rr == 0 and default is not None:
                    cand["R4_any0→ground"] = 0.0 if grounded else default
                    diag["0keep" if grounded else "0fill"] += 1
                    if grounded and r["gold_pts"] != 0:
                        diag["mis_keep"] += 1
                    if (not grounded) and r["gold_pts"] == 0:
                        diag["mis_fill"] += 1
                else:
                    cand["R4_any0→ground"] = rr
                for k in RULES:
                    c = correct(cand[k], r)
                    ok[k] += c
                    if not c and k in ("R4_any0→ground",):
                        fails[k].append("%s(raw=%s hint=%s q=%s→%s gold_pts=%.0f)"
                                        % (r["category"], rr, rh, "Y" if grounded else "N", cand[k], r["gold_pts"]))
            fs = " ".join(fails["R4_any0→ground"][:4])
            print("### %s [%d] default=%s  R4=%s" % (card, i, default, fs or "✓"))
        print()

    print("=" * 72)
    print("★결합규칙별 정확도 (같은 데이터·n=%d·거래 %d):" % (a.n, ntot))
    for k in RULES:
        print("   %-24s %d/%d = %3.0f%%" % (k, ok[k], ntot, 100 * ok[k] / max(ntot, 1)))
    print("\n★R4 grounding 동작: 0유지 %d(잘못 %d) · 0백필 %d(예외파괴 %d)"
          % (diag["0keep"], diag["mis_keep"], diag["0fill"], diag["mis_fill"]))
    print("판정: R3/R4가 R0~R2보다 높고 예외파괴 0 = grounding이 잔여 닫음")


if __name__ == "__main__":
    main()
