#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""★C113 F1 게이트 — base_rate formalize가 **다른 카드/카테고리서도** 정확한가 (무료·2026-07-18 NIGHT handoff §0.2).

C113은 task_026 한 셀조합(Business Silver/Silver × Travel/Software·4거래)에서만 100%를 봤다.
ratefix producer의 **전제** = "32B는 KB 텍스트 해석(base_rate)엔 강하다"([[45]] 부하가 아니다).
전제가 다른 카드/카테고리서 깨지면 → base_rate도 부하 → §PROD-2 원결론(producer 분담선) 복귀.

측정 대상 = **ratefix 계약 그대로**(`a2/banking_knowledge.gate.json` §variants.ratefix):
  모델 = 거래별 base_rate(percent) + promo 파라미터만 formalize (★최종 rate·곱셈·날짜판정 안 함)
  엔진 = promo 적격/활성 판정(날짜) + 곱셈 → 최종 배율
판정 = 엔진이 합성한 최종 배율 == gold 배율. 부차 = 모델 raw base_rate 전수([[08]] 포렌식).

⚠️gold·거래·문서 전부 **런타임에 벤치 데이터에서 유도**(리터럴 0·[[03b]]):
  gold 배율 = tasks.json의 gold `update_transaction_rewards_*`(new_rewards_earned) ÷ db.json 거래금액
  거래속성(카드·카테고리·금액·날짜)·계정개설일 = db.json
  KB 문서 = **기계적 규칙**(title.startswith(카드명))로 카드-스코프 전량 — 정답 문서 골라주기(spoonfeed) 금지
⚠️이 스크립트는 측정 코드다(엔진 아님) — `bank_rate_formalize_probe.py` 선례와 동일.

Run: python3 bank_rate_f1_gate_probe.py --base http://localhost:8141/v1 --n 8
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bank_fab_probes import post  # noqa: E402
from bank_rate_toolcall_probe import _d, _add_months  # noqa: E402

DOM_DEFAULT = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge"
REWARD_TOOL = "update_transaction_rewards"


def _num(s):
    return float(re.sub(r"[^0-9.]", "", str(s)))


def load_gold(dom):
    """gold 배율 = (gold new_rewards_earned 포인트) / (거래금액). 벤치 데이터서 유도·리터럴 0."""
    tasks = json.load(open(os.path.join(dom, "tasks.json"), encoding="utf-8"))
    db = json.load(open(os.path.join(dom, "db.json"), encoding="utf-8"))
    txns = db["credit_card_transaction_history"]["data"]
    accts = {}
    for a in db["credit_card_accounts"]["data"].values():
        accts[(a["user_id"], a["card_type"])] = a["date_of_account_open"]

    rows = []
    for t in tasks:
        for act in (t.get("evaluation_criteria") or {}).get("actions", []) or []:
            if act.get("name") != "call_discoverable_agent_tool":
                continue
            args = act.get("arguments") or {}
            if REWARD_TOOL not in (args.get("agent_tool_name") or ""):
                continue
            inner = args.get("arguments")
            if isinstance(inner, str):
                inner = json.loads(inner)
            tid = inner.get("transaction_id")
            r = txns.get(tid)
            if not r:
                continue
            amt = _num(r["transaction_amount"])
            rows.append({
                "task": t["id"], "transaction_id": tid, "card": r["credit_card_type"],
                "category": r["category"], "amount": amt, "date": r["transaction_date"],
                "actual_rewards": r["rewards_earned"],
                "account_open": accts.get((r["user_id"], r["credit_card_type"])),
                "gold_rate": round(_num(inner["new_rewards_earned"]) / amt, 2),
            })
    return rows


def load_docs(dom, card):
    """카드-스코프 문서 전량 — 기계적 규칙(제목 접두)뿐. 정답 문서 선별 금지."""
    dd = os.path.join(dom, "documents")
    out = []
    for fn in sorted(os.listdir(dd)):
        d = json.load(open(os.path.join(dd, fn), encoding="utf-8"))
        if (d.get("title") or "").startswith(card):
            out.append(d)
    return out


PROMPT = (
    "You are computing cash-back reward rates for a bank. Below are the reward-rate policy documents for "
    "the customer's card, the account opening date, and their transactions on that card.\n\n"
    "=== POLICY DOCUMENTS ===\n{docs}\n\n"
    "=== ACCOUNT ===\nCard: {card}\nAccount opened: {open}\n\n"
    "=== TRANSACTIONS ===\n{txns}\n\n"
    "For EACH transaction, read the policy documents and report:\n"
    "- base_rate: the BASE cash-back rate for this card AND this purchase category, as a percent number "
    "(e.g. 10 for 10%, 4 for 4%). The rate depends on the card and the category of the purchase.\n"
    "- has_promo / promo_mult / promo_window_months / promo_start / promo_end: if a limited-time promo "
    "(e.g. double cash back for new customers) exists for this card, its multiplier, window length in "
    "months, and the promo period dates (MM/DD/YYYY). If no promo exists, use has_promo=false, "
    "promo_mult=1 and empty dates.\n"
    "★Do NOT compute the final rate, do NOT apply the promo, do NOT multiply anything, and do NOT "
    "decide whether the promo dates apply — a separate deterministic system does all of that. Only report "
    "what the policy documents say.\n"
    "Reply with EXACTLY one JSON object and nothing else:\n{schema}"
)


def build_prompt(card, open_date, rows):
    docs = "\n\n".join("### %s\n%s" % (d["title"], d["content"]) for d in rows[0]["_docs"])
    txn_lines = "\n".join(
        "  %s: category=%s, merchant_category=%s, amount=$%.2f, date=%s"
        % (r["transaction_id"], r["category"], r["category"], r["amount"], r["date"]) for r in rows)
    schema = json.dumps({r["transaction_id"]: {
        "base_rate": "<n>", "has_promo": "<bool>", "promo_mult": "<n>",
        "promo_window_months": "<n>", "promo_start": "<MM/DD/YYYY or empty>",
        "promo_end": "<MM/DD/YYYY or empty>"} for r in rows})
    return PROMPT.format(docs=docs, card=card, open=open_date, txns=txn_lines, schema=schema)


def engine_rate(base_rate, has_promo, promo_mult, account_open, txn_date, window_months,
                promo_start, promo_end):
    """결정론 엔진 몫(ratefix op와 동일 산술: date_between + date_in_window + multiply)."""
    base = float(base_rate)
    if not has_promo:
        return base
    ao, td = _d(account_open), _d(txn_date)
    if ao is None or td is None:
        return base
    elig = True
    if promo_start and promo_end:
        ps, pe = _d(promo_start), _d(promo_end)
        elig = bool(ps and pe and ps <= ao <= pe)
    active = ao <= td <= _add_months(ao, int(window_months or 6))
    return base * float(promo_mult or 1) if (elig and active) else base


def parse_json(text, keys):
    for i in range(len(text or "")):
        if text[i] != "{":
            continue
        for j in range(len(text), i, -1):
            try:
                j_obj = json.loads(text[i:j])
            except Exception:
                continue
            if isinstance(j_obj, dict) and any(k in j_obj for k in keys):
                return j_obj
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8141/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--dom", default=DOM_DEFAULT)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=6000)  # [[08]] 절단이 날조로 오독되는 사고 방지
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    gold = load_gold(a.dom)
    bycard = defaultdict(list)
    for r in gold:
        bycard[r["card"]].append(r)
    print("★C113 F1 게이트 — gold %d거래 · 카드 %d종 · 셀 %d개 (전부 벤치 데이터서 유도)"
          % (len(gold), len(bycard), len({(r["card"], r["category"]) for r in gold})))
    for r in sorted(gold, key=lambda x: (x["card"], x["category"])):
        print("   %-32s %-10s $%8.2f %s  gold_rate=%-5s (task %s)"
              % (r["card"], r["category"], r["amount"], r["date"], r["gold_rate"], r["task"]))
    print()

    cell_ok = defaultdict(int)
    cell_n = defaultdict(int)
    raw = []
    for card, rows in sorted(bycard.items()):
        docs = load_docs(a.dom, card)
        if not docs:
            print("[SKIP] %s: 카드-스코프 문서 0" % card)
            continue
        rows[0]["_docs"] = docs
        opens = {r["account_open"] for r in rows}
        prompt = build_prompt(card, sorted(opens)[0], rows)
        print("### %s | 문서 %d개 · 거래 %d개 · 프롬프트 %d자" % (card, len(docs), len(rows), len(prompt)))
        for i in range(a.n):
            try:
                r = post(a.base, {"model": a.model, "temperature": a.temp,
                                  "max_tokens": a.max_tokens, "n": 1,
                                  "messages": [{"role": "user", "content": prompt}]}, timeout=600)
                ch = r["choices"][0]
                out = parse_json(ch["message"].get("content"), [x["transaction_id"] for x in rows])
                fin = ch.get("finish_reason")
            except Exception as e:
                print("   [%d] ERR %r" % (i, str(e)[:70]))
                continue
            if out is None:
                print("   [%d] 파싱실패 (finish=%s)" % (i, fin))
                continue
            line = []
            for row in rows:
                cell = (row["card"], row["category"])
                cell_n[cell] += 1
                v = out.get(row["transaction_id"]) or {}
                try:
                    got = engine_rate(v.get("base_rate"), bool(v.get("has_promo")), v.get("promo_mult", 1),
                                      row["account_open"], row["date"], v.get("promo_window_months", 6),
                                      v.get("promo_start"), v.get("promo_end"))
                except Exception:
                    got = None
                ok = got is not None and abs(float(got) - row["gold_rate"]) < 0.01
                cell_ok[cell] += ok
                raw.append({"card": row["card"], "category": row["category"], "sample": i,
                            "base_rate": v.get("base_rate"), "has_promo": v.get("has_promo"),
                            "final": got, "gold": row["gold_rate"], "ok": bool(ok)})
                line.append("%s%s=%s(base %s)" % ("✓" if ok else "✗", row["category"],
                                                  got, v.get("base_rate")))
            print("   [%d] %s" % (i, " ".join(line)))
        print()

    print("=" * 78)
    print("★셀별 최종배율 정확율 (엔진 합성 후 == gold)")
    worst = 1.0
    for cell in sorted(cell_n):
        p = cell_ok[cell] / max(cell_n[cell], 1)
        worst = min(worst, p)
        print("   %-32s %-10s %2d/%2d = %3.0f%%" % (cell[0], cell[1], cell_ok[cell], cell_n[cell], 100 * p))
    tot_ok, tot_n = sum(cell_ok.values()), sum(cell_n.values())
    print("   %-43s %2d/%2d = %3.0f%%" % ("[전체]", tot_ok, tot_n, 100 * tot_ok / max(tot_n, 1)))
    print("\n판정(handoff §0.2): 최저 셀 %.0f%% — <90%%면 base_rate 해석 자체가 [[45]] 부하 ⇒ §PROD-2 복귀 검토"
          % (100 * worst))
    if a.out:
        json.dump(raw, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("raw → %s" % a.out)


if __name__ == "__main__":
    main()
