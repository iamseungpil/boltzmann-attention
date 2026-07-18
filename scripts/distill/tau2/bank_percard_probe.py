#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""카드당 문서-주입 프로브 (무료·2026-07-18 NIGHT+·`RATE_SUBAGENT §2e` 검증 (a)).

질문: 검색 없이 **그 카드 문서 전부**(제목 접두 매칭·~3k토큰)를 서브에 주면, 서브가 base_rate를 정확히 내나?
  = 재설계(카드당 격리+문서 주입)의 make-or-break. 통과 시 검색부실(오늘 over-flag 원인) 우회 실증.

대조 = 오늘 라이브 iso5/trace: 2카드 섞어 bm25 검색 → base_rate 오독 15/26.
이번 = 1카드씩·그 카드 문서 통째·검색 0 → 오독 몇 개?

★spoon 아님([[03b]]): 그 카드 문서 **전부**(rate·예외·프로모·보험·수수료·referral 24개) 주입·정답 문서 선별 0.
  서브가 그 안서 rate를 스스로 찾아야(노이즈 문서 다수). 도메인 리터럴 0(카드명=레코드·필터=제목매칭).
gold = 벤치 데이터서 유도(dispute 아닌 거래=옳음·rewards/amount=rate). 리터럴 0.

Run: python3 bank_percard_probe.py --base http://localhost:8140/v1 --n 3
"""
import argparse
import json
import os
import re
import sys
import glob
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
from bank_fab_probes import post  # noqa: E402
import bank_rate_f1_gate_probe as P  # noqa: E402
import t2_scaffold_get as SG  # noqa: E402

DOM = P.DOM_DEFAULT


def _num(s):
    return float(re.sub(r"[^0-9.]", "", str(s)))


# ★rate-키워드(비교 arm 전용·배포 아님): 제목에 이 단어가 있으면 "rate 관련"으로 취급.
#   ⚠️이건 spoon-feed 방향(정답 문서 선별)이라 배포 금지 — "전체 vs 선별" 필요성 측정용.
RATE_KW = ["earn", "cash back", "cashback", "rate", "exceptions", "exclusion", "qualify",
           "double", "promo", "conversion", "green", "sustainability"]


def card_docs(card, mode="all"):
    """카드 문서. mode='all'=전부(§2e 배포안) · mode='rate'=제목 rate-키워드만(비교 arm)."""
    dd = os.path.join(DOM, "documents")
    out = []
    for fn in sorted(os.listdir(dd)):
        d = json.load(open(os.path.join(dd, fn), encoding="utf-8"))
        if not (d.get("title") or "").startswith(card + ": "):
            continue
        if mode == "rate":
            sub = d["title"].split(": ", 1)[1].lower() if ": " in d["title"] else ""
            if not any(k in sub for k in RATE_KW):
                continue
        out.append(d)
    return out


def build_gold():
    """카드+카테고리+개설일 → gold rate. dispute 아닌 거래(벤치 옳음)의 rewards/amount + fixed gold."""
    tasks = json.load(open(os.path.join(DOM, "tasks.json"), encoding="utf-8"))
    db = json.load(open(os.path.join(DOM, "db.json"), encoding="utf-8"))
    tx = db["credit_card_transaction_history"]["data"]
    accts = {}
    for a in db["credit_card_accounts"]["data"].values():
        accts[(a["user_id"], a["card_type"])] = a["date_of_account_open"]
    fixed, disp, users = {}, defaultdict(set), {}
    for t in tasks:
        for act in (t.get("evaluation_criteria") or {}).get("actions", []) or []:
            args = act.get("arguments") or {}
            inner = args.get("arguments")
            if isinstance(inner, str):
                inner = json.loads(inner)
            if act.get("name") == "call_discoverable_agent_tool" and \
                    "update_transaction_rewards" in (args.get("agent_tool_name") or ""):
                fixed[inner["transaction_id"]] = _num(inner["new_rewards_earned"])
            elif act.get("name") == "call_discoverable_user_tool" and \
                    "submit_cash_back_dispute" in str(args.get("discoverable_tool_name")):
                disp[t["id"]].add(inner["transaction_id"])
                users[t["id"]] = inner["user_id"]
    rows = []
    for tid, uid in users.items():
        for r in tx.values():
            if r["user_id"] != uid:
                continue
            t_id = r["transaction_id"]
            if t_id in disp[tid]:
                if t_id not in fixed:
                    continue
                gp = fixed[t_id]
            else:
                gp = _num(r["rewards_earned"])
            rows.append({"transaction_id": t_id, "card": r["credit_card_type"],
                         "category": r["category"], "merchant": r["merchant_name"],
                         "amount": _num(r["transaction_amount"]), "date": r["transaction_date"],
                         "account_open": accts.get((uid, r["credit_card_type"])),
                         "gold_pts": gp})
    return rows


PROMPT = (
    "You are a bank rewards specialist. Below are ALL policy documents for the {card}, followed by the "
    "customer's transactions on that card. Read the documents and, for EACH transaction, report the base "
    "cash-back RATE that applies, as a percent NUMBER (e.g. 10 for 10%, 4 for 4%, 0 if it earns nothing).\n"
    "The rate depends on the purchase category, the merchant (some merchants are excluded = 0%), and how "
    "long a subscription has run. Do NOT apply any limited-time promo or multiply — just the base rate.\n"
    "{baseline}"
    "\n=== {card} — POLICY DOCUMENTS ===\n{docs}\n\n"
    "=== ACCOUNT ===\nCard: {card}\nAccount opened: {open}\n\n"
    "=== TRANSACTIONS ===\n{txns}\n\n"
    "Reply with EXACTLY one JSON object mapping each transaction_id to its base_rate number:\n{schema}")

# ★1안(프롬프트): 기본율 강조. 값 안 알려줌(도메인일반)·"0으로 두지 마라·기타구매 기본율 적용".
BASELINE_HINT = (
    "★IMPORTANT: Do NOT return 0 just because a purchase is not in a premium/bonus category. Almost every "
    "card earns a BASE rate on ALL other purchases (look for 'all other purchases', 'other purchases', or "
    "the general earn rate in the documents). Return 0 ONLY if a document explicitly excludes that specific "
    "merchant/category. Otherwise apply the card's base/other-purchases rate.\n")


def card_base_default(base, model, card, docs):
    """2안: 그 카드의 '기타구매 기본율'(all other purchases rate)을 KB서 formalize.
    ★값은 서브가 문서서 읽음(엔진 하드코딩 0·[[05]]). 엔진은 이 값을 default로 쓸 뿐."""
    docstr = "\n\n".join("### %s\n%s" % (d["title"], d["content"]) for d in docs)
    prompt = ("Read these %s policy documents. What is the BASE cash-back rate that applies to 'all other "
              "purchases' (purchases NOT in any bonus/premium category) on this card, as a percent number? "
              "Reply with EXACTLY one JSON object: {\"base_default\": <number>}\n\n%s" % (card, docstr))
    try:
        r = post(base, {"model": model, "temperature": 0.0, "max_tokens": 300, "n": 1,
                        "messages": [{"role": "user", "content": prompt}]}, timeout=300)
        j = SG._merge_json(r["choices"][0]["message"].get("content") or "", {"base_default"})
        return float(j.get("base_default")) if j and j.get("base_default") is not None else None
    except Exception:
        return None


def run_card(base, model, temp, card, rows, docs, baseline=False):
    docstr = "\n\n".join("### %s\n%s" % (d["title"], d["content"]) for d in docs)
    txns = "\n".join("  %s: merchant=%s, category=%s, amount=$%.2f, date=%s"
                     % (r["transaction_id"], r["merchant"], r["category"], r["amount"], r["date"]) for r in rows)
    schema = json.dumps({r["transaction_id"]: "<base_rate number>" for r in rows})
    prompt = PROMPT.format(card=card, docs=docstr, open=rows[0]["account_open"], txns=txns, schema=schema,
                           baseline=(BASELINE_HINT if baseline else ""))
    r = post(base, {"model": model, "temperature": temp, "max_tokens": 2000, "n": 1,
                    "messages": [{"role": "user", "content": prompt}]}, timeout=600)
    ch = r["choices"][0]
    return SG._merge_json(ch["message"].get("content") or "", {x["transaction_id"] for x in rows}), \
        ch.get("finish_reason"), len(prompt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--cards", default="", help="쉼표구분 카드명 필터(기본=전부)")
    ap.add_argument("--doc_mode", choices=["all", "rate"], default="all",
                    help="all=카드문서전부(§2e·배포) · rate=제목 rate-키워드만(비교arm·spoon방향)")
    ap.add_argument("--baseline", action="store_true",
                    help="1안: 프롬프트에 기본율 강조(0 남발 마라·기타구매 기본율 적용). 값 안 알려줌(도메인일반).")
    ap.add_argument("--mode2_default", action="store_true",
                    help="2안 진단: 서브가 카드 기본율(other-purchases rate)을 별도 formalize → 0셀 백필."
                         " ⚠예외(WeWork=0)까지 덮을 위험 → fixed/broken 함께 보고.")
    a = ap.parse_args()

    gold = build_gold()
    bycard = defaultdict(list)
    for r in gold:
        bycard[r["card"]].append(r)
    cards = [c for c in sorted(bycard) if not a.cards or c in a.cards.split(",")]
    print("★카드당 문서-주입 프로브 (검색 0·temp=%s·doc_mode=%s) · 카드 %d종\n"
          % (a.temp, a.doc_mode, len(cards)))

    cell_ok, cell_n = defaultdict(int), defaultdict(int)
    for card in cards:
        rows = bycard[card]
        docs = card_docs(card, a.doc_mode)
        # 카드당 셀 대표(카테고리×rate 중복 제거) — 프롬프트 크기·균형
        seen, uniq = set(), []
        for r in sorted(rows, key=lambda x: x["transaction_id"]):
            key = (r["category"], round(r["gold_pts"] / r["amount"], 1))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        if not docs or not uniq:
            print("### %s — 문서 %d·거래 %d (SKIP)" % (card, len(docs), len(uniq)))
            continue
        card_default = card_base_default(a.base, a.model, card, docs) if a.mode2_default else None
        for i in range(a.n):
            try:
                out, fin, plen = run_card(a.base, a.model, a.temp, card, uniq, docs, a.baseline)
            except Exception as e:
                print("### %s [%d] ERR %r" % (card, i, str(e)[:70]))
                continue
            if not out:
                print("### %s [%d] 파싱실패 (finish=%s)" % (card, i, fin))
                continue
            bad = []
            m2_fix = m2_break = 0
            for r in uniq:
                gr = r["gold_pts"] / r["amount"]
                cell = (card, "%s@%.1f" % (r["category"], gr))
                cell_n[cell] += 1
                v = out.get(r["transaction_id"])
                try:
                    br = float(v)
                except Exception:
                    br = None
                # 2안 진단: 서브가 0 낸 셀을 카드 기본율로 백필하면 고쳐지나/깨지나
                if a.mode2_default and card_default is not None and br == 0:
                    would = r["amount"] * card_default
                    gold_zero = r["gold_pts"] == 0
                    if abs(would - r["gold_pts"]) <= 1 and not gold_zero:
                        m2_fix += 1
                    elif gold_zero:
                        m2_break += 1
                # ★반올림 흡수 (2026-07-18 포렌식): gold_pts는 **정수 반올림**돼 저장(예 9.47×2=18.94→18pts)
                #   → gold_pts/amount는 참 rate보다 최대 1/amount 낮다. 판정 = 서브 base_rate로 **포인트 재구성**
                #   (base 또는 base×2=프로모)이 gold_pts와 ±1 이내인가(엔진 tolerance와 동일). rate 직접비교 금지.
                def hits(rate):
                    return abs(r["amount"] * rate - r["gold_pts"]) <= 1
                ok = br is not None and (hits(br) or hits(br * 2) or (r["gold_pts"] == 0 and br == 0))
                cell_ok[cell] += ok
                if not ok:
                    bad.append("%s=%s(gold_pts %.0f/amt %.0f=%.2f)" % (r["category"], v, r["gold_pts"], r["amount"], gr))
            tag = "✓" if not bad else "✗%d" % len(bad)
            m2 = (" [2안 default=%s: 0셀 fix=%d break=%d]" % (card_default, m2_fix, m2_break)) \
                if a.mode2_default else ""
            print("### %s [%d] docs=%d prompt=%dch %s %s%s"
                  % (card, i, len(docs), plen, tag, " ".join(bad[:6]), m2))
        print()

    print("=" * 70)
    tot_ok, tot_n = sum(cell_ok.values()), sum(cell_n.values())
    bycardacc = defaultdict(lambda: [0, 0])
    for cell in cell_n:
        bycardacc[cell[0]][0] += cell_ok[cell]
        bycardacc[cell[0]][1] += cell_n[cell]
    print("★카드별 base_rate 정확:")
    for c, v in sorted(bycardacc.items()):
        print("   %-32s %d/%d = %3.0f%%" % (c, v[0], v[1], 100 * v[0] / max(v[1], 1)))
    print("   %-32s %d/%d = %3.0f%%" % ("[전체]", tot_ok, tot_n, 100 * tot_ok / max(tot_n, 1)))
    print("\n판정: 높으면 §2e(카드당 문서주입) 실증 = 검색부실 우회 · 낮으면 formalize 부하([[45]])·§PROD-2")


if __name__ == "__main__":
    main()
