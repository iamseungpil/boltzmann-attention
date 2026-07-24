#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""결정론 참조-검증기 오프라인 증명 (C128·사용자 "sub-agent 격리" 제안의 결정론 종착점).
가설: 잘못 filed된 transaction_id의 (merchant, amount)는 손님 발화에 없고, gold의 것은 있다
→ 순수 결정론 검증기(LLM 0)가 모든 슬립을 잡는다(전사슬립 원천불가). 4 사례 전수.
Run: py -3 efiso_detmatch_proof.py"""
import gzip
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SIMR = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")

INST = [("bank_rall19_treat_20260723.results.json.gz", "task_039", 0, "r19.039"),
        ("bank_rall20a_20260723.results.json.gz", "task_031", 0, "r20.031"),
        ("bank_rall20a_20260723.results.json.gz", "task_039", 0, "r20.039"),
        ("bank_rall21a_20260724.results.json.gz", "task_039", 0, "r21.039")]


def sim_of(gz, task, tr):
    d = json.load(gzip.open(os.path.join(SIMR, gz)))
    for s in d.get("simulations", []):
        if str(s.get("task_id")) == task and s.get("trial") == tr:
            return s


def listing(sim):
    best = ""
    for m in sim.get("messages", []):
        c = str(m.get("content") or "")
        if m.get("role") == "tool" and "credit_card_transaction_history" in c and len(c) > len(best):
            best = c
    return best


def row_attr(lst, tid):
    k = lst.find(tid)
    if k < 0:
        return None
    blk = lst[k:k + 320]
    mer = re.search(r"merchant_name: ([^\n]*)", blk)
    amt = re.search(r"transaction_amount: \$?([0-9,]+\.[0-9]{2})", blk)
    return (mer.group(1).strip() if mer else None,
            amt.group(1).replace(",", "") if amt else None)


def customer_text(sim):
    # 손님 발화만·txn_id 제거(오염). merchant/amount만 남김.
    raw = "\n".join(str(m.get("content") or "") for m in sim.get("messages", [])
                    if m.get("role") == "user")
    return re.sub(r"txn_[0-9a-f]+", "", raw)


def mentioned(ctext, merchant, amount):
    """손님이 이 (merchant, amount)를 언급했나 — 결정론(substring·LLM 0)."""
    if not merchant or not amount:
        return None
    m_ok = merchant.lower() in ctext.lower()
    a_ok = (amount in ctext) or (("%.2f" % float(amount)) in ctext)
    return m_ok and a_ok


def gold_filed(sim):
    gold, filed = [], []
    for ac in (sim.get("reward_info") or {}).get("action_checks") or []:
        a = ac.get("action") or {}
        if "file_credit_card_transaction_dispute" in str(a.get("arguments", "")):
            mm = re.search(r"txn_[0-9a-f]+", str(a.get("arguments")))
            if mm:
                gold.append(mm.group())
    for msg in sim.get("messages", []):
        for tc in (msg.get("tool_calls") or []):
            aa = str(tc.get("arguments", ""))
            if "file_credit_card_transaction_dispute" in aa and tc.get("name") == "call_discoverable_agent_tool":
                mm = re.search(r"txn_[0-9a-f]+", aa)
                if mm:
                    filed.append(mm.group())
    return gold, list(dict.fromkeys(filed))


def main():
    print("결정론 검증기: filed id의 (merchant,amount)가 손님 발화에 있나? (없으면=슬립 검출)")
    tot_wrong = tot_caught = tot_gold = tot_gold_pass = 0
    for gz, task, tr, label in INST:
        sim = sim_of(gz, task, tr)
        lst = listing(sim)
        ctext = customer_text(sim)
        gold, filed = gold_filed(sim)
        wrong = [f for f in filed if f not in gold]
        print("\n== %s  gold=%d filed=%d wrong=%d" % (label, len(gold), len(filed), len(wrong)))
        for tid in wrong:
            mer, amt = row_attr(lst, tid) or (None, None)
            men = mentioned(ctext, mer, amt)
            caught = (men is False)
            tot_wrong += 1
            tot_caught += 1 if caught else 0
            print("   WRONG %s = %s $%s | customer_mentioned=%s -> %s"
                  % (tid[-6:], mer, amt, men, "CAUGHT(deny)" if caught else "MISSED"))
        for tid in gold:
            mer, amt = row_attr(lst, tid) or (None, None)
            men = mentioned(ctext, mer, amt)
            ok = (men is True)
            tot_gold += 1
            tot_gold_pass += 1 if ok else 0
            if not ok:
                print("   GOLD  %s = %s $%s | customer_mentioned=%s -> FALSE-BLOCK-RISK"
                      % (tid[-6:], mer, amt, men))
    print("\n==== SUMMARY (merchant+amount) ====")
    print("슬립 검출율(deny): %d/%d" % (tot_caught, tot_wrong))
    print("gold 통과율(no false-block): %d/%d" % (tot_gold_pass, tot_gold))

    # ★merchant-only 변형: 손님이 그 상점을 언급했나(근사액 false-block 회피).
    mc = mw = gc = gw = 0
    for gz, task, tr, label in INST:
        sim = sim_of(gz, task, tr)
        lst = listing(sim)
        ct = customer_text(sim).lower()
        gold, filed = gold_filed(sim)
        for tid in [f for f in filed if f not in gold]:
            mer = (row_attr(lst, tid) or (None, None))[0]
            mw += 1
            mc += 1 if (mer and mer.lower() not in ct) else 0
        for tid in gold:
            mer = (row_attr(lst, tid) or (None, None))[0]
            gw += 1
            gc += 1 if (mer and mer.lower() in ct) else 0
    print("\n==== SUMMARY (merchant-only) ====")
    print("슬립 검출율(deny): %d/%d" % (mc, mw))
    print("gold 통과율(no false-block): %d/%d" % (gc, gw))


if __name__ == "__main__":
    main()
