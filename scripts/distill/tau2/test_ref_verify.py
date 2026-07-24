#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2_REF_VERIFY selftest (C128/C129) — 결정론 참조-검증기 (LLM 0·substring만).
케이스: ①미언급 상점 filed→deny+처방 ②언급 상점→통과 ③id 레코드 못찾음→skip
④비대상 도구→skip ⑤rall22 실사건 재현(Facebook Ads for Marriott dispute→deny).
Run: py -3 test_ref_verify.py"""
import json
import os
import sys
from types import SimpleNamespace as NS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_gate_patch as G  # noqa: E402

A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))


def _find(o, key):
    if isinstance(o, dict):
        if key in o:
            return o[key]
        for v in o.values():
            r = _find(v, key)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = _find(v, key)
            if r is not None:
                return r


SPECS = _find(A2, "ref_verify")

LISTING = ("Found 3 record(s) in 'credit_card_transaction_history':\n"
           "1. Record ID: txn_adea68821a1d\n   transaction_id: txn_adea68821a1d\n"
           "   merchant_name: Marriott Hotels\n   transaction_amount: $167.34\n"
           "2. Record ID: txn_9a72b84326d1\n   transaction_id: txn_9a72b84326d1\n"
           "   merchant_name: Facebook Ads\n   transaction_amount: $203.58\n"
           "3. Record ID: txn_39469d5db822\n   transaction_id: txn_39469d5db822\n"
           "   merchant_name: Costco\n   transaction_amount: $267.34\n")


def um(c):
    return NS(role="user", content=c, tool_calls=None, error=False, id=None)


def tm(c):
    return NS(role="tool", content=c, tool_calls=None, error=False, id="p1")


def file_tc(txn):
    inner = json.dumps({"transaction_id": txn, "card_action": "keep_active"})
    return NS(name="call_discoverable_agent_tool", id="c1",
              arguments={"agent_tool_name": "file_credit_card_transaction_dispute_4829",
                         "arguments": inner})


CUST_MARRIOTT = um("I need to dispute a charge. I stayed at a Marriott hotel but got the wrong room.")


def run(cust, txn, listing=True):
    msgs = [cust]
    if listing:
        msgs.append(tm(LISTING))
    return G._ref_verify_deny(msgs, file_tc(txn), SPECS)


def main():
    ok = True
    # ①rall22 실사건: Marriott 손님인데 Facebook Ads(9a72b) filing → deny
    fb = run(CUST_MARRIOTT, "txn_9a72b84326d1")
    good = fb is not None and "Facebook Ads" in fb and "Marriott" in fb
    print("[%s] rall22_harm_reproduced (Facebook Ads for Marriott -> deny + prescribes Marriott)"
          % ("PASS" if good else "FAIL"))
    ok &= good
    # ②정답 Marriott filing → 통과(no deny)
    fb = run(CUST_MARRIOTT, "txn_adea68821a1d")
    good = fb is None
    print("[%s] gold_marriott_passes (no false-block)" % ("PASS" if good else "FAIL"))
    ok &= good
    # ③다른 미언급 상점(Costco) for Marriott 손님 → deny
    fb = run(CUST_MARRIOTT, "txn_39469d5db822")
    good = fb is not None and "Costco" in fb
    print("[%s] unmentioned_costco_deny" % ("PASS" if good else "FAIL"))
    ok &= good
    # ④id 레코드 못 찾음(listing 없음) → skip(no deny)
    fb = run(CUST_MARRIOTT, "txn_9a72b84326d1", listing=False)
    good = fb is None
    print("[%s] no_listing_skip" % ("PASS" if good else "FAIL"))
    ok &= good
    # ⑤비대상 도구 → skip
    tc = NS(name="call_discoverable_agent_tool", id="c1",
            arguments={"agent_tool_name": "update_transaction_rewards_3847",
                       "arguments": json.dumps({"transaction_id": "txn_9a72b84326d1"})})
    fb = G._ref_verify_deny([CUST_MARRIOTT, tm(LISTING)], tc, SPECS)
    good = fb is None
    print("[%s] non_target_tool_skip" % ("PASS" if good else "FAIL"))
    ok &= good
    # ⑥Costco 손님이 Costco charge filing → 통과(언급 상점)
    fb = run(um("Dispute my Costco charge of $267.34 please."), "txn_39469d5db822")
    good = fb is None
    print("[%s] mentioned_costco_passes" % ("PASS" if good else "FAIL"))
    ok &= good
    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
