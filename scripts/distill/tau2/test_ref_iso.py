#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2_REF_ISO selftest (C124/C125) — 격리 재선택 치환 로직 검증 (la=stub·모델 불요).
케이스: ①오선택→정답 치환 ②정답 유지 ③UNSURE no-op ④목록-밖 답(날조) no-op
⑤listing 부재 no-op ⑥비대상 도구 no-op. Run: py -3 test_ref_iso.py"""
import json
import os
import sys
from types import SimpleNamespace as NS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_gate_patch as G  # noqa: E402

A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))


def _find(o):
    if isinstance(o, dict):
        if "ref_iso" in o:
            return o["ref_iso"]
        for v in o.values():
            r = _find(v)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = _find(v)
            if r is not None:
                return r


SPECS = _find(A2)

LISTING = ("Found 3 record(s) in 'credit_card_transaction_history':\n"
           "1. Record ID: txn_39469d5db822\n   transaction_id: txn_39469d5db822\n"
           "   merchant_name: Costco\n   transaction_amount: $267.34\n   transaction_date: 11/04/2025\n"
           "2. Record ID: txn_41735bd2d06d\n   transaction_id: txn_41735bd2d06d\n"
           "   merchant_name: Home Depot\n   transaction_amount: $456.78\n   transaction_date: 11/04/2025\n")


def mk_msgs(with_listing=True):
    msgs = [NS(role="user", content="Dispute my Costco $267.34 charge on 11/04.",
               tool_calls=None, error=False, id=None),
            NS(role="assistant", content=None, error=False, id=None,
               tool_calls=[NS(name="get_credit_card_transactions_by_user",
                              arguments={"user_id": "u1"}, id="p1")])]
    if with_listing:
        msgs.append(NS(role="tool", content=LISTING, tool_calls=None, error=False, id="p1"))
    return msgs


def mk_am(txn, tool="call_discoverable_agent_tool",
          agent_tool="file_credit_card_transaction_dispute_4829"):
    inner = json.dumps({"transaction_id": txn, "card_action": "keep_active"})
    return NS(content=None, tool_calls=[
        NS(name=tool, id="c1", arguments={"agent_tool_name": agent_tool, "arguments": inner})])


class StubLA:
    def __init__(self, answer):
        self.answer = answer

    def generate(self, **kw):
        return NS(content=self.answer)


def cur_txn(am):
    return json.loads(am.tool_calls[0].arguments["arguments"])["transaction_id"]


def run(answer, txn="txn_41735bd2d06d", with_listing=True, am=None):
    slf = NS(llm="m", llm_args={})
    am = am or mk_am(txn)
    G._ref_iso_repair(slf, StubLA(answer), lambda **k: NS(role="user", **k),
                      mk_msgs(with_listing), am, SPECS)
    return am


def main():
    ok = True
    cases = [
        ("switch_wrong_to_gold", run("txn_39469d5db822"), "txn_39469d5db822"),
        ("keep_correct", run("txn_39469d5db822", txn="txn_39469d5db822"), "txn_39469d5db822"),
        ("unsure_noop", run("UNSURE"), "txn_41735bd2d06d"),
        ("fabricated_answer_noop", run("txn_a1b2c3d4e502"), "txn_41735bd2d06d"),
        ("no_listing_noop", run("txn_39469d5db822", with_listing=False), "txn_41735bd2d06d"),
    ]
    for name, am, want in cases:
        got = cur_txn(am)
        st = "PASS" if got == want else "FAIL"
        ok &= (got == want)
        print("[%s] %s -> %s" % (st, name, got))
    am = run("txn_39469d5db822", am=mk_am("txn_41735bd2d06d", agent_tool="update_transaction_rewards_3847"))
    got = cur_txn(am)
    st = "PASS" if got == "txn_41735bd2d06d" else "FAIL"
    ok &= (got == "txn_41735bd2d06d")
    print("[%s] non_target_tool_noop -> %s" % (st, got))
    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
