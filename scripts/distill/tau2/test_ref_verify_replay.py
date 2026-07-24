#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2_REF_VERIFY 실궤적 replay (C128/C129) — rall19-22의 file_dispute 호출을 라이브 엔진
술어(_ref_verify_deny)로 재판정. 기대: wrong-pick=deny·gold=pass(무회귀). efiso_detmatch_proof와
같은 결론(슬립 8/8·false-block 0)을 *라이브 엔진 경로*로 재현.
Run: py -3 test_ref_verify_replay.py"""
import gzip
import json
import os
import re
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
SIMR = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
INST = [("bank_rall19_treat_20260723.results.json.gz", "task_039", 0),
        ("bank_rall20a_20260723.results.json.gz", "task_031", 0),
        ("bank_rall20a_20260723.results.json.gz", "task_039", 0),
        ("bank_rall21a_20260724.results.json.gz", "task_039", 0),
        ("bank_rall22a_20260724.results.json.gz", "task_031", 0)]


def msgs_ns(sim):
    return [NS(role=m.get("role"), content=m.get("content"), error=m.get("error"))
            for m in sim.get("messages", [])]


def file_tc(txn):
    inner = json.dumps({"transaction_id": txn, "card_action": "keep_active"})
    return NS(name="call_discoverable_agent_tool", id="c1",
              arguments={"agent_tool_name": "file_credit_card_transaction_dispute_4829",
                         "arguments": inner})


def gold_filed(sim):
    gold, filed = [], []
    for ac in (sim.get("reward_info") or {}).get("action_checks") or []:
        a = ac.get("action") or {}
        if "file_credit_card_transaction_dispute" in str(a.get("arguments", "")):
            mm = re.search(r"txn_[0-9a-f]+", str(a.get("arguments")))
            if mm:
                gold.append(mm.group())
    for m in sim.get("messages", []):
        for tc in (m.get("tool_calls") or []):
            aa = str(tc.get("arguments", ""))
            if "file_credit_card_transaction_dispute" in aa and tc.get("name") == "call_discoverable_agent_tool":
                mm = re.search(r"txn_[0-9a-f]+", aa)
                if mm:
                    filed.append(mm.group())
    return gold, list(dict.fromkeys(filed))


def main():
    caught = wrong = gpass = gtot = 0
    for gz, task, tr in INST:
        d = json.load(gzip.open(os.path.join(SIMR, gz)))
        sim = next(s for s in d["simulations"]
                   if str(s.get("task_id")) == task and s.get("trial") == tr)
        M = msgs_ns(sim)
        gold, filed = gold_filed(sim)
        for tid in [f for f in filed if f not in gold]:
            wrong += 1
            deny = G._ref_verify_deny(M, file_tc(tid), SPECS)
            caught += 1 if deny else 0
            print("  %s WRONG %s -> %s" % (task, tid[-6:], "DENY" if deny else "MISS"))
        for tid in gold:
            gtot += 1
            deny = G._ref_verify_deny(M, file_tc(tid), SPECS)
            gpass += 0 if deny else 1
            if deny:
                print("  %s GOLD  %s -> FALSE-BLOCK: %s" % (task, tid[-6:], deny[:80]))
    print("\n슬립 검출(deny): %d/%d · gold 통과(no false-block): %d/%d" % (caught, wrong, gpass, gtot))
    ok = (caught == wrong and gpass == gtot)
    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
