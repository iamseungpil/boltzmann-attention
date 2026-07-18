#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""격리 서브 e2e 궤적 per-step 포렌식 (무료·[[08]]) — 집계→결론 직행 금지.
태스크별: 종료사유 · reward · 격리 서브 발화(라운드/getter/operand) · producer base_rate 정확도(전수)
 · producer 반환 discrepant vs gold dispute(오탐/누락) · 판정불가행(under-action 계측).

Run: python3 bank_iso_forensic.py <tag> [<tag2> ...]   # gz 태그(sim_results 상대 or 절대)
"""
import json
import gzip
import glob
import os
import re
import sys
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import bank_rate_f1_gate_probe as P  # noqa: E402

DOM = P.DOM_DEFAULT
SIMDIR = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")


def _disputes():
    tasks = json.load(open(os.path.join(DOM, "tasks.json"), encoding="utf-8"))
    out = collections.defaultdict(set)
    for t in tasks:
        for a in (t.get("evaluation_criteria") or {}).get("actions", []) or []:
            arg = a.get("arguments") or {}
            if a.get("name") == "call_discoverable_user_tool" and "dispute" in str(arg.get("discoverable_tool_name")):
                inner = arg.get("arguments")
                if isinstance(inner, str):
                    inner = json.loads(inner)
                out[t["id"]].add(inner["transaction_id"])
    return out


def _load(tag):
    if os.path.isabs(tag) and os.path.exists(tag):
        p = tag
    else:
        hits = glob.glob(os.path.join(SIMDIR, tag + "*results.json.gz"))
        if not hits:
            hits = glob.glob(tag + "*results.json.gz")
        p = hits[0]
    op = gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")
    return json.load(op)


def main():
    gold = {r["transaction_id"]: r for r in P.load_gold(DOM)}
    disp = _disputes()
    for tag in sys.argv[1:]:
        R = _load(tag)
        print("\n" + "#" * 78 + "\n# %s" % tag)
        term = collections.Counter()
        for s in sorted(R["simulations"], key=lambda x: str(x.get("task_id"))):
            tid = s.get("task_id")
            ri = s.get("reward_info") or {}
            term[s.get("termination_reason")] += 1
            print("\n=== %s | reward=%s | term=%s | msgs=%d | db=%s"
                  % (tid, ri.get("reward"), s.get("termination_reason"),
                     len(s.get("messages") or []), (ri.get("db_check") or {}).get("db_match")))
            if not (s.get("messages") or []):
                print("   (메시지 0 — infra_error, 판정 제외)")
                continue
            for m in s["messages"]:
                for tc in (m.get("tool_calls") or []):
                    fn = tc.get("function") or tc
                    if fn.get("name") != "get_reward_discrepancies":
                        continue
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        args = json.loads(args)
                    txs = args.get("transactions")
                    if isinstance(txs, str):
                        txs = json.loads(txs)
                    bad, strf = 0, 0
                    for t in txs:
                        g = gold.get(t.get("transaction_id"))
                        amt, br = t.get("transaction_amount"), t.get("base_rate")
                        if not (isinstance(amt, (int, float)) and isinstance(br, (int, float))):
                            strf += 1
                            continue
                        if not g:
                            continue
                        rate = P.engine_rate(br, bool(t.get("promo_start")), t.get("promo_mult", 1),
                                             t.get("account_open"), t.get("transaction_date"),
                                             t.get("promo_window_months", 6), t.get("promo_start"),
                                             t.get("promo_end"))
                        if abs(g["amount"] * float(rate) - g["gold_pts"]) > 1:
                            bad += 1
                    print("   ★producer 인자 %d행 · base_rate 틀림 %d · 문자열operand %d(판정불가)"
                          % (len(txs), bad, strf))
                if m.get("role") == "tool" and "require a cash back dispute" in str(m.get("content")):
                    got = set(re.findall(r"txn_[0-9a-f]+", str(m.get("content"))))
                    g = disp.get(tid, set())
                    print("   ▶반환 discrepant=%d · gold=%d · 오탐=%s · 누락=%s"
                          % (len(got), len(g), sorted(x[-8:] for x in got - g),
                             sorted(x[-8:] for x in g - got)))
        print("\n   종료사유 분포:", dict(term))


if __name__ == "__main__":
    main()
