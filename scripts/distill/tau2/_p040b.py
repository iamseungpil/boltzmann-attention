# -*- coding: utf-8 -*-
"""040 분쟁 8건 — gold 행 ↔ 궤적 호출을 **transaction_id 로 짝지어** 인자 축자 대조.
t7354 grpB1 한 런만. 판단 0 · 집계 0 — 값만 인쇄한다."""
import sys, collections
sys.path.insert(0, ".")
import t2_forensic as F

TOOL = "file_credit_card_transaction_dispute_4829"
for tag in ["bank_t7354_grpB1_20260825"]:
    for sim in F.sims(tag, ".results.json.gz"):
        if not str(sim.get("task_id")).endswith("040"):
            continue
        d = F.action_diff(sim, tag=tag)
        print("\n########## %s trial=%s reward=%s gold=%d matched=%d" %
              (tag, sim.get("trial"), d["reward"], d["n_gold"], d["n_matched"]))
        golds = [g for g in d["gold"] if g["inner"] == TOOL]
        tried = [t for t in d["tried"] if t["inner"] == TOOL]
        print("gold 분쟁 %d 건 · 궤적 호출 %d 건" % (len(golds), len(tried)))
        by_txn = collections.defaultdict(list)
        for t in tried:
            by_txn[(t.get("args") or {}).get("transaction_id")].append(t)
        for g in golds:
            ga = g.get("args") or {}
            txn = ga.get("transaction_id")
            cands = by_txn.get(txn, [])
            print("\n-- gold txn=%s  bench_match=%s  같은 txn 호출=%d" % (txn, g["bench_match"], len(cands)))
            if not cands:
                print("   (그 txn 으로 부른 적 없음)  gold args=%s" % ga)
                continue
            for t in cands:
                ta = t.get("args") or {}
                diffs = ["%s: gold=%r got=%r" % (k, ga.get(k), ta.get(k))
                         for k in sorted(set(ga) | set(ta)) if ga.get(k) != ta.get(k)]
                print("   msg=%s ok=%s deny=%s  diff=%d %s" %
                      (t.get("msg_i"), t["ok"], t.get("deny"), len(diffs), "" if diffs else "**전부 일치**"))
                for x in diffs:
                    print("      " + x)
        # gold 에 없는 txn 으로 부른 것
        gtxn = {(g.get("args") or {}).get("transaction_id") for g in golds}
        extra = [t for t in tried if (t.get("args") or {}).get("transaction_id") not in gtxn]
        print("\n-- gold 명단 밖 txn 호출 %d 건" % len(extra))
        for t in extra:
            ta = t.get("args") or {}
            print("   msg=%s ok=%s deny=%s txn=%r last4=%r reason=%r" %
                  (t.get("msg_i"), t["ok"], t.get("deny"), ta.get("transaction_id"),
                   ta.get("card_last_4_digits"), ta.get("dispute_reason")))
