# -*- coding: utf-8 -*-
"""040 남은 인자 결손 관측 — t7354 **한 런만**. gold ↔ 궤적 인자 축자 대조(판단 0)."""
import sys, collections
sys.path.insert(0, ".")
import t2_forensic as F

TAGS = ["bank_t7354_grpA1_20260825", "bank_t7354_grpA2_20260825",
        "bank_t7354_grpA3_20260825", "bank_t7354_grpA4_20260825",
        "bank_t7354_grpB1_20260825", "bank_t7354_grpB2_20260825"]

for tag in TAGS:
    try:
        ss = F.sims(tag, ".results.json.gz")
    except Exception as e:
        print("[skip] %s %s" % (tag, e)); continue
    for sim in ss:
        tid = str(sim.get("task_id") or "?")
        if not tid.endswith("040"):
            continue
        d = F.action_diff(sim, tag=tag)
        print("=== %s task=%s trial=%s reward=%s basis=%s gold=%d matched=%d" %
              (tag, tid, sim.get("trial"), d["reward"], d["basis"], d["n_gold"], d["n_matched"]))
        tried_by_name = collections.defaultdict(list)
        for t in d["tried"]:
            tried_by_name[(t["outer"], t["inner"])].append(t)
        for g in d["rows"]:
            ga = g.get("args") or {}
            cands = tried_by_name.get((g["outer"], g["inner"]), [])
            print("  GOLD %s.%s match=%s exact=%s  cands=%d" %
                  (g["outer"], g["inner"], g["bench_match"], g["called_exact"], len(cands)))
            for t in cands:
                ta = t.get("args") or {}
                diffs = ["%s: gold=%r got=%r" % (k, ga.get(k), ta.get(k))
                         for k in sorted(set(ga) | set(ta)) if ga.get(k) != ta.get(k)]
                if not diffs:
                    print("     got ok=%s  **인자 전부 일치**" % t["ok"]); continue
                print("     got ok=%s deny=%s msg=%s diff=%d" % (t["ok"], t.get("deny"), t.get("msg_i"), len(diffs)))
                for x in diffs:
                    print("        " + x)
