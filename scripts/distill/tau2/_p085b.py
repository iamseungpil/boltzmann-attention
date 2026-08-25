# -*- coding: utf-8 -*-
"""085 — gold 행 ↔ 궤적 호출을 자연키로 짝지어 인자 축자 대조. t7354 한 런만."""
import sys, collections
sys.path.insert(0, ".")
import t2_forensic as F

for tag in ["bank_t7354_grpA1_20260825", "bank_t7354_grpB2_20260825"]:
    for sim in F.sims(tag, ".results.json.gz"):
        if not str(sim.get("task_id")).endswith("085"):
            continue
        d = F.action_diff(sim, tag=tag)
        print("\n########## %s trial=%s reward=%s basis=%s gold=%d matched=%d" %
              (tag, sim.get("trial"), d["reward"], d["basis"], d["n_gold"], d["n_matched"]))
        tried = collections.defaultdict(list)
        for t in d["tried"]:
            tried[(t["outer"], t["inner"])].append(t)
        for g in d["rows"]:
            ga = g.get("args") or {}
            key = (g["outer"], g["inner"])
            print("\n-- GOLD %s.%s match=%s exact=%s" % (g["outer"], g["inner"], g["bench_match"], g["called_exact"]))
            if not g["bench_match"]:
                print("   gold args=%s" % ({k: v for k, v in ga.items()},))
            for t in tried.get(key, []):
                ta = t.get("args") or {}
                diffs = ["%s: gold=%r got=%r" % (k, ga.get(k), ta.get(k))
                         for k in sorted(set(ga) | set(ta)) if ga.get(k) != ta.get(k)]
                print("   msg=%s ok=%s deny=%s diff=%d %s" %
                      (t.get("msg_i"), t["ok"], t.get("deny"), len(diffs), "**전부 일치**" if not diffs else ""))
                for x in diffs[:12]:
                    print("      " + x)
        # gold 에 없는 이름으로 부른 변이
        gk = {(g["outer"], g["inner"]) for g in d["rows"]}
        extra = [t for t in d["tried"] if (t["outer"], t["inner"]) not in gk and t["ok"]]
        print("\n-- gold 명단 밖 성공 호출 %d 건" % len(extra))
        for t in extra[:20]:
            print("   msg=%s %s.%s args=%s" % (t.get("msg_i"), t["outer"], t["inner"], t.get("args")))
