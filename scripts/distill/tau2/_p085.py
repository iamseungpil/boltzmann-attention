# -*- coding: utf-8 -*-
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import t2_forensic as F
TASKS = {"085","040","016","057","063","072","074","055","079","094"}
for tag in ("bank_t7348_halfA_20260824", "bank_t7348_halfB_20260824"):
    try:
        ss = F.sims(tag, ".results.json.gz")
    except Exception as e:
        print("!! %s %r" % (tag, e)); continue
    for s in ss:
        tid = str(s.get("task_id"))
        if tid.replace("task_","") not in TASKS: continue
        ad = F.action_diff(s, tag=tag)
        print("=== %s %s reward=%s" % (tid, tag, ad.get("reward")))
        for r in ad.get("rows", []):
            b = r.get("blocked")
            print("   m=%s ex=%s nm=%s | %s.%s" % (r.get("bench_match"), r.get("called_exact"),
                  r.get("called_name"), r.get("outer"), r.get("inner")))
            if b:
                print("      BLOCKED deny=%r marker=%r msg=%s" % (str(b.get("deny"))[:300],
                      b.get("marker"), b.get("msg_i")))
