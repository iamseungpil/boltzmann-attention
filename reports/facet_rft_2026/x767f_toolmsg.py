import gzip, glob, os, json, sys
pat = sys.argv[1]
files = sys.argv[2:] or sorted(glob.glob("reports/facet_rft_2026/sim_results/*.results.json.gz"))
seen=set()
for p in files:
    try:
        with gzip.open(p,'rt',encoding='utf-8',errors='replace') as f:
            d=json.load(f)
    except Exception as e:
        continue
    for s in d.get("simulations",[]) or []:
        for m in s.get("messages",[]) or []:
            if m.get("role")!="tool": continue
            c=str(m.get("content") or "")
            if pat in c:
                k=c[:200]
                if k in seen: continue
                seen.add(k)
                print("=== %s | sim %s | task %s" % (os.path.basename(p), s.get("id","")[:8], s.get("task_id")))
                print(c[:2500])
                print()
