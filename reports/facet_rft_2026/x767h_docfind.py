import gzip, glob, os, json, sys
pat = sys.argv[1]
files = sorted(glob.glob("reports/facet_rft_2026/sim_results/*.results.json.gz"))
seen=set(); n=0
for p in files:
    try:
        with gzip.open(p,'rt',encoding='utf-8',errors='replace') as f:
            raw=f.read()
    except Exception:
        continue
    if pat not in raw: continue
    try: d=json.loads(raw)
    except Exception: continue
    for s in d.get("simulations",[]) or []:
        for m in s.get("messages",[]) or []:
            if m.get("role")!="tool": continue
            c=str(m.get("content") or "")
            if pat not in c: continue
            i=c.find(pat)
            seg=c[max(0,i-200):i+3500]
            k=seg[:150]
            if k in seen: continue
            seen.add(k); n+=1
            print("=== %s | task %s" % (os.path.basename(p), s.get("task_id")))
            print(seg); print()
            if n>=6: sys.exit(0)
