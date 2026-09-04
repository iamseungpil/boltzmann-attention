import gzip, glob, os, json, sys
pat = sys.argv[1]; maxn=int(sys.argv[2]) if len(sys.argv)>2 else 5
files = sorted(glob.glob("reports/facet_rft_2026/sim_results/*.results.json.gz"))
seen=set(); n=0
for p in files:
    try: raw=gzip.open(p,'rt',encoding='utf-8',errors='replace').read()
    except Exception: continue
    if pat not in raw: continue
    try: d=json.loads(raw)
    except Exception: continue
    for s in d.get("simulations",[]) or []:
        for m in s.get("messages",[]) or []:
            if m.get("role")!="tool": continue
            c=str(m.get("content") or "")
            if pat not in c: continue
            i=c.find(pat)
            seg=c[max(0,i-1800):i+1200]
            k=seg[-400:]
            if k in seen: continue
            seen.add(k); n+=1
            print("=== %s | task %s" % (os.path.basename(p), s.get("task_id")))
            print(seg); print()
            if n>=maxn: sys.exit(0)
