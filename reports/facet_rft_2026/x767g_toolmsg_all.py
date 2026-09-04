import gzip, glob, os, json, sys
pat = sys.argv[1]
files = sorted(glob.glob("reports/facet_rft_2026/sim_results/*.results.json.gz"))
seen=set(); nfound=0; nfiles=0
for p in files:
    try:
        with gzip.open(p,'rt',encoding='utf-8',errors='replace') as f:
            raw=f.read()
    except Exception:
        continue
    if pat not in raw:
        continue
    nfiles+=1
    try:
        d=json.loads(raw)
    except Exception:
        continue
    for s in d.get("simulations",[]) or []:
        for m in s.get("messages",[]) or []:
            if m.get("role")!="tool": continue
            c=str(m.get("content") or "")
            if pat in c:
                nfound+=1
                k=c[:200]
                if k in seen: continue
                seen.add(k)
                print("=== %s | task %s" % (os.path.basename(p), s.get("task_id")))
                print(c[:2500]); print()
print("SCANNED_FILES_WITH_PAT", nfiles, "TOOLMSG_HITS", nfound)
