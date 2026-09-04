import gzip, glob, os, sys, re
pats = sys.argv[1:]
files = sorted(glob.glob("reports/facet_rft_2026/sim_results/*.results.json.gz"))
tot={p:0 for p in pats}
where={p:set() for p in pats}
for f in files:
    try:
        raw=gzip.open(f,'rt',encoding='utf-8',errors='replace').read()
    except Exception: continue
    for p in pats:
        n=raw.count(p)
        if n:
            tot[p]+=n; where[p].add(os.path.basename(f))
for p in pats:
    print("PAT %r  total=%d  files=%d" % (p, tot[p], len(where[p])))
    for w in sorted(where[p])[:5]: print("   ", w)
