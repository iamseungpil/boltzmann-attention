import gzip, glob, os, sys, re
pat = sys.argv[1] if len(sys.argv)>1 else "submit_referral"
d = "reports/facet_rft_2026/sim_results"
files = sorted(glob.glob(os.path.join(d, "*.results.json.gz")))
print("NFILES", len(files))
for p in files:
    try:
        with gzip.open(p, 'rt', encoding='utf-8', errors='replace') as f:
            raw = f.read()
    except Exception as e:
        print("ERR", os.path.basename(p), e); continue
    n = raw.count(pat)
    if n:
        print(n, os.path.basename(p))
