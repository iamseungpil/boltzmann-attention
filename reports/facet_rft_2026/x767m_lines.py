# -*- coding: utf-8 -*-
import gzip, glob, json, sys
pats = sys.argv[1:]
uniq = {p: {} for p in pats}
for p in sorted(glob.glob("reports/facet_rft_2026/sim_results/*.results.json.gz")):
    try: raw = gzip.open(p,'rt',encoding='utf-8',errors='replace').read()
    except Exception: continue
    if not any(x in raw for x in pats): continue
    try: d = json.loads(raw)
    except Exception: continue
    for s in d.get("simulations",[]) or []:
        for m in s.get("messages",[]) or []:
            if m.get("role")!="tool": continue
            c=str(m.get("content") or "")
            for pat in pats:
                if pat not in c: continue
                for ln in c.splitlines():
                    if pat in ln:
                        uniq[pat][ln.strip()] = uniq[pat].get(ln.strip(),0)+1
for pat in pats:
    print("### %r  distinct=%d" % (pat, len(uniq[pat])))
    for ln, n in sorted(uniq[pat].items(), key=lambda kv:-kv[1]):
        print("   (%d) %s" % (n, ln[:300]))
    print()
