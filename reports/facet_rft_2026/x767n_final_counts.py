# -*- coding: utf-8 -*-
import gzip, glob, json, os
files = sorted(glob.glob("reports/facet_rft_2026/sim_results/*.results.json.gz"))
print("TOTAL_GZ", len(files))
withpat = 0; toolhits = {}; anyrole = {}
for p in files:
    try: raw = gzip.open(p,'rt',encoding='utf-8',errors='replace').read()
    except Exception: continue
    if "submit_referral" not in raw: continue
    withpat += 1
    try: d = json.loads(raw)
    except Exception: continue
    for s in d.get("simulations",[]) or []:
        for m in s.get("messages",[]) or []:
            c = str(m.get("content") or "")
            if "submit_referral" not in c: continue
            r = m.get("role")
            anyrole[r] = anyrole.get(r,0)+1
            if r == "tool":
                toolhits[c.strip()[:200]] = toolhits.get(c.strip()[:200],0)+1
print("FILES_WITH_PAT", withpat)
print("MSG_COUNT_BY_ROLE", anyrole)
print("DISTINCT_TOOL_ROLE_STRINGS", len(toolhits))
for k,v in sorted(toolhits.items(), key=lambda kv:-kv[1]):
    print("   (%d) %r" % (v,k))
