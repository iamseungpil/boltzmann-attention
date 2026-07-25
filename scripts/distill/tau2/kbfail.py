import gzip, json, re
from collections import Counter
GZ="/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results/bank_all97_nt1_v2_20260718.results.json.gz"
raw=json.load(gzip.open(GZ))
sims=raw["simulations"]
tot=len(sims); aff=[]; kbuse=Counter(); errcnt=Counter()
for s in sims:
    tid=s["task_id"]; msgs=s.get("messages") or []
    used=set(); nerr=0; ncall=0
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            nm=str(tc.get("name"))
            if nm.startswith("KB_search"):
                used.add(nm); ncall+=1
        if m.get("role")=="tool":
            c=str(m.get("content") or "")
            if "Missing credentials" in c or "OPENAI_API_KEY" in c:
                nerr+=1
    for u in used: kbuse[u]+=1
    if nerr:
        aff.append((tid,nerr,ncall,sorted(used)))
        errcnt[tuple(sorted(used))]+=1
print("total sims:", tot)
print("KB tool usage (tasks):", dict(kbuse))
print("\n*** sims with KB credential errors: %d / %d (%.0f%%)"%(len(aff),tot,100.0*len(aff)/tot))
for tid,nerr,ncall,used in aff[:40]:
    print("   %s  kb_errors=%d kb_calls=%d %s"%(tid,nerr,ncall,used))
# cross with db_match
pas=[s["task_id"] for s in sims if ((s.get("reward_info") or {}).get("db_check") or {}).get("db_match")]
affids=set(t for t,_,_,_ in aff)
print("\ndb_match PASS tasks:", sorted(pas))
print("PASS tasks that had KB errors:", sorted(set(pas)&affids))
print("\n=== dense-vs-bm25 by task (first 12) ===")
for s in sims[:12]:
    used=set()
    for m in (s.get("messages") or []):
        for tc in (m.get("tool_calls") or []):
            if str(tc.get("name")).startswith("KB_search"): used.add(str(tc.get("name")))
    print("   %s %s"%(s["task_id"], sorted(used) or "no-KB"))
