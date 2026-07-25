import sys, gzip, json, tempfile, re
from pathlib import Path
sys.path.insert(0,"/home/woori/scratch/tau2-bench/src")
from tau2.registry import registry
from tau2.data_model.simulation import Results
GZ="/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results/bank_all97_nt1_v2_20260718.results.json.gz"
raw=json.load(gzip.open(GZ))
# work directly on raw dicts (avoid model coercion surprises)
sims={s["task_id"]:s for s in raw["simulations"]}
CAND=["task_032","task_033","task_035","task_046","task_058","task_063","task_070","task_071","task_075","task_099","task_043"]
def events(msgs):
    ev=[]
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            nm=str(tc.get("name") or ""); a=tc.get("arguments") or {}
            if not isinstance(a,dict):
                try: a=json.loads(a)
                except Exception: a={}
            inner=str(a.get("agent_tool_name") or "")
            if nm=="call_discoverable_agent_tool": ev.append(("EXEC",inner))
            elif nm=="unlock_discoverable_agent_tool": ev.append(("UNLOCK",inner))
            else: ev.append(("DIRECT",nm))
    return ev
for tid in CAND:
    s=sims.get(tid)
    if s is None: print("%s : NOT IN RUN"%tid); continue
    msgs=s.get("messages") or []
    ev=events(msgs)
    term=s.get("termination_reason"); ri=s.get("reward_info") or {}
    print("\n=== %s  nmsg=%d  term=%s  db_match=%s"%(tid,len(msgs),term,(ri.get("db_check") or {}).get("db_match")))
    print("   EXEC  :",[n for k,n in ev if k=="EXEC"])
    print("   UNLOCK:",[n for k,n in ev if k=="UNLOCK"])
    dr=[n for k,n in ev if k=="DIRECT" and re.search(r"_\d{3,4}$",n)]
    print("   DIRECT-suffixed:",dr)
    print("   other DIRECT:",sorted(set(n for k,n in ev if k=="DIRECT" and not re.search(r"_\d{3,4}$",n))))
