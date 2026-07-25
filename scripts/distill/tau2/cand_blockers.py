import sys, gzip, json, tempfile, re
from pathlib import Path
from collections import Counter
sys.path.insert(0,"/home/woori/scratch/tau2-bench/src")
from tau2.registry import registry
from tau2.data_model.simulation import Results
GZ="/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results/bank_all97_nt1_v2_20260718.results.json.gz"
raw=json.load(gzip.open(GZ)); tmp=tempfile.NamedTemporaryFile("w",suffix=".json",delete=False); json.dump(raw,tmp); tmp.close()
results=Results.load(Path(tmp.name))
env_ctor=registry.get_env_constructor("banking_knowledge")
tasks={t.id:t for t in registry.get_tasks_loader("banking_knowledge")()}
CAND=["task_032","task_033","task_035","task_046",      # REG_ONLY
      "task_044","task_058","task_063","task_070","task_071","task_075","task_099",  # near-miss
      "task_043"]
def init_args(t):
    ist=getattr(t,"initial_state",None)
    if ist is None: return (None,None,[])
    return (ist.initialization_data, ist.initialization_actions, list(ist.message_history or []))
def diff(g,p,path="",out=None):
    if type(g)!=type(p): out.append(("TYPE",path,repr(g)[:90],repr(p)[:90])); return
    if isinstance(g,dict):
        for k in sorted(set(g)|set(p),key=str):
            if k not in g: out.append(("ONLY-PRED",path+"."+str(k),"",repr(p[k])[:110]))
            elif k not in p: out.append(("ONLY-GOLD",path+"."+str(k),repr(g[k])[:110],""))
            else: diff(g[k],p[k],path+"."+str(k),out)
    elif isinstance(g,list):
        if len(g)!=len(p): out.append(("LEN",path,str(len(g)),str(len(p))))
        for i in range(min(len(g),len(p))): diff(g[i],p[i],"%s[%d]"%(path,i),out)
    else:
        if g!=p: out.append(("DIFF",path,repr(g)[:90],repr(p)[:90]))
def execpath(msgs):
    """returns list of (kind, name) with kind in EXEC/UNLOCK/DIRECT"""
    ev=[]
    for m in msgs:
        tcs=(m.get("tool_calls") if isinstance(m,dict) else getattr(m,"tool_calls",None)) or []
        for tc in tcs:
            nm=(tc.get("name") if isinstance(tc,dict) else getattr(tc,"name",None)) or ""
            a=(tc.get("arguments") if isinstance(tc,dict) else getattr(tc,"arguments",None)) or {}
            if not isinstance(a,dict):
                try: a=json.loads(a)
                except Exception: a={}
            if nm=="call_discoverable_agent_tool": ev.append(("EXEC",str(a.get("agent_tool_name") or "")))
            elif nm=="unlock_discoverable_agent_tool": ev.append(("UNLOCK",str(a.get("agent_tool_name") or "")))
            else: ev.append(("DIRECT",nm))
    return ev
for sim in results.simulations:
    if sim.task_id not in CAND: continue
    t=tasks[sim.task_id]; a,b,c0=init_args(t)
    gold=env_ctor(retrieval_variant="no_knowledge"); gold.set_state(a,b,list(c0))
    goldacts=[]
    for act in (t.evaluation_criteria.actions or []):
        goldacts.append((act.name, json.dumps(act.arguments)[:110]))
        try: gold.make_tool_call(tool_name=act.name, requestor=act.requestor, **act.arguments)
        except Exception as e: pass
    pred=env_ctor(retrieval_variant="no_knowledge"); pred.set_state(a,b,list(sim.messages))
    match=gold.tools.get_db_hash()==pred.tools.get_db_hash()
    out=[]; diff(gold.tools.db.model_dump(), pred.tools.db.model_dump(),"",out)
    reg=[d for d in out if d[1].startswith(".agent_discoverable_tools")]
    st =[d for d in out if not d[1].startswith(".agent_discoverable_tools")]
    ev=execpath(sim.messages)
    direct=[n for k,n in ev if k=="DIRECT" and re.search(r"_\d{3,4}$",n)]
    print("\n===== %s  db_match=%s  reg=%d state=%d  nmsg=%d ====="%(sim.task_id,match,len(reg),len(st),len(sim.messages)))
    if reg:
        print("  MISSING REGISTRATIONS:")
        for d in reg[:8]:
            m=re.search(r"'tool_name': '([^']+)'",d[2] or ""); print("    -",m.group(1) if m else d[2][:80])
    if st:
        print("  STATE DIFFS:")
        for d in st[:8]: print("    %s %s | gold=%s pred=%s"%(d[0],d[1][:70],d[2][:60],d[3][:60]))
    if direct: print("  DIRECT suffixed calls:",sorted(set(direct)))
    print("  gold actions:", [g[0] for g in goldacts])
    print("  agent EXEC  :", [n for k,n in ev if k=="EXEC"])
