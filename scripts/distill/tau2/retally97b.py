import sys, gzip, json, tempfile, re
from pathlib import Path
from collections import Counter
sys.path.insert(0,"/home/woori/scratch/tau2-bench/src")
from tau2.registry import registry
from tau2.data_model.simulation import Results
GZ="/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results/bank_all97_nt1_v2_20260718.results.json.gz"
raw=json.load(gzip.open(GZ))
tmp=tempfile.NamedTemporaryFile("w",suffix=".json",delete=False); json.dump(raw,tmp); tmp.close()
results=Results.load(Path(tmp.name))
env_ctor=registry.get_env_constructor("banking_knowledge")
tasks={t.id:t for t in registry.get_tasks_loader("banking_knowledge")()}

def init_args(t):
    ist=getattr(t,"initial_state",None)
    if ist is None: return (None,None,[])
    return (ist.initialization_data, ist.initialization_actions, list(ist.message_history or []))

def diff(g,p,path="",out=None):
    if type(g)!=type(p): out.append(("TYPE",path,repr(g)[:80],repr(p)[:80])); return
    if isinstance(g,dict):
        for k in sorted(set(g)|set(p),key=str):
            if k not in g: out.append(("ONLY-PRED",path+"."+str(k),"",repr(p[k])[:100]))
            elif k not in p: out.append(("ONLY-GOLD",path+"."+str(k),repr(g[k])[:100],""))
            else: diff(g[k],p[k],path+"."+str(k),out)
    elif isinstance(g,list):
        if len(g)!=len(p): out.append(("LEN",path,str(len(g)),str(len(p))))
        for i in range(min(len(g),len(p))): diff(g[i],p[i],"%s[%d]"%(path,i),out)
    else:
        if g!=p: out.append(("DIFF",path,repr(g)[:80],repr(p)[:80]))

REG=".agent_discoverable_tools"
cat=Counter(); rows=[]; regtool=Counter(); mismatch=[]
n=0
for sim in results.simulations:
    t=tasks.get(sim.task_id)
    if t is None: continue
    n+=1
    ri=sim.reward_info
    stored=bool(ri.db_check.db_match) if (ri and ri.db_check) else None
    a,b,c0=init_args(t)
    try:
        gold=env_ctor(retrieval_variant="no_knowledge"); gold.set_state(a,b,list(c0))
        for act in (t.evaluation_criteria.actions or []):
            try: gold.make_tool_call(tool_name=act.name, requestor=act.requestor, **act.arguments)
            except Exception: pass
        pred=env_ctor(retrieval_variant="no_knowledge"); pred.set_state(a,b,list(sim.messages))
        match=gold.tools.get_db_hash()==pred.tools.get_db_hash()
    except Exception as e:
        cat["REPLAY_ERR"]+=1; rows.append((sim.task_id,"REPLAY_ERR",0,0)); continue
    if stored is not None and match!=stored: mismatch.append((sim.task_id,stored,match))
    out=[]; diff(gold.tools.db.model_dump(), pred.tools.db.model_dump(), "", out)
    reg=[d for d in out if d[1].startswith(REG)]
    state=[d for d in out if not d[1].startswith(REG)]
    for d in reg:
        m=re.search(r"'tool_name': '([^']+)'", d[2] or "")
        if m: regtool[m.group(1)]+=1
    if match: k="PASS"
    elif reg and not state: k="REG_ONLY"
    elif state and not reg: k="STATE_ONLY"
    elif not out: k="HASH_ODD"
    else: k="MIXED"
    cat[k]+=1; rows.append((sim.task_id,k,len(reg),len(state)))
print("N=%d"%n)
print("=== validation: replay vs stored db_match ===")
print("   mismatches: %d %s"%(len(mismatch),mismatch[:12]))
print("=== CATEGORIES ===")
for k,v in cat.most_common(): print("   %-12s %3d (%.0f%%)"%(k,v,100.0*v/max(n,1)))
print("=== REG_ONLY (behavior correct, only bookkeeping) ===")
ro=[r for r in rows if r[1]=="REG_ONLY"]
for r in ro: print("   %s reg=%d"%(r[0],r[2]))
print("   TOTAL=%d"%len(ro))
print("=== near-miss MIXED (state<=2) ===")
nm=[r for r in rows if r[1]=="MIXED" and r[3]<=2]
for r in nm: print("   %s reg=%d state=%d"%(r[0],r[2],r[3]))
print("   TOTAL=%d"%len(nm))
print("=== top missed registration tools ===")
for k,v in regtool.most_common(12): print("   %-44s %d"%(k,v))
