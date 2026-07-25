import sys, gzip, json, tempfile, os
from pathlib import Path
sys.path.insert(0,"/home/woori/scratch/tau2-bench/src")
from tau2.registry import registry
from tau2.data_model.simulation import Results

GZ="/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results/bank_reg043nt4_20260725.results.json.gz"
raw=json.load(gzip.open(GZ))
tmp=tempfile.NamedTemporaryFile("w",suffix=".json",delete=False)
json.dump(raw,tmp); tmp.close()
results=Results.load(Path(tmp.name))
env_ctor=registry.get_env_constructor("banking_knowledge")
tasks=registry.get_tasks_loader("banking_knowledge")()
task=[t for t in tasks if t.id=="task_043"][0]
istate=task.initial_state
sims=[s for s in results.simulations if s.task_id=="task_043"]
print("n sims:",len(sims))

def diff(g,p,path="",out=None):
    if type(g)!=type(p): out.append("TYPE %s: %r vs %r"%(path,g,p)); return
    if isinstance(g,dict):
        for k in sorted(set(g)|set(p),key=str):
            if k not in g: out.append("ONLY-PRED %s.%s = %r"%(path,k,p[k]))
            elif k not in p: out.append("ONLY-GOLD %s.%s = %r"%(path,k,g[k]))
            else: diff(g[k],p[k],path+"."+str(k),out)
    elif isinstance(g,list):
        if len(g)!=len(p): out.append("LEN %s: gold=%d pred=%d"%(path,len(g),len(p)))
        for i in range(min(len(g),len(p))): diff(g[i],p[i],"%s[%d]"%(path,i),out)
    else:
        if g!=p: out.append("DIFF %s: gold=%r pred=%r"%(path,g,p))

for si,sim in enumerate(sims):
    gold=env_ctor(retrieval_variant="no_knowledge")
    gold.set_state(istate.initialization_data, istate.initialization_actions, list(istate.message_history or []))
    for a in (task.evaluation_criteria.actions or []):
        try: gold.make_tool_call(tool_name=a.name, requestor=a.requestor, **a.arguments)
        except Exception as e: pass
    pred=env_ctor(retrieval_variant="no_knowledge")
    pred.set_state(istate.initialization_data, istate.initialization_actions, list(sim.messages))
    match = gold.tools.get_db_hash()==pred.tools.get_db_hash()
    out=[]
    diff(gold.tools.db.model_dump(), pred.tools.db.model_dump(), "", out)
    print("\n########## sim%d (trial=%s) db_match=%s  ndiff=%d ##########"%(si,getattr(sim,'trial',None),match,len(out)))
    for line in out[:60]:
        print("   ",line[:260])
