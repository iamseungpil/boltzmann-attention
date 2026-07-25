import sys, gzip, json, tempfile
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
stored_pass=[]; replay_pass=[]; errs={}
for sim in results.simulations:
    ri=sim.reward_info
    sp = bool(ri.db_check.db_match) if (ri and ri.db_check) else None
    if sp: stored_pass.append(sim.task_id)
    t=tasks.get(sim.task_id)
    if t is None: errs[sim.task_id]="NO_TASK"; continue
    ist=t.initial_state
    try:
        gold=env_ctor(retrieval_variant="no_knowledge")
        gold.set_state(ist.initialization_data, ist.initialization_actions, list(ist.message_history or []))
        for a in (t.evaluation_criteria.actions or []):
            try: gold.make_tool_call(tool_name=a.name, requestor=a.requestor, **a.arguments)
            except Exception as e: pass
        pred=env_ctor(retrieval_variant="no_knowledge")
        pred.set_state(ist.initialization_data, ist.initialization_actions, list(sim.messages))
        if gold.tools.get_db_hash()==pred.tools.get_db_hash(): replay_pass.append(sim.task_id)
    except Exception as e:
        errs[sim.task_id]=type(e).__name__+":"+str(e)[:90]
print("stored db_match PASS: %d  -> %s"%(len(stored_pass),sorted(stored_pass)))
print("replay        PASS: %d  -> %s"%(len(replay_pass),sorted(replay_pass)))
print("stored-but-not-replay:",sorted(set(stored_pass)-set(replay_pass)))
print("replay-but-not-stored:",sorted(set(replay_pass)-set(stored_pass)))
print("\nREPLAY ERRORS: %d"%len(errs))
c=Counter(v.split(":")[0] for v in errs.values())
for k,v in c.most_common(): print("   %-30s %d"%(k,v))
for k,v in list(errs.items())[:6]: print("   e.g.",k,v[:120])
