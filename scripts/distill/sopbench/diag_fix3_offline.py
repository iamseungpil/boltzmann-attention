#!/usr/bin/env python3
"""Fix-3 B-3 zero-cost pre-validation (GPU 전 필수).
For each should_T that is BOTH(dg&acc) but NOT full success in the L1C(logincall) run, TRUNCATE the
call sequence to the FIRST SUCCESSFUL goal-call (inclusive), recompute final_database by strict-replay
of that truncated prefix, and re-score with the authoritative evaluator. flip = becomes full success.
This is reliable by PREFIX-IDENTITY: STOPSUCCESS only changes behavior AFTER the first success, so the
prefix up to (and incl) the first successful goal-call is identical S0=S1; truncating is exactly S1's
expected trajectory (prefix-preserving, NOT injection -> distinct from the retracted forced-ACT offline).
Output: per-task flip + total flip = Fix-3 upper bound. flip 0 => looping not the cause => DISCARD."""
import json, hashlib, sys, copy
sys.path.insert(0, "/home/woori/scratch/SOPBench")
from env.evaluator import evaluator_function_directed_graph
from env.variables import domain_keys, domain_assistant_keys
from env.task import get_default_dep_full

EVAL = "/home/woori/scratch/sft_alias_run/eval_t1c_logincall/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
def try_eval(x):
    try: return eval(x) if isinstance(x, str) else x
    except Exception: return x
def sig(t): return hashlib.md5(json.dumps([t.get("user_goal"),t.get("constraints"),t.get("user_known")],sort_keys=True,default=str).encode()).hexdigest()[:12]
def extract(it):
    fc=[]
    for i in range(len(it)-1):
        if it[i].get("tool_calls", []):
            tcs=[tc for tc in it[i]["tool_calls"] if tc["function"]["name"].lower() not in ["n/a","na","none","null"]]
            if tcs: fc.append({"tool_name":it[i+1]["tool_name"],"arguments":try_eval(tcs[0]["function"]["arguments"]),"content":try_eval(it[i+1]["content"])})
    return fc
def succ_truthy(c):
    return (c is True) or (isinstance(c,(list,tuple)) and len(c) and c[0] is True) or (isinstance(c,str) and c.strip().startswith(("True","(True","[True")))

d=json.load(open(EVAL))
flips=0; tot=0; rows=[]
for e in d:
    t=e["task"]; ev=e["evaluations"][0]; goal=t["user_goal"]
    if not ev.get("action_should_succeed"): continue
    both=ev.get("dirgraph_satisfied") and ev.get("action_successfully_called")
    full=ev.get("success")
    if not (both and not full): continue
    tot+=1
    it=e["interactions"][0]["interaction"]
    fc=extract(it)
    # truncate to first SUCCESSFUL goal-call (inclusive)
    trunc=[]; cut=False
    for c in fc:
        trunc.append(c)
        if c["tool_name"]==goal and succ_truthy(c["content"]) and "Error" not in str(c["content"]):
            cut=True; break
    if not cut:
        rows.append((goal,sig(t),"NO-SUCCESS-GOAL-CALL","")); continue
    # strict-replay the truncated prefix to get the counterfactual final_database
    di=domain_assistant_keys["bank"].action_innate_dependencies
    ddf=get_default_dep_full("bank","full"); ddf[goal]=t["constraints"]
    dss=domain_keys["bank_strict"](copy.deepcopy(t["initial_database"]),di,ddf,t["constraint_parameters"])
    for c in trunc:
        try: getattr(dss,c["tool_name"])(**c["arguments"])
        except Exception: pass
    res={"final_database": dss.evaluation_get_database()}
    r=evaluator_function_directed_graph(domain_str="bank",task=t,log_msg_fcall=it,func_calls=trunc,results=res,default_constraint_option="full")
    f=bool(r.get("success"))
    if f: flips+=1
    rows.append((goal,sig(t),"FLIP->full" if f else "no",
                 f"(cnv={r.get('constraint_not_violated')},dbm={r.get('database_match')},ntce={r.get('no_tool_call_error')},dg={r.get('dirgraph_satisfied')},acc={r.get('action_successfully_called')}) ncalls {len(fc)}->{len(trunc)}"))

for g,s,tag,detail in rows:
    print(f"  {g:<24} {s} {tag:<22} {detail}")
print(f"\nFix-3 B-3: {flips}/{tot} BOTH-but-not-full FLIP to full success when truncated to first successful goal-call.")
print("판정: flip 충분(>=8/12) -> 구현·A/B 진행. flip 0 -> looping 원인 아님 -> 폐기.")
