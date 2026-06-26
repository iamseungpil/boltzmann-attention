#!/usr/bin/env python3
"""Reconcile vs bug-report: why 14 cred-absent when bug-report Part B = 6?
Hypothesis: cred-absent != impossible. The 6 are cred-absent AND login-MANDATORY
(no no-login OR branch). The rest are cred-absent but WINNABLE via a no-login path.
For each cred-absent should_T, TEST winnability WITHOUT login/creds on the authoritative
evaluator, trying several no-login trajectories. Classify UNWINNABLE(defect) vs WINNABLE."""
import json, hashlib, sys, copy
sys.path.insert(0, "/home/woori/scratch/SOPBench")
from env.evaluator import evaluator_function_directed_graph
from env.variables import domain_keys, domain_assistant_keys
from env.task import get_default_dep_full

EVAL = "/home/woori/scratch/sft_alias_run/eval_t1c_dggate/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
CANON = "/home/woori/scratch/SOPBench/data/bank_tasks.json"
def H(x): return hashlib.md5(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()[:16]
def ident(t): return (t.get("user_goal"),H(t.get("constraints")),H(t.get("constraint_parameters")),H(t.get("initial_database")))
def sig(t): return hashlib.md5(json.dumps([t.get("user_goal"),t.get("constraints"),t.get("user_known")],sort_keys=True,default=str).encode()).hexdigest()[:12]
def try_eval(x):
    try: return eval(x) if isinstance(x,str) else x
    except Exception: return x
def extract(interaction):
    fc=[]
    for i in range(len(interaction)-1):
        if interaction[i].get("tool_calls", []):
            tcs=[tc for tc in interaction[i]["tool_calls"] if tc["function"]["name"].lower() not in ["n/a","na","none","null"]]
            if tcs: fc.append({"tool_name":interaction[i+1]["tool_name"],"arguments":try_eval(tcs[0]["function"]["arguments"]),"content":try_eval(interaction[i+1]["content"])})
    return fc
PARTA={"cancel_credit_card","pay_bill_with_credit_card"}
LOGIN_FN={"login_user","authenticate_admin_password"}

canon=json.load(open(CANON)); cl=[]
if isinstance(canon,dict):
    for v in canon.values(): cl+=v if isinstance(v,list) else [v]
else: cl=canon
cmap={}
for ct in cl:
    if isinstance(ct,dict): cmap.setdefault(ident(ct),[]).append(ct.get("user_known") or {})

def strict(t):
    di=domain_assistant_keys["bank"].action_innate_dependencies
    ddf=get_default_dep_full("bank","full"); ddf[t["user_goal"]]=t["constraints"]
    return domain_keys["bank_strict"](copy.deepcopy(t["initial_database"]),di,ddf,t["constraint_parameters"])
def run(t, seq, inter):
    dss=strict(t); ideal=[]
    for fn,ar in seq:
        try: r=copy.deepcopy(getattr(dss,fn)(**ar))
        except Exception as ex: r=f"EXC:{ex}"
        ideal.append({"tool_name":fn,"arguments":ar,"content":r})
    res={"final_database":dss.evaluation_get_database()}
    rr=evaluator_function_directed_graph(domain_str="bank",task=t,log_msg_fcall=inter,func_calls=ideal,results=res,default_constraint_option="full")
    return rr.get("constraint_not_violated") and rr.get("dirgraph_satisfied") and rr.get("action_successfully_called") and rr.get("database_match")

d=json.load(open(EVAL))
unwinnable=[]; winnable=[]
print(f"{'goal':<24}{'sig':<13}{'status':<10}{'no-login winnable?':<20}")
print("="*70)
for e in d:
    t=e["task"]; ev=e["evaluations"][0]
    if not ev.get("action_should_succeed") or t["user_goal"] in PARTA: continue
    cks=cmap.get(ident(t))
    if cks is None: continue
    cred_present = all(("identification" in u) for u in cks)
    if cred_present: continue   # only cred-absent / cred-mixed
    both = ev.get("dirgraph_satisfied") and ev.get("action_successfully_called")
    status = "BOTH" if both else ("premature" if ev.get("action_successfully_called") else "DENY")
    goal=t["user_goal"]; uk=t["user_known"]
    fc=extract(e["interactions"][0]["interaction"])
    # non-login calls the model made (getters etc), with their args; plus goal args
    nonlogin={}; goal_args=None
    for c in fc:
        if c["tool_name"]==goal: goal_args=c["arguments"]
        elif c["tool_name"] not in LOGIN_FN: nonlogin[c["tool_name"]]=c["arguments"]
    if goal_args is None: goal_args=uk  # fallback
    inter=e["interactions"][0]["interaction"]
    # try several no-login trajectories
    trajs=[]
    base=[("internal_check_username_exist",{"username":uk.get("username")})]
    # all non-login getters the model used
    getters=[(fn,ar) for fn,ar in nonlogin.items() if fn!="internal_check_username_exist"]
    trajs.append(base+getters+[(goal,goal_args)])           # check + getters + goal (no login)
    trajs.append(base+[(goal,goal_args)])                    # check + goal
    trajs.append([(goal,goal_args)])                          # goal only
    win=any(run(t,seq,inter) for seq in trajs)
    tag = "WINNABLE (no-login)" if win else "UNWINNABLE (login mandatory)"
    (winnable if win else unwinnable).append((goal,sig(t),status))
    print(f"{goal:<24}{sig(t):<13}{status:<10}{tag}")

print(f"\nUNWINNABLE without login/creds (= true Part-B defects) = {len(unwinnable)}")
for g,s,st in unwinnable: print(f"   {g:<22} {s} [{st}]")
print(f"\nWINNABLE via no-login path (cred-absent but NOT defect) = {len(winnable)}")
for g,s,st in winnable: print(f"   {g:<22} {s} [{st}]")
