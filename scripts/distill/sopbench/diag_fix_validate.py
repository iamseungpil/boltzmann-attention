#!/usr/bin/env python3
"""Validate the prescribed fix BEFORE any GPU run (zero-cost, real evaluator).
Hypothesis: residual premature fail because login_user is missing/mis-ordered before
the value-getters and authenticate_admin_password. Fix = establish login_user FIRST.
Construct an ideal-ordered single-goal trace:
   internal_check_username_exist, login_user(creds), authenticate_admin_password(if used),
   <value getters in original args>, <goal once>
and re-run the authoritative evaluator. If cnv & dg & acc => True, fix is confirmed."""
import json, hashlib, sys, copy
sys.path.insert(0, "/home/woori/scratch/SOPBench")
from env.evaluator import evaluator_function_directed_graph

EVAL = "/home/woori/scratch/sft_alias_run/eval_t1c_dggate/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"

def try_eval(x):
    try: return eval(x) if isinstance(x, str) else x
    except Exception: return x
def sig(t):
    return hashlib.md5(json.dumps([t.get("user_goal"),t.get("constraints"),t.get("user_known")],
        sort_keys=True, default=str).encode()).hexdigest()[:12]
def extract(interaction):
    fc=[]
    for i in range(len(interaction)-1):
        if interaction[i].get("tool_calls", []):
            tcs=[tc for tc in interaction[i]["tool_calls"] if tc["function"]["name"].lower() not in ["n/a","na","none","null"]]
            if tcs:
                fc.append({"tool_name": interaction[i+1]["tool_name"],
                           "arguments": try_eval(tcs[0]["function"]["arguments"]),
                           "content": try_eval(interaction[i+1]["content"])})
    return fc

# order priority: login first, then admin auth, then getters, then everything else, goal last
PRIORITY = {"internal_check_username_exist":0, "login_user":1, "authenticate_admin_password":2,
            "get_account_balance":3, "get_account_owed_balance":3, "internal_get_credit_score":3,
            "internal_get_database":3}

d=json.load(open(EVAL))
print(f"{'goal':<26}{'sig':<14}orig(cnv,dg,acc)  ideal(cnv,dg,acc)  FIXED?")
print("="*84)
nfix=0; tot=0
for e in d:
    t=e["task"]; ev=e["evaluations"][0]
    if not (ev.get("action_should_succeed") and ev.get("action_successfully_called") and not ev.get("dirgraph_satisfied")):
        continue
    tot+=1
    s=sig(t); goal=t["user_goal"]; uk=t["user_known"]
    fc=extract(e["interactions"][0]["interaction"])
    # collect unique non-goal calls (keep last args seen), plus one goal call
    uniq={}
    goal_args=None
    for c in fc:
        if c["tool_name"]==goal:
            goal_args=c["arguments"]
        else:
            uniq[c["tool_name"]]=c["arguments"]
    # ensure login_user present if any login-requiring fn used (getter/admin) and creds available
    needs_login = any(fn in uniq for fn in ["get_account_balance","get_account_owed_balance",
                      "internal_get_credit_score","authenticate_admin_password"])
    if needs_login and "login_user" not in uniq and uk.get("identification") is not None:
        uniq["login_user"]={"username":uk.get("username"),"identification":uk.get("identification")}
    # build ordered list
    ordered=sorted(uniq.items(), key=lambda kv: PRIORITY.get(kv[0],5))
    ideal=[]
    # we need tool 'content' for cnv: recompute by NOT supplying agent content (use placeholder).
    # cnv compares agent content vs strict gt; to fairly test we set agent content = strict gt
    # so cnv reflects ONLY ordering/structure, not the agent's earlier wrong responses.
    import importlib
    # Set up strict system to get correct gt content for each call in order
    from env.variables import domain_keys, domain_assistant_keys
    from env.task import get_default_dep_full
    dep_innate=domain_assistant_keys["bank"].action_innate_dependencies
    ddf=get_default_dep_full("bank","full"); ddf[goal]=t["constraints"]
    dss=domain_keys["bank_strict"](copy.deepcopy(t["initial_database"]),dep_innate,ddf,t["constraint_parameters"])
    def call(fn,ar):
        return copy.deepcopy(getattr(dss,fn)(**ar))
    for fn,ar in ordered:
        r=call(fn,ar)
        ideal.append({"tool_name":fn,"arguments":ar,"content":r})
    # goal last
    gr=call(goal,goal_args or {})
    ideal.append({"tool_name":goal,"arguments":goal_args or {},"content":gr})
    res={"final_database": dss.evaluation_get_database()}
    r=evaluator_function_directed_graph(domain_str="bank",task=t,log_msg_fcall=e["interactions"][0]["interaction"],
        func_calls=ideal,results=res,default_constraint_option="full")
    o=(ev.get("constraint_not_violated"),ev.get("dirgraph_satisfied"),ev.get("action_successfully_called"))
    n=(r.get("constraint_not_violated"),r.get("dirgraph_satisfied"),r.get("action_successfully_called"))
    fixed = all(n) and r.get("database_match")
    if fixed: nfix+=1
    order_str="+".join(fn for fn,_ in ordered)
    print(f"{goal:<26}{s:<14}{str(o):<18}{str(n):<19}{'YES dbm='+str(r.get('database_match')) if fixed else 'no  cnv/dg/dbm='+str((n[0],n[1],r.get('database_match')))}")
    print(f"      order: {order_str}+{goal}")
print(f"\n{nfix}/{tot} premature FIXED by login-first ordering (cnv&dg&acc&dbm all True).")
