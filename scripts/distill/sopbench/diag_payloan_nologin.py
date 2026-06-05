#!/usr/bin/env python3
"""Decisive test: can pay_loan 92f3/81ba pass WITHOUT login (no creds), via a no-login OR
branch (bug-report claims pay_loan 66/67 do)? Try candidate no-login trajectories on the
authoritative evaluator using ONLY canonical (cred-absent) user_known + model args.
Also dump pay_loan constraint processes / restr getter login-dependence."""
import json, hashlib, sys, copy, itertools
sys.path.insert(0, "/home/woori/scratch/SOPBench")
from env.evaluator import evaluator_function_directed_graph
from env.variables import domain_keys, domain_assistant_keys
from env.task import get_default_dep_full

EVAL = "/home/woori/scratch/sft_alias_run/eval_t1c_dggate/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
def sig(t): return hashlib.md5(json.dumps([t.get("user_goal"),t.get("constraints"),t.get("user_known")],
        sort_keys=True, default=str).encode()).hexdigest()[:12]
TARGETS={"92f35479191d","81ba61427f77"}
d=json.load(open(EVAL))

def strict(t):
    dep_innate=domain_assistant_keys["bank"].action_innate_dependencies
    ddf=get_default_dep_full("bank","full"); ddf[t["user_goal"]]=t["constraints"]
    return domain_keys["bank_strict"](copy.deepcopy(t["initial_database"]),dep_innate,ddf,t["constraint_parameters"])

for e in d:
    t=e["task"]
    if sig(t) not in TARGETS: continue
    goal=t["user_goal"]; uk=t["user_known"]
    print(f"\n===== pay_loan sig={sig(t)} constraints={json.dumps(t['constraints'])}")
    print(f"  user_known(augmented)={uk}")
    # candidate no-login trajectories (no login_user, no creds needed)
    un=uk.get("username"); amt=uk.get("pay_owed_amount_request")
    cands = {
      "check+owed+goal": [("internal_check_username_exist",{"username":un}),("get_account_owed_balance",{"username":un}),(goal,{"username":un,"pay_owed_amount_request":amt})],
      "check+bal+goal": [("internal_check_username_exist",{"username":un}),("get_account_balance",{"username":un}),(goal,{"username":un,"pay_owed_amount_request":amt})],
      "check+bal+owed+goal": [("internal_check_username_exist",{"username":un}),("get_account_balance",{"username":un}),("get_account_owed_balance",{"username":un}),(goal,{"username":un,"pay_owed_amount_request":amt})],
      "check+goal": [("internal_check_username_exist",{"username":un}),(goal,{"username":un,"pay_owed_amount_request":amt})],
    }
    for name,seq in cands.items():
        dss=strict(t)
        ideal=[]
        for fn,ar in seq:
            try: r=copy.deepcopy(getattr(dss,fn)(**ar))
            except Exception as ex: r=f"EXC:{ex}"
            ideal.append({"tool_name":fn,"arguments":ar,"content":r})
        res={"final_database":dss.evaluation_get_database()}
        rr=evaluator_function_directed_graph(domain_str="bank",task=t,log_msg_fcall=e["interactions"][0]["interaction"],
            func_calls=ideal,results=res,default_constraint_option="full")
        ok=rr.get("constraint_not_violated") and rr.get("dirgraph_satisfied") and rr.get("action_successfully_called") and rr.get("database_match")
        gtv=[c["content"] for c in ideal]
        print(f"  [{name}] success={ok}  (cnv,dg,acc,dbm)={(rr.get('constraint_not_violated'),rr.get('dirgraph_satisfied'),rr.get('action_successfully_called'),rr.get('database_match'))}  strict_returns={gtv}")
