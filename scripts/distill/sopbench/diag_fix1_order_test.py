#!/usr/bin/env python3
"""Fix-1 design disambiguation (zero-cost, canonical creds, real evaluator).
For the 4 cred-present premature, test whether the gate can REPAIR late vs must FRONT-LOAD login:
  (a) front-load:  check, login, [admin], getters, goal            (known to pass)
  (b) late-repair: check, getter(no-login), login, getter(again), goal
If (b) passes => gate may drive login + RE-drive getter when about to ACT (late repair OK).
If (b) fails  => evaluator's in-order check dooms the early out-of-order getter => gate must
                 FRONT-LOAD login (drive login before the model's first getter)."""
import json, hashlib, sys, copy
sys.path.insert(0, "/home/woori/scratch/SOPBench")
from env.evaluator import evaluator_function_directed_graph
from env.variables import domain_keys, domain_assistant_keys
from env.task import get_default_dep_full

EVAL="/home/woori/scratch/sft_alias_run/eval_t1c_dggate/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
CRED_PRESENT={"fbaa1c37e3fb","1cb22c9750c5","c6454ec05ef3","79c3a7e8d18e"}
def sig(t): return hashlib.md5(json.dumps([t.get("user_goal"),t.get("constraints"),t.get("user_known")],sort_keys=True,default=str).encode()).hexdigest()[:12]
def try_eval(x):
    try: return eval(x) if isinstance(x,str) else x
    except Exception: return x
def extract(it):
    fc=[]
    for i in range(len(it)-1):
        if it[i].get("tool_calls", []):
            tcs=[tc for tc in it[i]["tool_calls"] if tc["function"]["name"].lower() not in ["n/a","na","none","null"]]
            if tcs: fc.append({"tool_name":it[i+1]["tool_name"],"arguments":try_eval(tcs[0]["function"]["arguments"]),"content":try_eval(it[i+1]["content"])})
    return fc
def strict(t):
    di=domain_assistant_keys["bank"].action_innate_dependencies
    ddf=get_default_dep_full("bank","full"); ddf[t["user_goal"]]=t["constraints"]
    return domain_keys["bank_strict"](copy.deepcopy(t["initial_database"]),di,ddf,t["constraint_parameters"])
def run(t,seq,it):
    dss=strict(t); ideal=[]
    for fn,ar in seq:
        try: r=copy.deepcopy(getattr(dss,fn)(**ar))
        except Exception as ex: r=f"EXC:{ex}"
        ideal.append({"tool_name":fn,"arguments":ar,"content":r})
    res={"final_database":dss.evaluation_get_database()}
    rr=evaluator_function_directed_graph(domain_str="bank",task=t,log_msg_fcall=it,func_calls=ideal,results=res,default_constraint_option="full")
    return (rr.get("constraint_not_violated"),rr.get("dirgraph_satisfied"),rr.get("action_successfully_called"),rr.get("database_match"))

d=json.load(open(EVAL))
for e in d:
    t=e["task"]
    if sig(t) not in CRED_PRESENT: continue
    goal=t["user_goal"]; uk=t["user_known"]; un=uk.get("username")
    it=e["interactions"][0]["interaction"]
    fc=extract(it)
    getters=[]; goal_args=None; has_admin=False
    for c in fc:
        if c["tool_name"]==goal: goal_args=c["arguments"]
        elif c["tool_name"] in ("get_account_balance","get_account_owed_balance","internal_get_credit_score"): getters.append((c["tool_name"],c["arguments"]))
        elif c["tool_name"]=="authenticate_admin_password": has_admin=True
    goal_args=goal_args or uk
    login=("login_user",{"username":un,"identification":uk.get("identification")})
    admin=("authenticate_admin_password",{"username":un,"admin_password":uk.get("admin_password")}) if has_admin else None
    check=("internal_check_username_exist",{"username":un})
    g0=getters[0] if getters else None
    # (a) front-load
    a=[check,login]+([admin] if admin else [])+getters+[(goal,goal_args)]
    # (b) late-repair: out-of-order getter first, then login, then re-getter, then goal
    b=[check]+([g0] if g0 else [])+[login]+([admin] if admin else [])+getters+[(goal,goal_args)]
    ra=run(t,a,it); rb=run(t,b,it)
    oka=all(ra); okb=all(rb)
    print(f"[{goal}] sig={sig(t)}")
    print(f"   (a) front-load : success={oka} {ra}")
    print(f"   (b) late-repair: success={okb} {rb}")
