#!/usr/bin/env python3
"""Re-analysis per bug-report Part B: the divider is whether login credentials are
present in the CANONICAL (unaugmented) user_known.
- Match each premature eval-task to canonical bank_tasks.json by augment-INVARIANT
  identity = (user_goal, constraints, constraint_parameters, initial_database) [user_known
  excluded because AUGMENT_CRED mutates it].
- Report canonical user_known cred keys (identification / admin_password present?).
- Re-run the login-first fix using ONLY canonical creds (no augment). A task with no
  canonical creds AND a mandatory-login dirgraph (no no-login OR branch) cannot be fixed
  => confirmed Part-B defect (honest-34 holds). A task WITH canonical creds that flips
  => genuine credentialed login-ordering residual (the real fixable set)."""
import json, hashlib, sys, copy
sys.path.insert(0, "/home/woori/scratch/SOPBench")
from env.evaluator import evaluator_function_directed_graph
from env.variables import domain_keys, domain_assistant_keys
from env.task import get_default_dep_full

EVAL = "/home/woori/scratch/sft_alias_run/eval_t1c_dggate/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
CANON = "/home/woori/scratch/SOPBench/data/bank_tasks.json"

def H(x): return hashlib.md5(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]
def sig(t): return hashlib.md5(json.dumps([t.get("user_goal"),t.get("constraints"),t.get("user_known")],
        sort_keys=True, default=str).encode()).hexdigest()[:12]
def ident(t):  # augment-invariant identity (excludes user_known)
    return (t.get("user_goal"), H(t.get("constraints")), H(t.get("constraint_parameters")), H(t.get("initial_database")))
def try_eval(x):
    try: return eval(x) if isinstance(x,str) else x
    except Exception: return x
def extract(interaction):
    fc=[]
    for i in range(len(interaction)-1):
        if interaction[i].get("tool_calls", []):
            tcs=[tc for tc in interaction[i]["tool_calls"] if tc["function"]["name"].lower() not in ["n/a","na","none","null"]]
            if tcs: fc.append({"tool_name":interaction[i+1]["tool_name"],
                               "arguments":try_eval(tcs[0]["function"]["arguments"]),
                               "content":try_eval(interaction[i+1]["content"])})
    return fc

# canonical map: identity -> list of canonical user_known
canon = json.load(open(CANON))
canon_list = []
if isinstance(canon, dict):
    for v in canon.values():
        canon_list += v if isinstance(v, list) else [v]
else:
    canon_list = canon
cmap = {}
for ct in canon_list:
    if isinstance(ct, dict):
        cmap.setdefault(ident(ct), []).append(ct.get("user_known"))

PRIORITY = {"internal_check_username_exist":0,"login_user":1,"authenticate_admin_password":2,
            "get_account_balance":3,"get_account_owed_balance":3,"internal_get_credit_score":3,"internal_get_database":3}

d = json.load(open(EVAL))
genuine=[]; partb=[]; unmatched=[]
for e in d:
    t=e["task"]; ev=e["evaluations"][0]
    if not (ev.get("action_should_succeed") and ev.get("action_successfully_called") and not ev.get("dirgraph_satisfied")):
        continue
    s=sig(t); goal=t["user_goal"]
    cks = cmap.get(ident(t))
    # canonical user_known: pick the matching one; if multiple, prefer the cred-MINIMAL (the true canonical for this instance is ambiguous, so report all cred-states)
    if not cks:
        print(f"[{goal}] sig={s}  CANON UNMATCHED on identity"); unmatched.append(s); continue
    cred_states = []
    for u in cks:
        u = u or {}
        cred_states.append({"identification": "identification" in u, "admin_password": "admin_password" in u})
    any_no_ident = any(not c["identification"] for c in cred_states)
    all_have_ident = all(c["identification"] for c in cred_states)

    # build login-first fix using CANONICAL creds only. Use the canonical user_known that has the
    # FEWEST creds (worst case) to test whether the task is winnable WITHOUT augment.
    # pick canonical uk with fewest cred keys
    cuk = min(cks, key=lambda u: (("identification" in (u or {})) + ("admin_password" in (u or {}))))
    cuk = cuk or {}
    fc = extract(e["interactions"][0]["interaction"])
    uniq={}; goal_args=None
    for c in fc:
        if c["tool_name"]==goal: goal_args=c["arguments"]
        else: uniq[c["tool_name"]]=c["arguments"]
    needs_login = any(fn in uniq for fn in ["get_account_balance","get_account_owed_balance","internal_get_credit_score","authenticate_admin_password"])
    # can we login with CANONICAL creds?
    can_login = cuk.get("identification") is not None
    if needs_login and "login_user" not in uniq:
        if can_login:
            uniq["login_user"]={"username":cuk.get("username"),"identification":cuk.get("identification")}
        # else: cannot add login (no canonical creds) -> leave as is, fix will fail
    # fix admin_password arg from canonical if present
    if "authenticate_admin_password" in uniq and cuk.get("admin_password") is not None:
        uniq["authenticate_admin_password"]={"username":cuk.get("username"),"admin_password":cuk.get("admin_password")}
    ordered=sorted(uniq.items(), key=lambda kv: PRIORITY.get(kv[0],5))
    # strict system for gt content
    dep_innate=domain_assistant_keys["bank"].action_innate_dependencies
    ddf=get_default_dep_full("bank","full"); ddf[goal]=t["constraints"]
    dss=domain_keys["bank_strict"](copy.deepcopy(t["initial_database"]),dep_innate,ddf,t["constraint_parameters"])
    def call(fn,ar):
        try: return copy.deepcopy(getattr(dss,fn)(**ar))
        except Exception as ex: return f"EXC:{ex}"
    ideal=[]
    for fn,ar in ordered: ideal.append({"tool_name":fn,"arguments":ar,"content":call(fn,ar)})
    ideal.append({"tool_name":goal,"arguments":goal_args or {},"content":call(goal,goal_args or {})})
    res={"final_database":dss.evaluation_get_database()}
    r=evaluator_function_directed_graph(domain_str="bank",task=t,log_msg_fcall=e["interactions"][0]["interaction"],
        func_calls=ideal,results=res,default_constraint_option="full")
    fixed = r.get("constraint_not_violated") and r.get("dirgraph_satisfied") and r.get("action_successfully_called") and r.get("database_match")
    tag = "GENUINE(canonical-cred fixable)" if fixed else "PARTB-class(no canonical cred / unwinnable)"
    (genuine if fixed else partb).append((goal,s))
    print(f"[{goal}] sig={s}  canon_cred_states={cred_states}  can_login_canonical={can_login}  -> {tag}  (cnv,dg,acc,dbm)={(r.get('constraint_not_violated'),r.get('dirgraph_satisfied'),r.get('action_successfully_called'),r.get('database_match'))}")

print(f"\nGENUINE (canonical-cred login-first fixable) = {len(genuine)}: {genuine}")
print(f"PARTB-class (no canonical cred, unwinnable w/o augment) = {len(partb)}: {partb}")
print(f"UNMATCHED = {len(unmatched)}")
