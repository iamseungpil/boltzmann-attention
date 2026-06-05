#!/usr/bin/env python3
"""Is Fix-2 (pay_loan no-login routing) even feasible? Check HOW the released models that PASSED
pay_loan 92f3/81ba (augment-invariant identity) did it: which mode, did they call login_user
(success?) or internal_get_database (the react leak)? If the only passes are react+internal_get_database
or login-with-creds, then NO legitimate fc no-login path exists => pay_loan x2 are defect-class for
fc agents and Fix-2 is infeasible."""
import json, hashlib, glob
def H(x): return hashlib.md5(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()[:16]
def ident(t): return (t.get("user_goal"),H(t.get("constraints_original")),H(t.get("constraint_parameters")),H(t.get("initial_database")))

# recover the two pay_loan identities from OUR dggate eval (sigs 92f3,81ba were content-sigs there)
import collections
EVAL="/home/woori/scratch/sft_alias_run/eval_t1c_dggate/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
def sig(t): return hashlib.md5(json.dumps([t.get("user_goal"),t.get("constraints"),t.get("user_known")],sort_keys=True,default=str).encode()).hexdigest()[:12]
d=json.load(open(EVAL))
targets={}
for e in d:
    if sig(e["task"]) in ("92f35479191d","81ba61427f77"):
        targets[ident(e["task"])]=sig(e["task"])
print("pay_loan target identities:", list(targets.values()))

for fp in sorted(glob.glob("/home/woori/scratch/SOPBench/output/bank/ast_*.json")):
    try: dd=json.load(open(fp))
    except: continue
    if not isinstance(dd,list): continue
    model=fp.split("/")[-1].replace("ast_","")[:50]
    for e in dd:
        if not isinstance(e,dict) or "task" not in e: continue
        idt=ident(e["task"])
        if idt not in targets: continue
        for il in (e.get("interactions") or []):
            ev=(e.get("evaluations") or [{}])
            conv=il.get("interaction") if isinstance(il,dict) else None
            if not conv: continue
            calls=[]
            for m in conv:
                if isinstance(m,dict):
                    for tc in (m.get("tool_calls") or []):
                        calls.append((tc.get("function") or {}).get("name") if tc.get("function") else tc.get("tool_name"))
        # success?
        evl=e.get("evaluations") or []
        succ=evl[0].get("success") if evl else None
        if succ:
            usedidb="internal_get_database" in calls
            usedlogin="login_user" in calls
            print(f"  PASS {targets[idt]} model={model} idb={usedidb} login={usedlogin} calls={calls[:8]}")
