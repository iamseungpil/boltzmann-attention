#!/usr/bin/env python3
"""Full honest reconciliation over all 48 should_T, per the bug-report divider:
classify by (canonical credential presence) x (current status BOTH/premature/deny).
- cred-absent + BOTH   => passes ONLY because AUGMENT_CRED supplied withheld creds (defect-pass).
- cred-absent + resid   => PartB-class defect (unwinnable without augment).
- cred-present + premature => GENUINE login-ordering residual (model-fixable, no augment needed).
- cred-present + BOTH   => genuine pass.
Canonical creds recovered by augment-invariant identity match to bank_tasks.json."""
import json, hashlib, sys, collections
sys.path.insert(0, "/home/woori/scratch/SOPBench")

EVAL = "/home/woori/scratch/sft_alias_run/eval_t1c_dggate/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
CANON = "/home/woori/scratch/SOPBench/data/bank_tasks.json"
def H(x): return hashlib.md5(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()[:16]
def ident(t): return (t.get("user_goal"),H(t.get("constraints")),H(t.get("constraint_parameters")),H(t.get("initial_database")))
PARTA={"cancel_credit_card","pay_bill_with_credit_card"}

canon=json.load(open(CANON))
cl=[]
if isinstance(canon,dict):
    for v in canon.values(): cl+=v if isinstance(v,list) else [v]
else: cl=canon
cmap={}
for ct in cl:
    if isinstance(ct,dict): cmap.setdefault(ident(ct),[]).append(ct.get("user_known") or {})

d=json.load(open(EVAL))
rows=[]
for e in d:
    t=e["task"]; ev=e["evaluations"][0]
    if not ev.get("action_should_succeed"): continue
    goal=t["user_goal"]
    both = ev.get("dirgraph_satisfied") and ev.get("action_successfully_called")
    if both: status="BOTH"
    elif ev.get("action_successfully_called"): status="premature"
    else: status="DENY"
    cks=cmap.get(ident(t))
    if goal in PARTA:
        credclass="PARTA"
    elif cks is None:
        credclass="UNMATCHED"
    else:
        # canonical cred present if ANY matched canonical instance has identification (worst-case: take min)
        has_ident = all(("identification" in u) for u in cks)
        any_ident = any(("identification" in u) for u in cks)
        credclass = "cred-present" if all(("identification" in u) for u in cks) else ("cred-mixed" if any_ident else "cred-absent")
    rows.append((goal,status,credclass))

print(f"{'goal':<26}{'status':<11}{'cred(canonical)':<14}")
print("="*52)
for goal,status,cc in sorted(rows):
    print(f"{goal:<26}{status:<11}{cc:<14}")

print("\n--- summary ---")
c=collections.Counter((r[1],r[2]) for r in rows)
for (status,cc),n in sorted(c.items()):
    print(f"  {status:<10} {cc:<14} : {n}")
tot=len(rows)
both=sum(1 for r in rows if r[1]=="BOTH")
print(f"\ntotal should_T={tot}  BOTH={both}")
# honest buckets
parta=sum(1 for r in rows if r[2]=="PARTA")
cred_absent_resid=sum(1 for r in rows if r[2] in("cred-absent","cred-mixed") and r[1]!="BOTH")
cred_absent_both=sum(1 for r in rows if r[2] in("cred-absent","cred-mixed") and r[1]=="BOTH")
genuine_resid=sum(1 for r in rows if r[2]=="cred-present" and r[1]=="premature")
genuine_both=sum(1 for r in rows if r[2]=="cred-present" and r[1]=="BOTH")
print(f"  PARTA defect (DENY)              = {parta}")
print(f"  cred-absent defect, NOT passing  = {cred_absent_resid}")
print(f"  cred-absent, BOTH (augment-pass) = {cred_absent_both}  <-- pass ONLY via AUGMENT_CRED")
print(f"  cred-present GENUINE residual    = {genuine_resid}  <-- real model-fixable (login-ordering)")
print(f"  cred-present BOTH (true pass)    = {genuine_both}")
