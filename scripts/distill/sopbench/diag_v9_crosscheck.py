#!/usr/bin/env python3
"""Authoritative cross-check (bug-report method B): for the 9 UNWINNABLE cred-absent
instances, how many of the ~43 released output/bank/*.json model runs PASS them?
Match by augment-INVARIANT identity (goal+constraints+constraint_parameters+initial_database;
user_known excluded so released[unaug] matches our[aug]). 0 passes => real defect.
Also print released user_known cred state to confirm cred-absent in the released (canonical) data."""
import json, hashlib, glob, os, collections

EVAL = "/home/woori/scratch/sft_alias_run/eval_t1c_dggate/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
RELEASED = "/home/woori/scratch/SOPBench/output/bank/ast_*.json"
def H(x): return hashlib.md5(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()[:16]
def ident(t): return (t.get("user_goal"),H(t.get("constraints")),H(t.get("constraint_parameters")),H(t.get("initial_database")))
def sig(t): return hashlib.md5(json.dumps([t.get("user_goal"),t.get("constraints"),t.get("user_known")],sort_keys=True,default=str).encode()).hexdigest()[:12]

# the 9 UNWINNABLE sigs from v8
TARGET_SIGS = {"1e6a5bf39915","afcfdb6a5394","6058f1540820","92f35479191d","81ba61427f77",
               "48e4601e9c66","3b48ea090e93","fc3ed175b568","047ddc88900c"}

d=json.load(open(EVAL))
targets={}  # ident -> (goal, sig)
for e in d:
    t=e["task"]
    if sig(t) in TARGET_SIGS:
        targets[ident(t)]=(t["user_goal"], sig(t))

def get_success(ev):
    if isinstance(ev,dict): return ev.get("success")
    return None

files=sorted(glob.glob(RELEASED))
print(f"scanning {len(files)} released files\n")
# per ident: count passes / total appearances, and record released user_known cred state
passcount=collections.defaultdict(lambda:[0,0])  # ident -> [pass, total]
released_uk={}
for fp in files:
    try: data=json.load(open(fp))
    except Exception: continue
    if not isinstance(data,list): continue
    for e in data:
        if not isinstance(e,dict) or "task" not in e: continue
        t=e["task"]; idt=ident(t)
        if idt not in targets: continue
        evs=e.get("evaluations") or []
        ev=evs[0] if evs else {}
        s=get_success(ev)
        if s is None: continue
        passcount[idt][1]+=1
        if s: passcount[idt][0]+=1
        if idt not in released_uk:
            uk=t.get("user_known") or {}
            released_uk[idt]=sorted(uk.keys())

print(f"{'goal':<22}{'sig':<13}{'passes/total(released)':<24}{'released user_known keys'}")
print("="*100)
for idt,(goal,s) in sorted(targets.items(), key=lambda kv: kv[1][0]):
    p,tot=passcount.get(idt,[0,0])
    uk=released_uk.get(idt,"<not found in released>")
    has_id = ("identification" in uk) if isinstance(uk,list) else "?"
    print(f"{goal:<22}{s:<13}{str(p)+'/'+str(tot):<24}id_in_uk={has_id}  {uk}")

n_zero=sum(1 for idt in targets if passcount.get(idt,[0,0])[0]==0)
print(f"\n{n_zero}/{len(targets)} UNWINNABLE instances passed by 0 released models (=> confirmed defects).")
print("If pay_loan/extra-transfer get 0 passes => bug-report Part-B under-counted (real count > 6).")
