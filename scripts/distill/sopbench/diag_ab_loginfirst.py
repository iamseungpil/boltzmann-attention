#!/usr/bin/env python3
"""A/B: BASE (augment off, loginfirst off) vs FIX1 (augment off, loginfirst on).
Join by augment-INVARIANT identity (goal+constraints_original+constraint_parameters+initial_database)
so it is stable across augment AND loginfirst. Report BOTH totals, flips, regressions, per-goal.
Also tag canonical-cred presence so we can see the cred-present 4 flip."""
import json, hashlib, collections
BASE="/home/woori/scratch/sft_alias_run/eval_t1c_base_noaug/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
FIX="/home/woori/scratch/sft_alias_run/eval_t1c_loginfirst/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
CANON="/home/woori/scratch/SOPBench/data/bank_tasks.json"
def H(x): return hashlib.md5(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()[:16]
def ident(t): return (t.get("user_goal"),H(t.get("constraints_original")),H(t.get("constraint_parameters")),H(t.get("initial_database")))
def both(ev): return bool(ev.get("dirgraph_satisfied") and ev.get("action_successfully_called"))
def acc(ev): return bool(ev.get("action_successfully_called"))

# canonical cred presence by identity
canon=json.load(open(CANON)); cl=[]
if isinstance(canon,dict):
    for v in canon.values(): cl+=v if isinstance(v,list) else [v]
else: cl=canon
cmap=collections.defaultdict(list)
for ct in cl:
    if isinstance(ct,dict): cmap[ident(ct)].append(ct.get("user_known") or {})
def credstate(idt):
    cks=cmap.get(idt)
    if not cks: return "?"
    return "cred-present" if all("identification" in u for u in cks) else "cred-absent"

def load(p):
    d=json.load(open(p)); m={}
    for e in d:
        ev=e["evaluations"][0]
        if ev.get("action_should_succeed"): m[ident(e["task"])]=(e["task"]["user_goal"],both(ev),acc(ev),ev.get("dirgraph_satisfied"),ev.get("constraint_not_violated"))
    return m
B=load(BASE); F=load(FIX)
keys=set(B)|set(F)
bB=sum(1 for k in B if B[k][1]); bF=sum(1 for k in F if F[k][1])
print(f"BASE should_T={len(B)} BOTH={bB}")
print(f"FIX1 should_T={len(F)} BOTH={bF}")
flips=[]; regr=[]
for k in keys:
    b=B.get(k); f=F.get(k)
    if not b or not f: continue
    if not b[1] and f[1]: flips.append((f[0],credstate(k)))
    if b[1] and not f[1]: regr.append((f[0],credstate(k)))
print(f"\nFLIPS not->BOTH ({len(flips)}): {collections.Counter(flips)}")
print(f"REGRESSIONS BOTH->not ({len(regr)}): {collections.Counter(regr)}")
# detail of all non-BOTH in FIX1 by cred state
print("\nFIX1 non-BOTH should_T:")
nb=collections.Counter()
for k in F:
    if not F[k][1]:
        g,bo,ac,dg,cnv=F[k]; nb[(g,credstate(k),"premature" if ac else "deny")]+=1
for kk,n in sorted(nb.items()): print(f"   {n}x {kk}")
