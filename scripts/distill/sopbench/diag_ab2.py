#!/usr/bin/env python3
"""A/B with FULL success + cred-state + flip/regression. argv[1]=runA json, argv[2]=runB json."""
import json, hashlib, sys, collections
A,B=sys.argv[1],sys.argv[2]
CANON="/home/woori/scratch/SOPBench/data/bank_tasks.json"
def H(x): return hashlib.md5(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()[:16]
def ident(t): return (t.get("user_goal"),H(t.get("constraints_original")),H(t.get("constraint_parameters")),H(t.get("initial_database")))
canon=json.load(open(CANON)); cl=[]
if isinstance(canon,dict):
    for v in canon.values(): cl+=v if isinstance(v,list) else [v]
else: cl=canon
cmap=collections.defaultdict(list)
for ct in cl:
    if isinstance(ct,dict): cmap[ident(ct)].append(ct.get("user_known") or {})
def cred(idt):
    cks=cmap.get(idt)
    if not cks: return "?"
    return "cred-present" if all("identification" in u for u in cks) else "cred-absent"
def load(p):
    d=json.load(open(p)); m={}
    for e in d:
        ev=e["evaluations"][0]
        if not ev.get("action_should_succeed"): continue
        full=bool(ev.get("success"))
        both=bool(ev.get("dirgraph_satisfied") and ev.get("action_successfully_called"))
        m[ident(e["task"])]=(e["task"]["user_goal"],full,both,ev.get("constraint_not_violated"),ev.get("database_match"))
    return m
mA=load(A); mB=load(B)
print(f"A={A.split('/')[-3]}: full_success={sum(1 for k in mA if mA[k][1])} BOTH={sum(1 for k in mA if mA[k][2])}")
print(f"B={B.split('/')[-3]}: full_success={sum(1 for k in mB if mB[k][1])} BOTH={sum(1 for k in mB if mB[k][2])}")
flips=[]; regr=[]
for k in set(mA)|set(mB):
    a=mA.get(k); b=mB.get(k)
    if not a or not b: continue
    if not a[2] and b[2]: flips.append((b[0],cred(k)))
    if a[2] and not b[2]: regr.append((b[0],cred(k)))
print(f"\nFLIP not->BOTH ({len(flips)}): {dict(collections.Counter(flips))}")
print(f"REGRESSION BOTH->not ({len(regr)}): {dict(collections.Counter(regr))}")
# full-success vs BOTH discrepancy in B (BOTH but not full = cnv/dbm fail)
print("\nB: BOTH-but-not-full-success (cnv or dbm fail):")
for k in mB:
    g,full,both,cnv,dbm=mB[k]
    if both and not full:
        print(f"   {g} [{cred(k)}] cnv={cnv} dbm={dbm}")
