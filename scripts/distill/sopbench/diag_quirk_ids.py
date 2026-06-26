#!/usr/bin/env python3
"""Identity of the should_T quirk-pass tasks in S1 (login called but ALL failed, yet success).
Compare to cred-absent set. argv1 = eval json."""
import json, hashlib, sys, collections
p=sys.argv[1] if len(sys.argv)>1 else "/home/woori/scratch/sft_alias_run/eval_t1c_stopsuccess/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
CANON="/home/woori/scratch/SOPBench/data/bank_tasks.json"
def H(x): return hashlib.md5(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()[:16]
def ident(t): return (t.get("user_goal"),H(t.get("constraints_original")),H(t.get("constraint_parameters")),H(t.get("initial_database")))
def truthy(c):
    if c is True: return True
    if isinstance(c,(list,tuple)) and len(c): return c[0] is True
    if isinstance(c,str): return c.strip().startswith(("True","(True","[True"))
    return False
canon=json.load(open(CANON)); cl=[]
for v in canon.values(): cl+= v if isinstance(v,list) else [v]
cmap=collections.defaultdict(list)
for ct in cl:
    if isinstance(ct,dict): cmap[ident({**ct,"user_goal":ct.get("user_goal")})].append(ct)
def credabsent(t):
    # match canonical by augment-invariant identity; cred-absent if no canonical instance has identification
    cks=cmap.get(ident(t))
    if not cks: return "?"
    return all("identification" not in (c.get("user_known") or {}) for c in cks)
d=json.load(open(p))
q=[]
for e in d:
    ev=e["evaluations"][0]
    if not (ev.get("action_should_succeed") and ev.get("success")): continue
    lc=[]; ac=[]
    for il in e.get("interactions") or []:
        conv=il.get("interaction") if isinstance(il,dict) else None
        if not conv: continue
        for m in conv:
            if isinstance(m,dict):
                if m.get("tool_name")=="login_user": lc.append(truthy(m.get("content")))
                if m.get("tool_name")=="authenticate_admin_password": ac.append(truthy(m.get("content")))
    if (lc and not any(lc)) or (ac and not any(ac)):
        q.append((e["task"]["user_goal"], "login-fail" if (lc and not any(lc)) else "", "admin-fail" if (ac and not any(ac)) else "", credabsent(e["task"])))
print(f"should_T quirk-pass tasks (n={len(q)}):")
for g,lf,af,ca in q:
    print(f"  {g:<24} {lf} {af}  cred-absent={ca}")
print("\nby goal:", dict(collections.Counter(x[0] for x in q)))
print("cred-absent count:", sum(1 for x in q if x[3] is True), " / cred-present:", sum(1 for x in q if x[3] is False))
