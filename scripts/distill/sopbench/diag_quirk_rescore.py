#!/usr/bin/env python3
"""Quirk EVIDENCE (corrected v2). v1 was WRONG: it treated any login_user NODE in the dirgraph as
'login required', but login_user is usually an OR-branch (login_user OR internal_get_database OR
internal_check_username_exist) -> a task can pass via the non-login branch without logging in. v1
over-penalized every model uniformly (OR-optional login mismarked as quirk-fail).

Correct quirk-exploit signature = success=True AND login_user was CALLED >=1 time AND NO login_user
call returned True (model TRIED to log in, FAILED, yet passed) -> the failed login was counted by
dfscheck (call-order, not auth-success). Same for authenticate_admin_password. This is NARROW, direct
evidence. We also report 'login never called' (OR-branch / no-auth path = NOT a quirk) separately."""
import json, glob

def truthy(c):
    if c is True: return True
    if isinstance(c,(list,tuple)) and len(c): return c[0] is True
    if isinstance(c,str): return c.strip().startswith(("True","(True","[True"))
    return False

def analyze(path):
    try: d=json.load(open(path))
    except Exception: return None
    if not isinstance(d,list): return None
    n=succ=quirk=clean_login=no_login_call=0
    for e in d:
        if not isinstance(e,dict): continue
        evs=e.get("evaluations") or []
        if not evs: continue
        ev=evs[0]
        n+=1
        if not ev.get("success"): continue
        succ+=1
        # collect login_user / admin call return values from trace
        login_calls=[]; admin_calls=[]
        for il in (e.get("interactions") or []):
            conv=il.get("interaction") if isinstance(il,dict) else None
            if not conv: continue
            for m in conv:
                if not isinstance(m,dict): continue
                tn=m.get("tool_name")
                if tn=="login_user": login_calls.append(truthy(m.get("content")))
                if tn=="authenticate_admin_password": admin_calls.append(truthy(m.get("content")))
        # quirk: an auth tool was CALLED but NONE succeeded, yet task passed
        login_quirk = login_calls and not any(login_calls)
        admin_quirk = admin_calls and not any(admin_calls)
        if login_quirk or admin_quirk: quirk+=1
        elif (login_calls and any(login_calls)) or (admin_calls and any(admin_calls)): clean_login+=1
        else: no_login_call+=1   # passed without calling any auth tool (OR-branch / no-auth task)
    return (n, succ, quirk, clean_login, no_login_call)

print("quirk = SUCCESS tasks where an auth tool was CALLED but ALL returned False (failed-login-but-passed)")
print(f"{'model':<46}{'succ/n':>10}{'quirk':>7}{'clean-auth':>11}{'no-auth-call':>13}")
rows=[]
for fp in sorted(glob.glob("/home/woori/scratch/SOPBench/output/bank/ast_*tool_full*.json")):
    r=analyze(fp)
    if not r or r[0]==0: continue
    rows.append((r[1],fp.split("/")[-1].replace("ast_","")[:44],r))
for _,name,r in sorted(rows, reverse=True):
    n,s,q,c,nl=r
    print(f"{name:<46}{(str(s)+'/'+str(n)):>10}{q:>7}{c:>11}{nl:>13}")

print("\n=== OUR runs ===")
for nm,fp in [("base_noaug","eval_t1c_base_noaug"),("loginfirst","eval_t1c_l1"),("logincall","eval_t1c_logincall"),("s0(fullstack)","eval_t1c_s0"),("stopsuccess(S1)","eval_t1c_stopsuccess")]:
    p=f"/home/woori/scratch/sft_alias_run/{fp}/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
    r=analyze(p)
    if not r: print(f"  {nm}: NA"); continue
    n,s,q,c,nl=r
    print(f"  {nm:<16} succ={s}/{n}  quirk(failed-login-but-passed)={q}  clean-auth={c}  no-auth-call={nl}")
