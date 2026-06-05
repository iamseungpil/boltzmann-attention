#!/usr/bin/env python3
"""Quirk re-score: strict_success requires GENUINE auth when the SOP requires it.
Official success (leaderboard) counts a FAILED login (login_user->False) as satisfying the dirgraph
login node (dfscheck = call-order, not auth-success). Strict success additionally requires:
  if directed_action_graph contains login_user  -> some login_user call returned True
  if it contains authenticate_admin_password     -> some authenticate call returned True
Applied UNIFORMLY to released leaderboard files and our runs. Reports official% vs strict% (134, tool_full).
Zero-cost (re-scores existing eval JSONs)."""
import json, glob, sys

def truthy(c):
    if c is True: return True
    if isinstance(c,(list,tuple)) and len(c): return c[0] is True
    if isinstance(c,str): return c.strip().startswith(("True","(True","[True"))
    return False

def rescore(path):
    try: d=json.load(open(path))
    except Exception: return None
    if not isinstance(d,list): return None
    n=off=strict=0
    for e in d:
        if not isinstance(e,dict): continue
        evs=e.get("evaluations") or []
        if not evs: continue
        ev=evs[0]; t=e.get("task",{})
        n+=1
        o=bool(ev.get("success")); off+=int(o)
        # auth requirements from directed_action_graph
        nodes=(t.get("directed_action_graph") or {}).get("nodes") or []
        req_login=any((not isinstance(nd,str)) and nd[0]=="login_user" for nd in nodes)
        req_admin=any((not isinstance(nd,str)) and nd[0]=="authenticate_admin_password" for nd in nodes)
        # genuine auth from trace
        login_ok=admin_ok=False
        for il in (e.get("interactions") or []):
            conv=il.get("interaction") if isinstance(il,dict) else None
            if not conv: continue
            for m in conv:
                if not isinstance(m,dict): continue
                tn=m.get("tool_name")
                if tn=="login_user" and truthy(m.get("content")): login_ok=True
                if tn=="authenticate_admin_password" and truthy(m.get("content")): admin_ok=True
        s = o and (login_ok or not req_login) and (admin_ok or not req_admin)
        strict+=int(s)
    return (n, off, strict)

def pct(x,n): return f"{100.0*x/n:.2f}%" if n else "-"

print("=== RELEASED bank leaderboard: official vs strict (quirk-excluded) ===")
print(f"{'model':<46}{'official':>10}{'strict':>10}{'quirk-Δ':>9}")
rows=[]
for fp in sorted(glob.glob("/home/woori/scratch/SOPBench/output/bank/ast_*tool_full*.json")):
    r=rescore(fp)
    if not r or r[0]==0: continue
    n,o,s=r
    name=fp.split("/")[-1].replace("ast_","")[:44]
    rows.append((o,name,n,o,s))
for _,name,n,o,s in sorted(rows, reverse=True):
    print(f"{name:<46}{pct(o,n):>10}{pct(s,n):>10}{('-'+str(o-s)):>9}")

print("\n=== OUR runs: official vs strict ===")
for nm,fp in [("base_noaug","eval_t1c_base_noaug"),("loginfirst","eval_t1c_l1"),("logincall","eval_t1c_logincall"),("s0(fullstack)","eval_t1c_s0"),("stopsuccess(S1)","eval_t1c_stopsuccess")]:
    p=f"/home/woori/scratch/sft_alias_run/{fp}/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
    r=rescore(p)
    if not r: print(f"  {nm}: NA"); continue
    n,o,s=r
    print(f"  {nm:<16} official={pct(o,n)} ({o}/{n})  strict={pct(s,n)} ({s}/{n})  quirk-Δ=-{o-s}")
