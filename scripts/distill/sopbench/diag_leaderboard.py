#!/usr/bin/env python3
"""Confirm the leaderboard metric = official `success` (pass rate %) over ALL tasks (should_T+should_F),
and put our runs on the SAME basis. README bank pass rates: Qwen2.5-7B(ReAct) 5.22, Claude-3-5-Sonnet 71.90, etc."""
import json, glob, collections

def passrate(path):
    try: d=json.load(open(path))
    except Exception: return None
    if not isinstance(d,list): return None
    nT=sT=nF=sF=0
    for e in d:
        evs=e.get("evaluations") or []
        if not evs: continue
        ev=evs[0]
        should=ev.get("action_should_succeed")
        succ=bool(ev.get("success"))
        if should: nT+=1; sT+=int(succ)
        else: nF+=1; sF+=int(succ)
    n=nT+nF; s=sT+sF
    return (n, s, 100.0*s/n if n else 0, nT, sT, nF, sF)

print("=== RELEASED bank leaderboard (official success pass rate %) ===")
print(f"{'file':<58}{'all%':>7}{'(succ/n)':>10}{'  shouldT':>10}{'  shouldF':>10}")
for fp in sorted(glob.glob("/home/woori/scratch/SOPBench/output/bank/ast_*.json")):
    r=passrate(fp)
    if not r: continue
    n,s,pct,nT,sT,nF,sF=r
    name=fp.split("/")[-1].replace("ast_","")[:56]
    print(f"{name:<58}{pct:>6.2f}%{('('+str(s)+'/'+str(n)+')'):>11}{(str(sT)+'/'+str(nT)):>10}{(str(sF)+'/'+str(nF)):>10}")

print("\n=== OUR runs (same official-success basis) ===")
for nm,fp in [("base_noaug","eval_t1c_base_noaug"),("l1(loginfirst)","eval_t1c_l1"),("logincall","eval_t1c_logincall"),("dggate(aug)","eval_t1c_dggate")]:
    p=f"/home/woori/scratch/sft_alias_run/{fp}/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
    r=passrate(p)
    if not r: print(f"  {nm}: NA"); continue
    n,s,pct,nT,sT,nF,sF=r
    print(f"  {nm:<16} all={pct:5.2f}% ({s}/{n})  shouldT={sT}/{nT}  shouldF={sF}/{nF}")
