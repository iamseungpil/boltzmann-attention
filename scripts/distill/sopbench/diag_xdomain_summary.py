import json, glob, sys
def official(path):
    try: d=json.load(open(path))
    except: return None
    if not isinstance(d,list) or not d: return None
    n=s=sT=nT=sF=nF=0
    for e in d:
        ev=(e.get("evaluations") or [{}])[0]; n+=1; ok=bool(ev.get("success")); s+=ok
        if ev.get("action_should_succeed"): nT+=1; sT+=ok
        else: nF+=1; sF+=ok
    return (n,s,sT,nT,sF,nF)
DOMS=["bank","dmv","healthcare","hotel","library","online_market","university"]
# leaderboard per domain (released, official success, tool_full) — recompute max + qwen7b
print(f"{'domain':<14}{'adapter-only':>14}{'STACK':>14}{'scaffoldΔ':>11}{'LB-max(rel)':>22}")
for D in DOMS:
    a=official(f"/home/woori/scratch/sft_alias_run/xdom_{D}_adapteronly/{D}/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json")
    s=official(f"/home/woori/scratch/sft_alias_run/xdom_{D}_stack/{D}/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json")
    # released LB per domain: max official over tool_full files
    best=("",0); 
    for fp in glob.glob(f"/home/woori/scratch/SOPBench/output/{D}/ast_*tool_full*.json"):
        r=official(fp)
        if r and r[0]>0 and 100*r[1]/r[0]>best[1]: best=(fp.split("/")[-1][:24],100*r[1]/r[0])
    ap=f"{100*a[1]/a[0]:.1f}%({a[1]}/{a[0]})" if a else "NA"
    st=f"{100*s[1]/s[0]:.1f}%({s[1]}/{s[0]})" if s else "NA"
    dl=f"+{100*s[1]/s[0]-100*a[1]/a[0]:.0f}" if (a and s) else "-"
    print(f"{D:<14}{ap:>14}{st:>14}{dl:>11}{(best[0]+' '+f'{best[1]:.1f}%'):>22}")
