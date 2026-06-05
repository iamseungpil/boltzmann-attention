#!/usr/bin/env python3
"""Held-out cross-domain transfer summary: per held-out domain, adapter-only vs STACK official success
(+ should_T/F split + should_T quirk count), vs released leaderboard max + Qwen2.5-7B (base ref).
Transfer = test on a domain the adapter NEVER trained on (lodo_<X> tested on X; train1_bank tested on
the 6 non-bank). bank(lodo_bank)=43.28% already. scaffold Δ = stack - adapter-only."""
import json, glob
OUT="/home/woori/scratch/sft_alias_run"
def stats(path):
    try: d=json.load(open(path))
    except Exception: return None
    if not isinstance(d,list) or not d: return None
    n=s=sT=nT=sF=nF=q=0
    def truthy(c):
        if c is True: return True
        if isinstance(c,(list,tuple)) and len(c): return c[0] is True
        if isinstance(c,str): return c.strip().startswith(("True","(True","[True"))
        return False
    for e in d:
        ev=(e.get("evaluations") or [{}])[0]; n+=1; ok=bool(ev.get("success")); s+=ok
        if ev.get("action_should_succeed"):
            nT+=1; sT+=ok
            if ok:
                lc=[]
                for il in e.get("interactions") or []:
                    for m in il.get("interaction") or []:
                        if isinstance(m,dict) and m.get("tool_name")=="login_user": lc.append(truthy(m.get("content")))
                if lc and not any(lc): q+=1
        else: nF+=1; sF+=ok
    return (n,s,sT,nT,sF,nF,q)
def lb(domain):
    best=("",0); q7=0
    for fp in glob.glob(f"/home/woori/scratch/SOPBench/output/{domain}/ast_*tool_full*.json"):
        r=stats(fp)
        if r and r[0]>0:
            pct=100*r[1]/r[0]
            if pct>best[1]: best=(fp.split("/")[-1].split("-mode")[0].replace("ast_","")[:18],pct)
            if "qwen2.5-7b" in fp and "react" in fp: q7=pct
    return best,q7

# (tag, domain) held-out pairs
PAIRS=[("lodo_library","library"),("lodo_healthcare","healthcare"),
       ("train1bank","dmv"),("train1bank","healthcare"),("train1bank","hotel"),
       ("train1bank","library"),("train1bank","online_market"),("train1bank","university")]
print(f"{'tag/domain':<28}{'adapter-only':>14}{'STACK(official)':>17}{'Δ':>6}{'sT quirk':>9}{'LB-max':>22}{'Qwen7B':>8}")
print("="*108)
# bank (existing held-out) reference
b=stats(f"{OUT}/eval_t1c_headline/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json")
if b:
    (bm,_),bq=lb("bank"); print(f"{'lodo_bank/bank(headline)':<28}{'-':>14}{f'{100*b[1]/b[0]:.1f}%({b[1]}/{b[0]})':>17}{'-':>6}{'?':>9}{f'{bm} {_:.1f}%':>22}{f'{bq:.1f}%':>8}")
for tag,D in PAIRS:
    a=stats(f"{OUT}/xho_{tag}_{D}_adapteronly/{D}/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json")
    s=stats(f"{OUT}/xho_{tag}_{D}_stack/{D}/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json")
    (bm,bp),bq=lb(D)
    ap=f"{100*a[1]/a[0]:.1f}%" if a else "NA"
    st=f"{100*s[1]/s[0]:.1f}%({s[1]}/{s[0]})" if s else "NA"
    dl=f"+{100*s[1]/s[0]-100*a[1]/a[0]:.0f}" if (a and s) else "-"
    sq=str(s[6]) if s else "-"
    print(f"{(tag+'/'+D):<28}{ap:>14}{st:>17}{dl:>6}{sq:>9}{f'{bm} {bp:.1f}%':>22}{f'{bq:.1f}%':>8}")
