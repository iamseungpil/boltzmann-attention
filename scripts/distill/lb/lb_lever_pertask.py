# -*- coding: utf-8 -*-
"""Per-task 2x2 for the lever census: for each named task, every lever fired-vs-quiet x win, pooled
over every arm in out_lb, gold never read.  python lb_lever_pertask.py 066,074,075 [out_lb]"""
import collections, glob, gzip, json, os, re, sys
O=sys.argv[2] if len(sys.argv)>2 else "/home/woori/scratch/x768/out_lb"
SKIP=(".infra_void",".run1",".partial",".first",".pre_",".taint",".nt2","x818cloud"); SKIP_PRE={"smk","probe","dx","dbg"}
TASKS=sys.argv[1].split(",")
TAG=re.compile(r"\[([A-Z][A-Z_ -]{2,30})\]")
def keys(r):
    k,src,lb=r.get("kind"),str(r.get("source") or ""),str(r.get("lb") or ""); tag=TAG.search(str(r.get("text") or "")); tag=tag.group(1) if tag else ""
    if k in ("lb-advice","lb-deny","lb-block"): return ["%s:%s:%s"%(k[3:],lb or "?",src.split(":")[0] or tag or "?")]
    if k=="lb-tool": return ["tool:%s"%src]
    if k in ("lb-inject","lb-fold","lb-regen"): return [k[3:]]
    return []
for T in TASKS:
    cnt=collections.defaultdict(lambda:[0,0,0,0]); arms=collections.Counter(); wins=collections.Counter(); nsim=0
    for rp in sorted(glob.glob(O+"/*_task_%s.results.json.gz"%T)):
        b=os.path.basename(rp)
        if any(x in b for x in SKIP) or b.startswith("bank_x806"): continue
        pre=b.rsplit("_task_",1)[0]
        if pre in SKIP_PRE: continue
        d=json.load(gzip.open(rp,"rt",encoding="utf-8")); ss=d.get("simulations") or []
        if any(x.get("termination_reason")=="infrastructure_error" for x in ss): continue
        side=collections.defaultdict(set); sp=O+"/fb_%s_task_%s.jsonl.gz"%(pre,T)
        if os.path.exists(sp):
            for line in gzip.open(sp,"rt",encoding="utf-8",errors="replace"):
                try: r=json.loads(line)
                except Exception: continue
                for k in keys(r): side[str(r.get("sim"))].add(k)
        allk=set(k for s in side.values() for k in s)
        for s in ss:
            rw=1 if (s.get("reward_info") or {}).get("reward")==1.0 else 0; ks=side.get(str(s.get("id")),set()); nsim+=1; arms[pre]+=1; wins[pre]+=rw
            for k in allk:
                c=cnt[k]
                if k in ks: c[0 if rw else 1]+=1
                else: c[2 if rw else 3]+=1
    print("\n##### task_%s  sims %d  win %d  arms %s"%(T,nsim,sum(wins.values())," ".join("%s:%d/%d"%(a,wins[a],arms[a]) for a in sorted(arms))))
    rows=sorted(cnt.items(), key=lambda kv:-(kv[1][0]+kv[1][1]))
    for k,(fw,fl,qw,ql) in rows[:14]:
        fr=100.0*fw/(fw+fl) if fw+fl else float('nan'); qr=100.0*qw/(qw+ql) if qw+ql else float('nan')
        print("  %-44s fired %3d/%3d (%5.1f%%)  quiet %3d/%3d (%5.1f%%)"%(k,fw,fw+fl,fr,qw,qw+ql,qr))
