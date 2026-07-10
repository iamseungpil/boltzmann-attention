#!/usr/bin/env python3
"""신형 frontier(gpt-5.2 usersim) 궤적 기능 분해 — db_fail을 F1-F6/⋈/over-action으로.
정본 doc TAU2_FRONTIER_TRAJECTORY_INVESTIGATION_MASTER §3.2 확장. 기준=db_match. 로컬 실행."""
import json, io, glob, os, itertools
from collections import Counter

WRITE = {"return_delivered_order_items","exchange_delivered_order_items","cancel_pending_order",
         "modify_pending_order_items","modify_pending_order_address","modify_pending_order_payment",
         "modify_user_address","place_order"}

def ri(s): return s.get("reward_info") or {}
def dbm(s): return (ri(s).get("db_check") or {}).get("db_match")
def term(s): return str(s.get("termination_reason"))
def gold_w(s):
    return [(a.get("action",{}).get("name"), a.get("action",{}).get("arguments") or {})
            for a in (ri(s).get("action_checks") or []) if a.get("action",{}).get("name") in WRITE]
def exec_w(s):
    by={m.get("id") or m.get("tool_call_id"):m for m in s.get("messages",[]) if m.get("role")=="tool"}
    out=[]
    for m in s.get("messages",[]):
        if m.get("role")!="assistant": continue
        for tc in m.get("tool_calls") or []:
            if tc.get("name") not in WRITE: continue
            tm=by.get(tc.get("id"))
            if tm is not None and not tm.get("error"): out.append((tc["name"], tc.get("arguments") or {}))
    return out
def _n(v): return sorted(str(x) for x in v) if isinstance(v,list) else v
def dkeys(g,o): return [k for k in sorted((set(g)|set(o))-{"user_id"}) if _n(g.get(k))!=_n(o.get(k))]
def cost(g,o): return 100 if g[0]!=o[0] else len(dkeys(g[1],o[1]))
def pair(g,o):
    best=None
    for pm in itertools.permutations(range(len(o))):
        c=sum(cost(g[i],o[pm[i]]) for i in range(len(g)))
        if best is None or c<best[0]: best=(c,pm)
    return best[1]
def classify(keys,ga,oa):
    if "order_id" in keys: return "F3_join_order"
    if "item_ids" in keys and "new_item_ids" in keys: return "item+new_mix"
    if "new_item_ids" in keys: return "F2_variant"
    if "item_ids" in keys:
        g=_n(ga.get("item_ids") or []); o=_n(oa.get("item_ids") or [])
        return "F4_item_over" if len(o)>len(g) else ("F4_item_under" if len(o)<len(g) else "item_wrong")
    if "payment_method_id" in keys: return "payment"
    if any(k.startswith("address") or k in ("city","state","zip","country") for k in keys): return "address"
    if "reason" in keys: return "reason_enum"
    if "amount" in keys: return "amount"
    return "other:"+",".join(keys)
def bucket(s):
    need,did=len(gold_w(s)),len(exec_w(s))
    if need==0 and did>0: return "EXTRA_gold0"
    if did==0 and need>0: return "ZERO"
    if did<need: return "FEWER"
    if did>need: return "MORE"
    return "SAME"

def analyze(path):
    d=json.load(io.open(path,encoding="utf-8"))
    sims=d.get("simulations") if isinstance(d,dict) else d
    N=len(sims)
    terms=Counter(term(s) for s in sims)
    infra=sum(v for k,v in terms.items() if "infra" in k.lower() or "error" in k.lower())
    dbv=[dbm(s) for s in sims if dbm(s) is not None]
    dbpass=sum(1 for x in dbv if x)/max(len(dbv),1)
    fails=[s for s in sims if dbm(s) is False]
    buck=Counter(bucket(s) for s in fails)
    cls=Counter(); f2=f3=0
    for s in fails:
        G,O=gold_w(s),exec_w(s)
        if not G or len(G)!=len(O) or len(G)>6: continue
        pm=pair(G,O); h2=h3=False
        for i in range(len(G)):
            gn,ga=G[i]; on,oa=O[pm[i]]
            if gn!=on: cls["op_mismatch"]+=1; continue
            k=dkeys(ga,oa)
            if not k: continue
            c=classify(k,ga,oa); cls[c]+=1
            if c=="F2_variant": h2=True
            if c=="F3_join_order": h3=True
        f2+=h2; f3+=h3
    return N,dbpass,len(fails),infra,dict(terms),dict(buck),dict(cls),f2,f3

files=sorted(glob.glob(r"C:/tmp/traj/*_retail.json"))
print(f"{'model':14}{'N':>4}{'dbP':>6}{'fail':>5}{'infra':>6}  F2var%  F3join%   top-classes")
rows=[]
for f in files:
    nm=os.path.basename(f).replace("_retail.json","")
    try:
        N,dbp,nf,inf,tm,bk,cl,f2,f3=analyze(f)
    except Exception as e:
        print(f"{nm:14} ERR {str(e)[:50]}"); continue
    top=", ".join(f"{k}={v}" for k,v in sorted(cl.items(),key=lambda x:-x[1])[:4])
    print(f"{nm:14}{N:>4}{dbp:>6.3f}{nf:>5}{inf:>6}  {100*f2/N:>5.1f}  {100*f3/N:>6.1f}   {top}")
    rows.append((nm,dbp,f2,f3,N,bk,cl))
