# -*- coding: utf-8 -*-
"""x775 — 캠페인 회수분에서 T2_DISTINCT_ARGS / T2_GROUP_DUP 술어 발화 실측.
엔진 사본 0 (distinct_args_violation / group_dup_value 직접 임포트). CPU · 모델 0 · GPU 0.
"""
import os, sys, gzip, json, glob
sys.path.insert(0, r"C:\workspace\ba-frft\scripts\distill\tau2")
from t2_gate_patch import distinct_args_violation, group_dup_value, _exact_tool_name
from gate_interpreter import load_domain_a2
A2 = load_domain_a2("banking_knowledge")
SR = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"

class TC(object):
    def __init__(s, name, args):
        s.name = name; s.arguments = args
        s.function = type("F", (), {"name": name, "arguments": args})()

def ts_of(s):
    for k in ("start_time","timestamp","created_at","end_time"):
        if s.get(k): return str(s[k])
    return ""

best = {}
for p in sorted(glob.glob(os.path.join(SR, "bank_*.results.json.gz"))):
    tag = os.path.basename(p).replace(".results.json.gz","")
    try:
        with gzip.open(p,"rt",encoding="utf-8",errors="replace") as f: d=json.load(f)
    except Exception: continue
    for s in (d.get("simulations") or d.get("results") or []):
        if not isinstance(s,dict): continue
        t=str(s.get("task_id") or s.get("task") or ""); ts=ts_of(s)
        if not ("2026-09-03T13" <= ts <= "2026-09-05T03:30"): continue
        if t not in best or ts>best[t][0]: best[t]=(ts,tag,s)

def rw(s):
    r=s.get("reward_info") or {}
    try: return float(r.get("reward", s.get("reward")))
    except Exception: return None

DA={}; GD={}; TOOLSEEN={}
for t,(ts,tag,s) in sorted(best.items()):
    for m in (s.get("messages") or []):
        if (m.get("role") or "")!="assistant": continue
        for tc in (m.get("tool_calls") or []):
            fn=tc.get("function") or tc
            o=TC(fn.get("name") or tc.get("name"), fn.get("arguments", tc.get("arguments")))
            nm=_exact_tool_name(o) or o.name
            if "dispute" in str(nm): TOOLSEEN.setdefault(nm,set()).add(t)
            try: dv=distinct_args_violation(o, A2)
            except Exception: dv=None
            if dv: DA.setdefault(t,[]).append((rw(s),tag,dv))
            try: gv=group_dup_value(o, A2)
            except Exception: gv=None
            if gv: GD.setdefault(t,[]).append((rw(s),tag,gv))

print("tasks=%d pass=%d fail=%d" % (len(best),
      sum(1 for t,(a,b,s) in best.items() if (rw(s) or 0)>=1.0),
      sum(1 for t,(a,b,s) in best.items() if (rw(s) or 0)<1.0)))
print("\n== T2_DISTINCT_ARGS fires: %d tasks" % len(DA))
for t,h in sorted(DA.items()):
    print("  %-10s reward=%s tag=%s n=%d" % (t,h[0][0],h[0][1],len(h)))
    for r,tg,d in h[:4]: print("      ",d)
print("\n== T2_GROUP_DUP fires: %d tasks" % len(GD))
for t,h in sorted(GD.items()):
    print("  %-10s reward=%s tag=%s n=%d  ex=%s" % (t,h[0][0],h[0][1],len(h),h[0][2]))
print("\n== dispute 도구 이름 분포 (선언은 _6281 하나뿐) ==")
for nm,ts in sorted(TOOLSEEN.items()): print("  %-52s tasks=%s" % (nm, sorted(ts)))
