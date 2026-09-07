# -*- coding: utf-8 -*-
"""x923 — 회수분 재생: _have_value_reask_fb / _value_acquire_fb 실측 발화."""
import json,os,sys,io,collections
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
import t2_gate_patch as G
A2=json.load(open("a2/banking_knowledge.gate.json",encoding="utf-8"))
HV=A2.get("have_value_reask") or []; VA=A2.get("value_acquisition") or []
print("HV specs",len(HV),[s.get("write") for s in HV])
print("VA specs",len(VA),[s.get("write") for s in VA])
class TC:
    def __init__(s,d): s.name=d.get("name"); s.arguments=d.get("args"); s.id=None
class M:
    def __init__(s,d):
        s.role=d["r"]; s.content=d["c"]; s.error=d["e"]; s.id=None
        s.tool_calls=[TC(t) for t in d["tc"]] if d["tc"] else None
P=r"C:\Users\승원\AppData\Local\Temp\claude\C--workspace\7fc6c1a1-f227-4592-be9c-44f0ba6cffac\scratchpad\x921_slim.jsonl"
hv=[];va=[];nsim=0;nast=0;nmsg=0
for ln in open(P,encoding="utf-8"):
    d=json.loads(ln);nsim+=1
    msgs=[M(m) for m in d["msgs"]];nmsg+=len(msgs)
    for i,am in enumerate(msgs):
        if am.role!="assistant": continue
        nast+=1;pre=msgs[:i]
        try: h=G._have_value_reask_fb(am,pre,HV)
        except Exception: h=None
        try: v=G._value_acquire_fb(am,pre,VA,a2=None,executed=set())
        except Exception: v=None
        if h: hv.append((d["tag"],d["task"],d["sim"],d["reward"],i,am.content or ""))
        if v: va.append((d["tag"],d["task"],d["sim"],d["reward"],i,am.content or ""))
print("sims",nsim,"msgs",nmsg,"assistant",nast)
print("HV fires",len(hv),"VA fires",len(va))
def rep(n,H):
    print("=== %s ==="%n)
    print(" tasks",dict(collections.Counter(x[1] for x in H)))
    S={(x[2],x[3]) for x in H}
    print(" sims",len(S)," reward분포",dict(collections.Counter(r for _,r in S)))
    print(" 물음표 포함 %d/%d"%(sum(1 for x in H if "?" in x[5]),len(H)))
rep("HV",hv);rep("VA",va)
json.dump(hv,open("x923_hv.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
json.dump(va,open("x923_va.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
print("\n--- HV 발화 축자 전건 ---")
for x in hv:
    print("[%s %s r=%s msg%d]"%(x[1],x[2][:8],x[3],x[4]));print((x[5] or "")[:420]);print()
