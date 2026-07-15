# -*- coding: utf-8 -*-
"""reach 잔여 정밀분해 (C91 이론검증·C80 §14.6 정밀화·2026-07-15).
0-제출 gold dispute를 (a)enumerable(그 txn_id가 agent가 본 tool result/user 메시지에 등장)
vs (b)undiscovered(어디에도 없음)로 분해. enumerable 지배면 open-world 잔여 작음=H_min+열거로 닫힘.
로컬 무료(C:/tmp/traj·[[09]]무관)."""
import json,glob,re,sys,io
from collections import Counter
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
def nd(x):
    if isinstance(x,str):
        try:return json.loads(x)
        except:return {}
    return x if isinstance(x,dict) else {}
fam=lambda n:re.sub(r"_\d+$","",str(n))
TXN=re.compile(r"txn_[0-9a-f]+")

def gold_disputes(s):
    ri=s.get("reward_info") or {}; out=set()
    for ac in (ri.get("action_checks") or []):
        a=ac.get("action") or {}
        if "transaction_dispute" in fam(nd(a.get("arguments")).get("agent_tool_name","")):
            tid=nd(nd(a.get("arguments")).get("arguments")).get("transaction_id")
            if tid: out.add(str(tid))
    return out

def agent_subs(s):
    out=set()
    for m in (s.get("messages") or []):
        for tc in (m.get("tool_calls") or []):
            if tc.get("name")=="call_discoverable_agent_tool" and "transaction_dispute" in fam(nd(tc.get("arguments")).get("agent_tool_name","")):
                tid=nd(nd(tc.get("arguments")).get("arguments")).get("transaction_id")
                if tid: out.add(str(tid))
    return out

def seen_txns(s):
    """agent가 본(=tool result + user 메시지) 모든 txn_id."""
    ids=set()
    for m in (s.get("messages") or []):
        if m.get("role") in ("tool","user"):
            ids|=set(TXN.findall(str(m.get("content"))))
    return ids

def main():
    files=glob.glob("C:/tmp/traj/*_banking.json")
    cat=Counter(); by_model=Counter(); n_sims=0; n_disp_sims=0
    ex_und=[]
    for f in files:
        model=f.replace("\\","/").split("/")[-1].replace("_banking.json","")
        d=json.load(open(f,encoding="utf-8"))
        for s in d.get("simulations",[]):
            g=gold_disputes(s)
            if not g: continue
            n_disp_sims+=1
            sub=agent_subs(s); seen=seen_txns(s)
            missed=g-sub                       # 0-제출 gold
            for tid in missed:
                if tid in seen:
                    cat["enumerable(본 적 있음→under-action)"]+=1; by_model[model+":enum"]+=1
                else:
                    cat["undiscovered(어디에도 없음)"]+=1; by_model[model+":und"]+=1
                    if len(ex_und)<5: ex_und.append((s.get("task_id"),model,tid))
        n_sims+=len(d.get("simulations",[]))
    tot=sum(cat.values())
    print("=== reach 잔여 분해 (0-제출 gold dispute·%d dispute-sim·%d 파일) ==="%(n_disp_sims,len(files)))
    print("총 0-제출 gold dispute: %d"%tot)
    for k,v in cat.most_common():
        print("  %-34s %5d  (%.1f%%)"%(k,v,100*v/max(tot,1)))
    print("\n★enumerable 지배 → open-world 잔여 작음 = H_min+열거가능-원천으로 reach 닫힘(C91 §6 지지)")
    print("  undiscovered 예시(task·model·tid):")
    for t,m,tid in ex_und: print("   ",t,m,tid[-8:])

if __name__=="__main__":
    main()

def refine():
    """★[[08]] 정밀화: '못 봄'을 queryable(원천 존재·다른 run이 surface함) vs never-surfaced로 분해."""
    import glob as G
    files=G.glob("C:/tmp/traj/*_banking.json")
    # per-task queryable universe = 그 task의 어느 sim이든 tool result에 등장한 txn
    universe={}
    data={}
    for f in files:
        d=json.load(open(f,encoding="utf-8")); data[f]=d
        for s in d.get("simulations",[]):
            t=str(s.get("task_id"))
            u=universe.setdefault(t,set())
            for m in (s.get("messages") or []):
                if m.get("role")=="tool": u|=set(TXN.findall(str(m.get("content"))))
    c=Counter()
    for f,d in data.items():
        for s in d.get("simulations",[]):
            g=gold_disputes(s)
            if not g: continue
            t=str(s.get("task_id"))
            sub=agent_subs(s); seen=seen_txns(s); uni=universe.get(t,set())
            for tid in (g-sub):
                if tid in seen: c["A.enumerable(본 적)→under-action(ACT)"]+=1
                elif tid in uni: c["B.queryable(원천有·미조회)→under-action(ENUM)"]+=1
                else: c["C.never-surfaced(open-world 후보)"]+=1
    tot=sum(c.values())
    print("\n=== ★[[08]] 정밀화: 0-제출 gold dispute 3분 ===")
    for k,v in c.most_common(): print("  %-42s %5d (%.1f%%)"%(k,v,100*v/max(tot,1)))
    cw=c["A.enumerable(본 적)→under-action(ACT)"]+c["B.queryable(원천有·미조회)→under-action(ENUM)"]
    print("\n★closed-world(A+B·원천 존재·under-action) = %d (%.1f%%)"%(cw,100*cw/max(tot,1)))
    print("★open-world 후보(C·never-surfaced) = %d (%.1f%%)"%(c["C.never-surfaced(open-world 후보)"],100*c["C.never-surfaced(open-world 후보)"]/max(tot,1)))
    print("  (C도 완비 GET 도구 존재하면 closed-world·이 측정은 '어느 run도 surface 안 함'=상한)")

refine()
