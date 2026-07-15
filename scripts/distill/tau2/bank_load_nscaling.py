# -*- coding: utf-8 -*-
"""부하 N-scaling 결정적 실험 (2026-07-15) — "horizon=부하" 직접 실증.
per-step skip율(liability-wrong·reach-miss)을 N(sim당 gold dispute 수)으로 층화.
N↑일수록 skip↑이면 = 부하가 N에 따라 커짐 = horizon 곱붕괴를 부하가 몰고 감. 로컬 무료."""
import json,glob,re,sys,io
from datetime import datetime
from collections import defaultdict
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
def nd(x):
    if isinstance(x,str):
        try:x=json.loads(x)
        except:return {}
    return x if isinstance(x,dict) else {}
fam=lambda n:re.sub(r"_\d+$","",str(n))
def amt(v):
    try:return round(abs(float(re.sub(r"[$,]","",str(v)))),2)
    except:return None

def main():
    # N-bucket → 통계
    by_n=defaultdict(lambda:{"sims":0,"gold":0,"missed":0,"idcorrect":0,"liab_wrong":0})
    for f in glob.glob("C:/tmp/traj/*_banking.json"):
        d=json.load(open(f,encoding="utf-8"))
        for s in d.get("simulations",[]):
            ri=s.get("reward_info") or {}
            if ri.get("reward") is None: continue
            # gold disputes
            gm={}
            for ac in (ri.get("action_checks") or []):
                a=ac.get("action") or {}
                if "transaction_dispute" in fam(nd(a.get("arguments")).get("agent_tool_name","")):
                    ga=nd(nd(a.get("arguments")).get("arguments")); tid=str(ga.get("transaction_id") or "")
                    if tid:gm[tid]=ga
            if not gm: continue
            N=len(gm)                          # sim당 gold dispute 수 = 부하 N
            nb=5 if N>=5 else N                 # bucket
            # agent submissions
            am={}
            for m in (s.get("messages") or []):
                for tc in (m.get("tool_calls") or []):
                    if tc.get("name")=="call_discoverable_agent_tool" and "transaction_dispute" in fam(nd(tc.get("arguments")).get("agent_tool_name","")):
                        a=nd(nd(tc.get("arguments")).get("arguments")); tid=str(a.get("transaction_id") or "")
                        if tid:am.setdefault(tid,a)
            st=by_n[nb]; st["sims"]+=1; st["gold"]+=N
            for tid,ga in gm.items():
                if tid not in am: st["missed"]+=1; continue
                if ga.get("customer_max_liability_amount") is not None:
                    st["idcorrect"]+=1
                    if amt(am[tid].get("customer_max_liability_amount"))!=amt(ga.get("customer_max_liability_amount")):
                        st["liab_wrong"]+=1
    print("=== 부하 N-scaling (N=sim당 gold dispute 수) ===")
    print("%-6s %6s %6s | reach-miss%%   liability-wrong%%(id-correct중)"%("N","sims","gold"))
    for nb in sorted(by_n):
        st=by_n[nb]
        mr=100*st["missed"]/max(st["gold"],1)
        lw=100*st["liab_wrong"]/max(st["idcorrect"],1)
        lab=("%d"%nb) if nb<5 else "5+"
        print("%-6s %6d %6d | %6.1f       %6.1f  (id-correct n=%d)"%(lab,st["sims"],st["gold"],mr,lw,st["idcorrect"]))
    print("\n★N↑일수록 reach-miss·liability-wrong 오르면 = 부하 N-scaling = horizon 곱붕괴를 부하가 몰고 감(직접 실증).")

if __name__=="__main__":main()
