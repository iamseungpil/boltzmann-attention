# -*- coding: utf-8 -*-
"""① E-PLAN 오프라인 replay ([[09]] 무료 先·2026-07-15) — COMPUTE 연산 강제가 pass 올리나(offline 상한).
실패 sim의 id-correct dispute에 결정론 COMPUTE(liability lookup) 적용 → dispute가 완전정답 되나(flip).
+ sim-pass 상한(compute fix + 모든 gold 제출 가정). 라이브 e2e=유료(승인·컨트롤러 배선 필요)."""
import json,glob,re,sys,io
from datetime import datetime
from collections import Counter
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
def nd(x):
    if isinstance(x,str):
        try:x=json.loads(x)
        except:return {}
    return x if isinstance(x,dict) else {}
fam=lambda n:re.sub(r"_\d+$","",str(n))
COMPUTE={"customer_max_liability_amount","provisional_credit_eligible","eligible_for_provisional_credit","partial_refund_amount","card_action"}
def pd(s):
    for f in ("%m/%d/%Y","%m/%d/%y","%m/%d"):
        try:return datetime.strptime(str(s).strip()[:10],f)
        except:pass
    return None
def amt(v):
    try:return round(abs(float(re.sub(r"[$,]","",str(v)))),2)
    except:return None
def norm(v):
    s=str(v).strip().lower(); m=re.sub(r"[$,]","",s)
    try:return round(float(m),2)
    except:pass
    return {"true":True,"yes":True,"false":False,"no":False}.get(s,s)
def liability(ga):
    a=pd(ga.get("transaction_date")); b=pd(ga.get("discovery_date"))
    if a and b:
        dd=(b-a).days
        return 50.0 if dd<=30 else (500.0 if dd<=60 else amt(ga.get("disputed_amount")))
    return None
def main():
    disp_fix=Counter(); sim=Counter()
    for f in glob.glob("C:/tmp/traj/*_banking.json"):
        d=json.load(open(f,encoding="utf-8"))
        for s in d.get("simulations",[]):
            ri=s.get("reward_info") or {}
            if ri.get("reward") in (None,1.0): continue    # 실패 sim
            gm={}
            for ac in (ri.get("action_checks") or []):
                a=ac.get("action") or {}
                if "transaction_dispute" in fam(nd(a.get("arguments")).get("agent_tool_name","")):
                    ga=nd(nd(a.get("arguments")).get("arguments")); tid=str(ga.get("transaction_id") or "")
                    if tid:gm[tid]=ga
            if not gm:continue
            am={}
            for m in (s.get("messages") or []):
                for tc in (m.get("tool_calls") or []):
                    if tc.get("name")=="call_discoverable_agent_tool" and "transaction_dispute" in fam(nd(tc.get("arguments")).get("agent_tool_name","")):
                        a=nd(nd(tc.get("arguments")).get("arguments")); tid=str(a.get("transaction_id") or "")
                        if tid:am.setdefault(tid,a)
            sim["fail_sims"]+=1
            all_ok_after=True   # 모든 gold dispute가 (제출됐고) compute-fix 후 완전정답?
            for tid,ga in gm.items():
                if tid not in am: all_ok_after=False; continue  # 0제출=reach(compute로 안 닫힘)
                aa=dict(am[tid])
                # 원래 완전정답?
                wrong0={k for k,gv in ga.items() if k!="transaction_id" and norm(aa.get(k))!=norm(gv)}
                if not wrong0: disp_fix["already_ok"]+=1; continue
                # COMPUTE 강제: compute 필드를 결정론 값으로
                aa["customer_max_liability_amount"]=liability(ga)
                wrong1={k for k,gv in ga.items() if k!="transaction_id" and norm(aa.get(k))!=norm(gv)}
                if not wrong1: disp_fix["FLIP(COMPUTE로 완전정답)"]+=1
                elif wrong1<wrong0: disp_fix["부분개선(잔여 non-compute)"]+=1; all_ok_after=False
                else: disp_fix["미개선"]+=1; all_ok_after=False
            if all_ok_after and all(t in am for t in gm): sim["would_pass_after_COMPUTE"]+=1
    td=sum(v for k,v in disp_fix.items() if k!="already_ok")
    print("=== ① COMPUTE 오프라인 replay (실패 sim·id-correct dispute) ===")
    print("id-correct 오답 dispute %d:"%td)
    for k,v in disp_fix.most_common():
        if k=="already_ok":continue
        print("  %-32s %5d (%.1f%%)"%(k,v,100*v/max(td,1)))
    print("\nsim-level: 실패 sim %d 중 COMPUTE-fix만으로 완전 pass %d (%.1f%%)"%(
        sim["fail_sims"],sim["would_pass_after_COMPUTE"],100*sim["would_pass_after_COMPUTE"]/max(sim["fail_sims"],1)))
    print("  (COMPUTE는 compute-필드만 닫음·reach 0제출/⋈/gather는 별도 연산 필요 → sim-pass 상한 낮음=예상)")
if __name__=="__main__":main()
