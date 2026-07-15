# -*- coding: utf-8 -*-
"""부하 정체 규명 (2026-07-15) — compute 실패(liability 51%오답)를 엄밀 분해:
격리(gold입력)=94% vs in-situ(agent)=49%. 그 gap(=부하)이:
  (a) GATHER 오류: agent 입력facts(date/discovery/amount)가 gold와 다름 → 잘못된 입력에 정확계산
  (b) COMPUTE 오류: 입력 gold와 같은데 liability 공식 틀림
  (c) 혼합
lookup(agent입력) vs agent_liability vs gold_liability 3자 대조로 분해. 로컬 무료."""
import json,gzip,re,sys,io,os
from datetime import datetime
from collections import Counter
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
HERE=os.path.dirname(os.path.abspath(__file__))
GZ=os.path.join(HERE,"..","..","..","reports","facet_rft_2026","sim_results","bank_compute_cases.jsonl.gz")
def pd(s):
    for f in ("%m/%d/%Y","%m/%d/%y","%m/%d"):
        try:return datetime.strptime(str(s).strip()[:10],f)
        except:pass
    return None
def amt(v):
    try:return round(abs(float(re.sub(r"[$,]","",str(v)))),2)
    except:return None
def lookup(td,dd,am):
    a=pd(td); b=pd(dd)
    if a and b:
        days=(b-a).days
        if days<=30:return 50.0
        if days<=60:return 500.0
        return amt(am)
    return None

# compute cases는 gold facts만 저장했으므로, 원 궤적서 agent facts를 다시 뽑아야 함.
def nd(x):
    if isinstance(x,str):
        try:x=json.loads(x)
        except:return {}
    return x if isinstance(x,dict) else {}
fam=lambda n:re.sub(r"_\d+$","",str(n))
import glob
def main():
    cat=Counter(); ex={}
    for f in glob.glob("C:/tmp/traj/*_banking.json"):
        d=json.load(open(f,encoding="utf-8"))
        for s in d.get("simulations",[]):
            ri=s.get("reward_info") or {}
            if ri.get("reward") in (None,1.0): continue
            acalls={}
            for m in (s.get("messages") or []):
                for tc in (m.get("tool_calls") or []):
                    if tc.get("name")=="call_discoverable_agent_tool" and "transaction_dispute" in fam(nd(tc.get("arguments")).get("agent_tool_name","")):
                        a=nd(nd(tc.get("arguments")).get("arguments")); tid=str(a.get("transaction_id") or "")
                        if tid:acalls.setdefault(tid,a)
            for ac in (ri.get("action_checks") or []):
                a=ac.get("action") or {}
                if "transaction_dispute" not in fam(nd(a.get("arguments")).get("agent_tool_name","")): continue
                ga=nd(nd(a.get("arguments")).get("arguments")); tid=str(ga.get("transaction_id") or "")
                if not tid or tid not in acalls or ga.get("customer_max_liability_amount") is None: continue
                aa=acalls[tid]
                gl=amt(ga.get("customer_max_liability_amount")); al=amt(aa.get("customer_max_liability_amount"))
                if gl==al: cat["PASS(liability 정확)"]+=1; continue
                # 입력 facts 대조
                inputs_match = (str(ga.get("transaction_date"))==str(aa.get("transaction_date"))
                                and str(ga.get("discovery_date"))==str(aa.get("discovery_date"))
                                and amt(ga.get("disputed_amount"))==amt(aa.get("disputed_amount")))
                al_from_agent_inputs = lookup(aa.get("transaction_date"),aa.get("discovery_date"),aa.get("disputed_amount"))
                if inputs_match:
                    cat["(b)COMPUTE오류: 입력=gold인데 liability틀림"]+=1
                    ex.setdefault("compute",(tid,ga.get("transaction_date"),ga.get("discovery_date"),gl,al))
                else:
                    # agent 입력이 다름 → agent liability가 자기입력의 정확계산인가?
                    if al_from_agent_inputs is not None and amt(al_from_agent_inputs)==al:
                        cat["(a)GATHER오류: 입력틀림·자기입력엔 정확계산"]+=1
                        ex.setdefault("gather",(tid,"gold(%s→%s)"%(ga.get("transaction_date"),ga.get("discovery_date")),"agent(%s→%s)"%(aa.get("transaction_date"),aa.get("discovery_date")),gl,al))
                    else:
                        cat["(c)혼합: 입력도틀리고 계산도틀림"]+=1
    tot=sum(v for k,v in cat.items() if not k.startswith("PASS"))
    print("=== 부하 정체 규명: liability 오답 분해 (n_fail=%d) ==="%tot)
    for k,v in cat.most_common():
        if k.startswith("PASS"):continue
        print("  %-40s %5d (%.1f%%)"%(k,v,100*v/max(tot,1)))
    print("\n해석:")
    print("  (a)GATHER 지배 → 부하=입력수집 오류(loop 격리 무효·FIND/GET로 정확수집 필요)")
    print("  (b)COMPUTE 지배 → 부하=계산 자체(loop 격리/결정론COMPUTE로 닫힘)")
    for k,v in ex.items():print("  예시[%s]:"%k,v)

if __name__=="__main__":main()
