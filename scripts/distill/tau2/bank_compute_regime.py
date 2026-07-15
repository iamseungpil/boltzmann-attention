# -*- coding: utf-8 -*-
"""E-REGIME compute 버킷 (트랙2·verify가 voting 이기는 곳 실증·2026-07-15).
id-correct dispute의 liability(customer_max_liability_amount)를:
  - 결정론 verify = keystone lookup_table(days=discovery-transaction: ≤30→50·≤60→500·else→amount) → gold 도달?
  - resample(k회·T>0) = 모델이 facts→liability 계산 → voting(maj@k)이 gold 도달? (systematic이면 실패)
= "decidable-systematic → verify가 voting 이긴다"(C89 하위유형②) 실증.
로컬(--dry: verify 천장·서버불요) / full(resample: localhost:8140 32B).
"""
import json,glob,re,argparse,urllib.request,math,sys,io
from datetime import datetime
from collections import Counter
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
def nd(x):
    if isinstance(x,str):
        try:x=json.loads(x)
        except:return {}
    return x if isinstance(x,dict) else {}
fam=lambda n:re.sub(r"_\d+$","",str(n))
MODEL="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
FACTS=["dispute_category","transaction_date","discovery_date","disputed_amount","transaction_type",
       "card_in_possession","pin_compromised","contacted_merchant","police_report_filed","written_statement_provided"]

def parse_date(s):
    for f in ("%m/%d/%Y","%m/%d/%y","%m/%d"):
        try:return datetime.strptime(str(s).strip()[:10],f)
        except:pass
    return None

def liability_lookup(ga):
    """keystone flat lookup: days=disc-tx ≤30→50·≤60→500·else→disputed_amount."""
    td=parse_date(ga.get("transaction_date")); dd=parse_date(ga.get("discovery_date"))
    amt=ga.get("disputed_amount")
    try:amt=round(abs(float(amt)),2)
    except:amt=None
    if td and dd:
        days=(dd-td).days
        if days<=30:return 50.0
        if days<=60:return 500.0
        return amt
    return None

def norm(v):
    try:return round(abs(float(re.sub(r"[$,]","",str(v)))),2)
    except:return str(v).strip().lower()

def extract():
    cases=[]
    for f in glob.glob("C:/tmp/traj/*_banking.json"):
        model=f.replace("\\","/").split("/")[-1].replace("_banking.json","")
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
                if tid and tid in acalls and ga.get("customer_max_liability_amount") is not None:
                    cases.append({"task":s.get("task_id"),"model":model,"tid":tid,
                                  "gold_liab":ga.get("customer_max_liability_amount"),
                                  "agent_liab":acalls[tid].get("customer_max_liability_amount"),
                                  "facts":{k:ga.get(k) for k in FACTS}})
    return cases

def prompt(facts):
    pol=("Bank dispute liability policy: customer_max_liability_amount is $50 if the dispute is reported within 30 days "
         "of the transaction; $500 if reported within 31-60 days; otherwise the full disputed amount. "
         "Report timing = days between transaction_date and discovery_date.")
    return (pol+"\nDispute facts:\n"+json.dumps(facts,ensure_ascii=False)+
            "\nReturn ONLY the customer_max_liability_amount as a number (dollars).")

def call(port,pr,temp,n,timeout=90):
    body={"model":MODEL,"messages":[{"role":"user","content":pr}],"temperature":temp,"max_tokens":40}
    if n>1:body["n"]=n
    req=urllib.request.Request("http://localhost:%d/v1/chat/completions"%port,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=timeout) as r:d=json.load(r)
    return [c["message"]["content"] for c in d["choices"]]

def num(txt):
    m=re.search(r"-?\$?[\d,]+\.?\d*",str(txt or ""))
    if not m:return None
    try:return round(abs(float(m.group(0).replace("$","").replace(",",""))),2)
    except:return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--port",type=int,default=8140); ap.add_argument("--k",type=int,default=8)
    ap.add_argument("--temperature",type=float,default=0.7); ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--dry",action="store_true"); ap.add_argument("--workers",type=int,default=12)
    a=ap.parse_args()
    cases=extract()
    if a.limit:cases=cases[:a.limit]
    # 결정론 verify 천장 (offline)
    vok=vtot=0; aok=0
    for c in cases:
        gl=norm(c["gold_liab"]); vl=liability_lookup(c["facts"])
        if vl is not None:
            vtot+=1
            if norm(vl)==gl:vok+=1
        if norm(c["agent_liab"])==gl:aok+=1
    print("compute cases(id-correct·liability有): %d"%len(cases))
    print("★결정론 verify(keystone lookup) gold도달: %d/%d = %.1f%% (=verify 천장)"%(vok,vtot,100*vok/max(vtot,1)))
    print("  agent 실제 liability 정확: %d/%d = %.1f%% (C81 51%%오답 정합)"%(aok,len(cases),100*aok/max(len(cases),1)))
    if a.dry:
        print("[--dry] verify 천장만. resample voting=서버 필요.")
        return
    # resample (server)
    from concurrent.futures import ThreadPoolExecutor
    def proc(c):
        gl=norm(c["gold_liab"])
        try:
            g=num(call(a.port,prompt(c["facts"]),0.0,1)[0])
            raws=call(a.port,prompt(c["facts"]),a.temperature,a.k)
        except Exception as e:return {"err":str(e)[:80]}
        vals=[num(r) for r in raws]; valid=[v for v in vals if v is not None]
        if len(valid)<5:return {"meas":False}
        cnt=Counter(valid); maj,_=cnt.most_common(1)[0]
        H=-sum((v/len(valid))*math.log2(v/len(valid)) for v in cnt.values())
        return {"meas":True,"gl":gl,"greedy_ok":g==gl,"maj_ok":maj==gl,
                "gold_in_support":gl in cnt,"H_k":round(H,2),"verify_ok":norm(liability_lookup(c["facts"]))==gl}
    res=[]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(proc,cases):res.append(r)
    m=[r for r in res if r.get("meas")]
    gw=[r for r in m if not r["greedy_ok"]]
    C=sum(1 for r in gw if r["maj_ok"]); D=len(gw)-C
    vwin=sum(1 for r in gw if r["verify_ok"])
    print("\n=== compute resample (n=%d meas·k=%d) ==="%(len(m),a.k))
    print("greedy-wrong %d: voting-win(maj_ok) %d = %.1f%% · gold∈support %d"%(len(gw),C,100*C/max(len(gw),1),sum(1 for r in gw if r["gold_in_support"])))
    print("★verify(결정론 lookup)가 greedy-wrong 구제: %d/%d = %.1f%%"%(vwin,len(gw),100*vwin/max(len(gw),1)))
    print("⇒ voting %.1f%% vs verify %.1f%% (verify가 이기면 decidable-systematic 실증)"%(100*C/max(len(gw),1),100*vwin/max(len(gw),1)))

if __name__=="__main__":main()
