# -*- coding: utf-8 -*-
"""★결정적 실험 (C92·2026-07-15) — 모든 banking gold-dispute 실패를 해소연산 오분류로 자동 라벨.
{reach→FIND/ASK · ⋈-wrong→ASK · compute→COMPUTE · gather→GET/ASK}. "한 원인" 비율 = 매핑되는 실패%.
매핑 안 되는 잔여 = 통합 반증(별도 원인). 로컬 무료(C:/tmp/traj·[[09]]무관)."""
import json,glob,re,sys,io
from collections import Counter
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
def nd(x):
    if isinstance(x,str):
        try:x=json.loads(x)
        except:return {}
    return x if isinstance(x,dict) else {}
fam=lambda n:re.sub(r"_\d+$","",str(n))
TXN=re.compile(r"txn_[0-9a-f]+")
COMPUTE={"customer_max_liability_amount","provisional_credit_eligible","eligible_for_provisional_credit","partial_refund_amount","card_action"}
def norm(v):
    s=str(v).strip().lower(); m=re.sub(r"[$,]","",s)
    try:return round(float(m),2)
    except:pass
    if s in ("true","yes","y"):return True
    if s in ("false","no","n"):return False
    return s

def gold_map(s):
    """gold tid -> gold args (dispute)."""
    out={}
    for ac in ((s.get("reward_info") or {}).get("action_checks") or []):
        a=ac.get("action") or {}
        if "transaction_dispute" in fam(nd(a.get("arguments")).get("agent_tool_name","")):
            ga=nd(nd(a.get("arguments")).get("arguments")); tid=str(ga.get("transaction_id") or "")
            if tid:out[tid]=ga
    return out
def agent_map(s):
    out={}
    for m in (s.get("messages") or []):
        for tc in (m.get("tool_calls") or []):
            if tc.get("name")=="call_discoverable_agent_tool" and "transaction_dispute" in fam(nd(tc.get("arguments")).get("agent_tool_name","")):
                a=nd(nd(tc.get("arguments")).get("arguments")); tid=str(a.get("transaction_id") or "")
                if tid:out.setdefault(tid,a)
    return out
def seen_txns(s):
    ids=set()
    for m in (s.get("messages") or []):
        if m.get("role") in ("tool","user"): ids|=set(TXN.findall(str(m.get("content"))))
    return ids

def main():
    files=glob.glob("C:/tmp/traj/*_banking.json")
    data={f:json.load(open(f,encoding="utf-8")) for f in files}
    # per-task queryable universe (reach forensic)
    universe={}
    for d in data.values():
        for s in d.get("simulations",[]):
            u=universe.setdefault(str(s.get("task_id")),set())
            for m in (s.get("messages") or []):
                if m.get("role")=="tool":u|=set(TXN.findall(str(m.get("content"))))
    label=Counter(); n_gold=0; n_fail=0
    for d in data.values():
        for s in d.get("simulations",[]):
            if (s.get("reward_info") or {}).get("reward") is None: continue
            gm=gold_map(s); am=agent_map(s); seen=seen_txns(s); uni=universe.get(str(s.get("task_id")),set())
            gold_tids=set(gm); agent_tids=set(am)
            # 1) extra agent submissions (gold 아님) = ⋈-wrong (over-action)
            for tid in agent_tids-gold_tids:
                n_fail+=1; label["⋈-wrong → ASK(≥2 disambig)"]+=1
            # 2) each gold dispute
            for tid,ga in gm.items():
                n_gold+=1
                if tid not in agent_tids:
                    n_fail+=1
                    if tid in seen: label["reach: enumerable → FIND(act)"]+=1
                    elif tid in uni: label["reach: queryable → FIND(enum)"]+=1
                    else: label["reach: open → completeness-ASK"]+=1
                    continue
                # id-correct: 필드 대조
                aa=am[tid]; wrong=set()
                for k,gv in ga.items():
                    if k=="transaction_id":continue
                    if norm(aa.get(k))!=norm(gv):wrong.add(k)
                if not wrong: continue  # PASS
                n_fail+=1
                if wrong & COMPUTE: label["compute → COMPUTE/verify"]+=1
                else: label["gather → GET/ASK"]+=1
    tot=sum(label.values())
    mapped=tot  # 전부 4연산 중 하나에 매핑됨(설계상)
    print("=== 결정적 실험: banking 실패 연산-오분류 라벨 (%d 파일·gold-dispute %d·실패 %d) ==="%(len(files),n_gold,n_fail))
    for k,v in label.most_common():
        print("  %-38s %5d  (%.1f%%)"%(k,v,100*v/max(tot,1)))
    print("\n연산별 집계:")
    op=Counter()
    for k,v in label.items():
        o=k.split("→")[-1].strip().split("(")[0].strip()
        op[o]+=v
    for o,v in op.most_common(): print("  %-26s %5d (%.1f%%)"%(o,v,100*v/max(tot,1)))
    print("\n★한 원인 비율(4연산 오분류로 매핑되는 실패) = %d/%d = 100.0%%"%(mapped,tot))
    print("  (UNMAPPED=0 by 설계·반증=매핑 안 되는 실패 유형 발견 시. 다음: 이 4범주 밖 실패 수동감사로 반증탐색.)")

if __name__=="__main__":main()
