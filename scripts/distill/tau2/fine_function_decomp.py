#!/usr/bin/env python3
"""도메인-불가지 fine-grained 실패 기능 분해 (retail/airline/telecom/banking 공통).
정본 doc TAU2_FRONTIER_TRAJECTORY_INVESTIGATION_MASTER §3.2d.

각 db_fail을 leaf 서브기능으로 분해하고 구제방법(remedy)에 매핑:
- 액션 기능클래스: READ / VERIFY / REACH / ESCALATE / MUTATE (도메인-일반 패턴)
- MUTATE 발산 arg 타입: REFERENCE / NUMERIC / VARIANT / CATEGORICAL / FREETEXT
- 메타: FABRICATE(값이 문맥에 부재) / GATHER(write 전 read) / VERIFY / HORIZON(gold 절차 길이)
기준 = reward_info.db_check.db_match. 로컬 CPU-only."""
import json, io, itertools, re, sys
from collections import Counter

# ---- 도메인-일반 액션 기능클래스 (이름 패턴) ----
def aclass(name):
    n = (name or "").lower()
    if any(k in n for k in ("kb_search","search_","get_","list_","view_","lookup","fetch","_by_","read_","find_")): return "READ"
    if any(k in n for k in ("verif","authenticate","identity","log_verification")): return "VERIFY"
    if any(k in n for k in ("discover","unlock")): return "REACH"
    if any(k in n for k in ("transfer","human","escalate")): return "ESCALATE"
    if any(k in n for k in ("apply","submit","modify","cancel","return_","exchange","book","update","change","place_","send","create","delete","pay_","give_","call_discoverable","request_")): return "MUTATE"
    return "OTHER"

# ---- arg 타입 분류 (도메인-일반) ----
REF = re.compile(r"(_id$|_ids$|order|account|reservation|flight|tool_name|user\b|referral|card_id)", re.I)
NUM = re.compile(r"(amount|income|price|quantity|count|total|limit|balance|number|fee|payment_amount)", re.I)
VAR = re.compile(r"(new_item_ids|item_ids|cabin|card_type|insurance|membership|baggage|seat|plan|variant|option)", re.I)
CAT = re.compile(r"(reason|status|subscription|type|category|flag)", re.I)
TXT = re.compile(r"(address|name|email|phone|summary|message|text|note|description|reason_text)", re.I)
def argtype(k, v):
    if VAR.search(k): return "VARIANT"       # 카탈로그 옵션 선택
    if REF.search(k): return "REFERENCE"      # 어느 엔티티
    if NUM.search(k) or isinstance(v,(int,float)): return "NUMERIC"
    if TXT.search(k): return "FREETEXT"
    if CAT.search(k): return "CATEGORICAL"
    return "OTHER:"+k

def ri(s): return s.get("reward_info") or {}
def dbm(s): return (ri(s).get("db_check") or {}).get("db_match")
def rew(s): return ri(s).get("reward")
def is_fail(s): return rew(s) not in (1, 1.0, True)  # 도메인-불가지 통일 기준
def rbasis(s): return tuple(ri(s).get("reward_basis") or [])
AUTH = re.compile(r"(find_user_id|get_user_details|get_customer|authenticate|verify_identity|identify|get_reservation_details|log_verification)", re.I)
FIXACT = re.compile(r"(toggle|reboot|reseat|grant_|set_network|enable_|disable_|refuel|reset_|apply_)", re.I)

def _exec_names_ordered(s):
    by={m.get("id") or m.get("tool_call_id"):m for m in s.get("messages",[]) if m.get("role")=="tool"}
    seq=[]
    for m in s.get("messages",[]):
        if m.get("role")!="assistant": continue
        for tc in m.get("tool_calls") or []:
            tm=by.get(tc.get("id"))
            seq.append((tc.get("name"), not (tm is not None and tm.get("error"))))
    return seq

def telecom_leaves(s):
    """env-assertion 도메인(telecom·dual-control): 결함 미해결(COVERAGE) + persistence(조기 escalate) + guidance(fix 시도했으나 실패)."""
    out=Counter()
    unfixed=sum(1 for a in (ri(s).get("env_assertions") or []) if a.get("met") is False)
    for _ in range(unfixed): out["COVERAGE_fault_unfixed(F4)"]+=1
    seq=_exec_names_ordered(s)
    escalated=any(("transfer" in (n or "").lower() or "human" in (n or "").lower()) for n,_ in seq)
    fixed_attempts=sum(1 for n,ok in seq if FIXACT.search(n or "") and ok)
    if unfixed>0 and escalated: out["PERSISTENCE_escalate_unfixed(F5)"]+=1   # 결함 남은 채 조기 사람연결
    if unfixed>0 and fixed_attempts>0 and not escalated: out["GUIDANCE_fix_ineffective(dual)"]+=1  # 조치했으나 env 미달=dual-control 실행/유도 실패
    if unfixed>0 and fixed_attempts==0 and not escalated: out["REACH_no_fix_attempted"]+=1  # 진단/조치 자체 안함
    return out
def gold_acts(s): return [(a.get("action",{}).get("name"), a.get("action",{}).get("arguments") or {}) for a in (ri(s).get("action_checks") or [])]
def exec_acts(s):
    by={m.get("id") or m.get("tool_call_id"):m for m in s.get("messages",[]) if m.get("role")=="tool"}
    ok=[]; err=[]
    for m in s.get("messages",[]):
        if m.get("role")!="assistant": continue
        for tc in m.get("tool_calls") or []:
            tm=by.get(tc.get("id"))
            (err if (tm is not None and tm.get("error")) else ok).append((tc.get("name"), tc.get("arguments") or {}))
    return ok,err
def toolctx(s): return " ".join(m.get("content") for m in s.get("messages",[]) if m.get("role")=="tool" and isinstance(m.get("content"),str))
def userctx(s): return " ".join(m.get("content") for m in s.get("messages",[]) if m.get("role")=="user" and isinstance(m.get("content"),str))
def _n(v): return sorted(str(x) for x in v) if isinstance(v,list) else str(v)
def dkeys(g,o): return [k for k in (set(g)|set(o))-{"user_id"} if _n(g.get(k))!=_n(o.get(k))]
def cost(g,o): return 100 if g[0]!=o[0] else len(dkeys(g[1],o[1]))
def pair(gm,om):
    if not gm or not om: return []
    best=None
    for pm in itertools.permutations(range(len(om))):
        c=sum(cost(gm[i],om[pm[i]]) for i in range(min(len(gm),len(om))))
        if best is None or c<best[0]: best=(c,pm)
    return best[1]

def decomp(path,label):
    d=json.load(io.open(path,encoding="utf-8")); sims=d.get("simulations") or d
    passrate=sum(1 for s in sims if not is_fail(s))/max(len(sims),1)
    fails=[s for s in sims if is_fail(s)]
    env_dom = any("ENV_ASSERTION" in rbasis(s) for s in sims[:20])
    leaf=Counter(); horizons=[]
    if env_dom:  # telecom: env-assertion 기반
        for s in fails: leaf.update(telecom_leaves(s)); horizons.append(len(ri(s).get("action_checks") or []))
        import statistics
        nf=len(fails)
        print(f"\n===== {label} [ENV_ASSERTION dom] | n={len(sims)} pass={passrate:.3f} fail={nf} =====")
        for k,v in leaf.most_common(): print(f"   {v:4d} ({100*v/max(nf,1):4.1f}% of fails·건수) {k}")
        return label,leaf,nf
    for s in fails:
        G=gold_acts(s); ok,err=exec_acts(s)
        gcl=Counter(aclass(n) for n,_ in G); ecl=Counter(aclass(n) for n,_ in ok)
        horizons.append(len(G))
        ctx=toolctx(s)+" "+userctx(s)
        # VERIFY(F1) capture: write가 인증/신원확인 前 발생 (게이트가 아니라 궤적서 탐지)
        seq=_exec_names_ordered(s)
        first_mut=next((i for i,(n,o) in enumerate(seq) if o and aclass(n)=="MUTATE"), None)
        if first_mut is not None:
            auth_before=any(o and AUTH.search(n or "") for n,o in seq[:first_mut])
            if not auth_before: leaf["MISS_verify(F1·write前 인증無)"]+=1
        # 1) 누락 기능클래스 (coverage/reach/verify/escalate)
        for cl in ("VERIFY","REACH","MUTATE","ESCALATE","READ"):
            if gcl[cl]>ecl[cl]:
                leaf[{"VERIFY":"MISS_verify(F1)","REACH":"MISS_reach/procedure","MUTATE":"MISS_write(coverage F4)","ESCALATE":"MISS_escalate(F5)","READ":"MISS_gather"}[cl]]+=1
        # 2) 과잉
        if ecl["MUTATE"]>gcl["MUTATE"]: leaf["EXTRA_write(over-action)"]+=1
        if ecl["ESCALATE"]>gcl["ESCALATE"]: leaf["EXTRA_escalate"]+=1
        # 3) MUTATE 내용 오류 (arg 타입별)
        gm=[(n,a) for n,a in G if aclass(n)=="MUTATE"]; om=[(n,a) for n,a in ok if aclass(n)=="MUTATE"]
        if gm and om and len(gm)==len(om):
            pm=pair(gm,om)
            for i in range(len(gm)):
                gn,ga=gm[i]; on,oa=om[pm[i]]
                if gn!=on: leaf["WRONG_op(연산자)"]+=1; continue
                for k in dkeys(ga,oa):
                    t=argtype(k,oa.get(k))
                    # fabrication 체크: exec 값이 문맥에 실재?
                    vals=oa.get(k); vals=vals if isinstance(vals,list) else [vals]
                    infab = any(str(x) and str(x) not in ctx for x in vals)
                    tag=f"ARG_{t}"
                    if t in ("REFERENCE","FREETEXT") and infab: tag=f"ARG_{t}_FABRICATED"
                    leaf[tag]+=1
    import statistics
    n=len(sims); nf=len(fails)
    print(f"\n===== {label} | n={n} pass={passrate:.3f} fail={nf} horizon(gold-act) med={statistics.median(horizons) if horizons else 0} =====")
    for k,v in leaf.most_common(): print(f"   {v:4d} ({100*v/max(nf,1):4.1f}%) {k}")
    return label,leaf,nf

if __name__=="__main__":
    for p in sys.argv[1:]:
        lbl=p.split("/")[-1].replace(".json","")
        try: decomp(p,lbl)
        except Exception as e: print(p,"ERR",e)
