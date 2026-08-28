# -*- coding: utf-8 -*-
r"""x397 — G1-a: B_tail4 answerability 감사(결정론·LLM 0). gold 는 계측에만 쓴다(프롬프트 투입 없음)."""
import io, json, os, re, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import x395_compliance_iso as X
import t2_forensic as F

docs = X.load_docs()
RES = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/x395_compliance_iso.json"
rows_res = json.load(io.open(RES, encoding="utf-8"))

def targets(cases=12):
    out=[]; seen=set()
    for tag in X.TAGS:
        for sim in F.scored(tag, X.SUF):
            rw=(sim.get("reward_info") or {}).get("reward")
            if (rw or 0)>=1.0: continue
            cn=X.called_names(sim)
            for ck in ((sim.get("reward_info") or {}).get("action_checks") or []):
                if ck.get("action_match"): continue
                aa=ck.get("action") or {}; ar=aa.get("arguments") or {}
                nm=str(ar.get("agent_tool_name") or ar.get("user_tool_name")
                       or ar.get("discoverable_tool_name") or aa.get("name") or "")
                if not nm or cn.get(nm): continue
                pl=X.proc_lines(docs,nm)
                if not pl: continue
                body=" ".join(" ".join(str(m.get("content") or "").split())
                              for m in (sim.get("messages") or []) if m.get("role")=="tool")
                if not [s for s in pl if s.split("] ",1)[-1][:55] in body]: continue
                k=(F.task_id(sim),nm)
                if k in seen: continue
                seen.add(k)
                out.append({"task":F.task_id(sim),"trial":sim.get("trial"),"tool":nm,
                            "lines":pl,"sim":sim})
                if len(out)>=cases: return out
    return out

def alt_argsets(sim, tool):
    """그 도구를 **실제로 부르는** gold 체크들의 인자 조합 목록(대안 전부)."""
    alts=[]
    for ck in ((sim.get("reward_info") or {}).get("action_checks") or []):
        aa=ck.get("action") or {}; ar=aa.get("arguments") or {}
        nm=str(ar.get("agent_tool_name") or ar.get("discoverable_tool_name") or "")
        if nm!=tool: continue
        raw=ar.get("arguments")
        if not isinstance(raw,str): continue
        try: d=json.loads(raw)
        except Exception: continue
        vals=[str(v) for v in d.values() if isinstance(v,str) and len(str(v))>=4]
        alts.append({"argdict":d,"strvals":vals})
    return alts

print("표적 12건 · B_tail4 answerability 감사\n")
tab=[]
for c in targets():
    sim=c["sim"]; tool=c["tool"]
    msgs=sim.get("messages") or []
    t4=X.convo(sim,tail=4)
    calls_,ents=X.ledger_of(sim); ask=X.user_ask(sim)
    amin=("# 손님 요청\n%s\n\n# 지금까지의 진행(원장)\n호출한 도구: %s\n레코드 id: %s"
          %(ask,", ".join(calls_[:25]),", ".join(ents[:25])))
    bfull=X.convo(sim)
    proc="\n".join(c["lines"])
    alts=alt_argsets(sim,tool)
    def cov(hay,alt):
        if not alt["strvals"]: return None
        got=[v for v in alt["strvals"] if v.lower() in hay.lower()]
        return (len(got),len(alt["strvals"]),got)
    best_t4=best_am=best_bf=None
    for a in alts:
        for hay,slot in ((t4,"t4"),(amin,"am"),(bfull,"bf")):
            r=cov(hay,a)
            if r is None: continue
            cur={"t4":best_t4,"am":best_am,"bf":best_bf}[slot]
            frac=r[0]/float(r[1])
            if cur is None or frac>cur[0]/float(cur[1]):
                if slot=="t4": best_t4=r
                elif slot=="am": best_am=r
                else: best_bf=r
    tailmsgs=msgs[-4:]
    row={"task":c["task"],"trial":c["trial"],"tool":tool,"n_msgs":len(msgs),
         "roles":[m.get("role") for m in tailmsgs],
         "tool_in_proc":tool in proc,"tool_in_t4":tool.lower() in t4.lower(),
         "ask_in_t4":(ask[:60].lower() in t4.lower()) if ask else False,
         "user_msg_in_t4":any(m.get("role")=="user" and m.get("content") for m in tailmsgs),
         "n_alts":len(alts),
         "argcov_t4":best_t4,"argcov_amin":best_am,"argcov_bfull":best_bf,
         "t4":t4}
    tab.append(row)
    print("=== %-9s t%s %-40s (msgs=%d, tail roles=%s)"%(row["task"],row["trial"],tool,row["n_msgs"],row["roles"]))
    print("    도구이름: 절차문장에 %s / tail4 본문에 %s | tail4 에 user 메시지 %s / 첫 손님요청문 %s"
          %(row["tool_in_proc"],row["tool_in_t4"],row["user_msg_in_t4"],row["ask_in_t4"]))
    for i,a in enumerate(alts):
        print("    gold 인자셋#%d %s"%(i,json.dumps(a["argdict"],ensure_ascii=False)[:160]))
    def fmt(x): return "-" if x is None else "%d/%d %s"%(x[0],x[1],x[2])
    print("    인자값 커버리지  tail4=%s | A_min=%s | B_full=%s"%(fmt(best_t4),fmt(best_am),fmt(best_bf)))
    print("    --- tail4 축자 ---")
    for ln in t4.split("\n"): print("      "+ln[:300])
    print()

io.open("/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/x397_tail4_answerability.json","w",encoding="utf-8").write(json.dumps(tab,ensure_ascii=False,indent=1))
print("저장: x397_tail4_answerability.json")
