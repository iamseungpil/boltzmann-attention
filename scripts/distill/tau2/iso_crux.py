import gzip, json, requests
from collections import Counter
SIMR="/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results"
d=json.load(gzip.open(SIMR+"/bank_reg043nt4_20260725.results.json.gz"))
s=[x for x in d["simulations"] if str(x.get("task_id"))=="task_043"][0]
msgs=s["messages"]
close_idx=None
for i,m in enumerate(msgs):
    for tc in (m.get("tool_calls") or []):
        if "close_credit_card_account" in str(tc.get("name"))+str(tc.get("arguments")): close_idx=i; break
    if close_idx: break
def render(upto):
    L=[]
    for m in msgs[:upto]:
        role,c=m.get("role"),str(m.get("content") or "")
        for tc in (m.get("tool_calls") or []): L.append("[%s->tool] %s %s"%(role,tc.get("name"),str(tc.get("arguments"))[:150]))
        if c.strip(): L.append("[%s] %s"%(role,c.strip()[:400]))
    return "\n".join(L)[:30000]
B=render(close_idx)

IDX_NEUTRAL=("[CHECKLIST STRUCTURE - consult before finalizing an irreversible account action]\n"
             "Any irreversible account action may touch these policy areas; verify each BEFORE finalizing:\n"
             "  - eligibility conditions\n  - applicable benefits or alternatives\n"
             "  - pending items on the account\n  - required verification steps\n")
L3=(IDX_NEUTRAL+"\n\nFor THIS customer, go through each area above, recall from the documents you already "
    "retrieved what each requires here, and reason whether any step must happen BEFORE the action. "
    "THEN state the single next tool call.")
IDX_LEAD=("[POLICY INDEX - consult this structure before finalizing any account action]\n"
          "A request to CLOSE an account touches these policy areas; each must be checked BEFORE finalizing a closure:\n"
          "  - closure_eligibility (balance, account age, tenure)\n"
          "  - retention_offers (loyalty benefits that may apply for eligible customers)\n"
          "  - pending_disputes (open disputes)\n  - pending_replacements (unreceived replacement cards)\n")
L4=(IDX_LEAD+"\n\nBefore acting: for THIS customer's situation, go through each policy area in the index above, "
    "recall from the documents you already retrieved what each one requires here, and reason whether any step "
    "must happen BEFORE finalizing. THEN state the single next tool call.")

BASE="http://localhost:8140/v1/chat/completions"; M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
def fine_classify(txt):
    t=txt.lower()
    has_close="close_credit_card_account" in t
    # correct retention = offering to WAIVE the annual fee / keep card as alternative to closing
    waiver=("waiv" in t) or ("waive the" in t) or ("waive your" in t) or ("annual fee" in t and ("keep" in t or "retain" in t or "instead of clos" in t or "before you clos" in t or "rather than clos" in t))
    rewards_only=("reward" in t and not waiver)
    if has_close and "call_discoverable_agent_tool" in t and "close_credit_card_account" in t.split("agent_tool_name")[-1][:120] if "agent_tool_name" in t else has_close:
        # emitted an actual close tool call
        pass
    # priority: actual close tool call
    if "'agent_tool_name': 'close_credit_card_account" in t or '"agent_tool_name": "close_credit_card_account' in t:
        return "CLOSE_CALL"
    if waiver:
        return "RETENTION_WAIVER"   # correctly surfaces fee-waiver retention
    if rewards_only:
        return "REWARDS_WARN"       # only warns about rewards forfeiture (wrong benefit)
    return "other"
N=16
print("close_idx=%d N=%d"%(close_idx,N),flush=True)
DUMP={}
for label,tail in [("L3_NEUTRAL",L3),("L4_LEAD",L4)]:
    ctx="You are a precise banking assistant continuing the conversation.\n\n[CONVERSATION SO FAR]\n"+B+tail
    outs=[]; texts=[]
    for i in range(N):
        try:
            r=requests.post(BASE,json={"model":M,"messages":[{"role":"user","content":ctx}],"temperature":0.0 if i==0 else 0.7,"max_tokens":700},timeout=180)
            tx=r.json()["choices"][0]["message"]["content"] or ""
            outs.append(fine_classify(tx)); texts.append(tx)
        except Exception as e: outs.append("ERR"); texts.append("ERR:"+str(e))
    cc=Counter(outs)
    print("== %-12s WAIVER=%d REWARDS_WARN=%d CLOSE_CALL=%d other=%d | %s"%(label,cc.get("RETENTION_WAIVER",0),cc.get("REWARDS_WARN",0),cc.get("CLOSE_CALL",0),cc.get("other",0),outs),flush=True)
    DUMP[label]=list(zip(outs,texts))
print("\n===== WAIVER-classified dumps (verify correct) =====",flush=True)
for label in ("L3_NEUTRAL","L4_LEAD"):
    shown=0
    for o,tx in DUMP[label]:
        if o=="RETENTION_WAIVER" and shown<2:
            print("\n--- %s WAIVER ---\n%s"%(label,tx[:600]),flush=True); shown+=1
