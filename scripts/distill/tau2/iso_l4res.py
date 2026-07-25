import gzip, json, requests
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
IDX_LEAD=("[POLICY INDEX - consult this structure before finalizing any account action]\n"
          "A request to CLOSE an account touches these policy areas; each must be checked BEFORE finalizing a closure:\n"
          "  - closure_eligibility (balance, account age, tenure)\n"
          "  - retention_offers (loyalty benefits that may apply for eligible customers)\n"
          "  - pending_disputes (open disputes)\n  - pending_replacements (unreceived replacement cards)\n")
L4=(IDX_LEAD+"\n\nBefore acting: for THIS customer's situation, go through each policy area in the index above, "
    "recall from the documents you already retrieved what each one requires here, and reason whether any step "
    "must happen BEFORE finalizing. THEN state the single next tool call.")
BASE="http://localhost:8140/v1/chat/completions"; M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
N=16
ctx="You are a precise banking assistant continuing the conversation.\n\n[CONVERSATION SO FAR]\n"+B+L4
print("close_idx=%d N=%d  === FULL DUMPS FOR FORENSIC READING ==="%(close_idx,N),flush=True)
for i in range(N):
    try:
        r=requests.post(BASE,json={"model":M,"messages":[{"role":"user","content":ctx}],"temperature":0.0 if i==0 else 0.7,"max_tokens":900},timeout=200)
        tx=r.json()["choices"][0]["message"]["content"] or ""
    except Exception as e: tx="ERR:"+str(e)
    print("\n########## SAMPLE %d ##########\n%s"%(i,tx),flush=True)
