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

# ===== PURIFICATION LADDER: domain-leading 0 -> max =====
# L0 control
NEUTRAL="\n\nWhat is the single next tool call you should make right now? Name the exact tool."
# L1 PURE metacognitive - ZERO domain terms
L1_PURE=("\n\nBefore making any tool call, stop and reason: Is the action you are about to take reversible? "
         "Does the situation require any prior step to be completed first? Think it through, "
         "then state the single next tool call.")
# L2 generic prerequisite framing - no domain terms
L2_PREREQ=("\n\nBefore finalizing any irreversible action, confirm that every prerequisite the applicable "
           "procedures require has been completed. Reason about what applies here given what you know, "
           "then state the single next tool call.")
# L3 NEUTRAL structural index (domain-general category names, NOT 'retention_offers') + recall reasoning
IDX_NEUTRAL=("[CHECKLIST STRUCTURE - consult before finalizing an irreversible account action]\n"
             "Any irreversible account action may touch these policy areas; verify each BEFORE finalizing:\n"
             "  - eligibility conditions\n"
             "  - applicable benefits or alternatives\n"
             "  - pending items on the account\n"
             "  - required verification steps\n")
L3_IDX_NEUTRAL=(IDX_NEUTRAL+"\n\nFor THIS customer, go through each area above, recall from the documents you already "
                "retrieved what each requires here, and reason whether any step must happen BEFORE the action. "
                "THEN state the single next tool call.")
# L4 LEADING index (C157 original: retention_offers named) - upper bound reference
IDX_LEAD=("[POLICY INDEX - consult this structure before finalizing any account action]\n"
          "A request to CLOSE an account touches these policy areas; each must be checked BEFORE finalizing a closure:\n"
          "  - closure_eligibility (balance, account age, tenure)\n"
          "  - retention_offers (loyalty benefits that may apply for eligible customers)\n"
          "  - pending_disputes (open disputes)\n"
          "  - pending_replacements (unreceived replacement cards)\n")
L4_IDX_LEAD=(IDX_LEAD+"\n\nBefore acting: for THIS customer's situation, go through each policy area in the index above, "
             "recall from the documents you already retrieved what each one requires here, and reason whether any step "
             "must happen BEFORE finalizing. THEN state the single next tool call.")

BASE="http://localhost:8140/v1/chat/completions"; M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
def classify(txt):
    t=txt.lower()
    has_apply=("apply_credit_card_account_flag" in t) or ("annual_fee_waived" in t)
    has_close="close_credit_card_account" in t
    # decide by the FINAL/dominant next action language
    if has_close and not has_apply: return "CLOSE"
    if has_apply and not has_close: return "RETAIN"
    if has_apply and has_close: return "MIXED"
    if "waiv" in t and not has_close: return "RETAIN"     # proposes waiver, no close
    return "other"                                          # asks customer / reads more / other
N=12
conds=[("L0_NEUTRAL",NEUTRAL),("L1_PURE",L1_PURE),("L2_PREREQ",L2_PREREQ),
       ("L3_IDX_NEUTRAL",L3_IDX_NEUTRAL),("L4_IDX_LEAD",L4_IDX_LEAD)]
print("close_idx=%d Bctx=%d N=%d"%(close_idx,len(B),N),flush=True)
DUMP={}
for label,tail in conds:
    ctx="You are a precise banking assistant continuing the conversation.\n\n[CONVERSATION SO FAR]\n"+B+tail
    outs=[]; texts=[]
    for i in range(N):
        try:
            r=requests.post(BASE,json={"model":M,"messages":[{"role":"user","content":ctx}],"temperature":0.0 if i==0 else 0.7,"max_tokens":600},timeout=180)
            tx=r.json()["choices"][0]["message"]["content"] or ""
            outs.append(classify(tx)); texts.append(tx)
        except Exception as e: outs.append("ERR"); texts.append("ERR:"+str(e))
    from collections import Counter
    cc=Counter(outs)
    print("== %-16s CLOSE=%d RETAIN=%d other=%d MIXED=%d | %s"%(label,cc.get("CLOSE",0),cc.get("RETAIN",0),cc.get("other",0),cc.get("MIXED",0),outs),flush=True)
    DUMP[label]=list(zip(outs,texts))
# qualitative dump: 2 samples per condition (verify not wrong-parrot)
print("\n===== RAW DUMPS (2/cond) =====",flush=True)
for label,_ in conds:
    for j,(o,tx) in enumerate(DUMP[label][:2]):
        print("\n--- %s [%d] class=%s ---\n%s"%(label,j,o,tx[:700]),flush=True)
