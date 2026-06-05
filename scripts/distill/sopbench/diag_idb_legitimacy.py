#!/usr/bin/env python3
"""(A) When released models called internal_get_database, was it in the OFFERED tools (prompt),
    i.e. an intended tool, or a hallucinated call? Check the interaction prompt tool list.
(B) How do cred-absent transfers pass in OUR loginfirst run (no augment)? Dump call traces."""
import json, hashlib, glob

def H(x): return hashlib.md5(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()[:16]
def ident(t): return (t.get("user_goal"),H(t.get("constraints_original")),H(t.get("constraint_parameters")),H(t.get("initial_database")))

print("=== (A) released: internal_get_database OFFERED vs hallucinated ===")
for fp in sorted(glob.glob("/home/woori/scratch/SOPBench/output/bank/ast_*llama3.1-70b*tool_full*.json")):
    d=json.load(open(fp))
    for e in d:
        inter=e.get("interactions") or []
        for il in inter:
            conv=il.get("interaction") if isinstance(il,dict) else None
            prompt=il.get("prompt") if isinstance(il,dict) else None
            if not conv: continue
            called_idb=any((tc.get("function") or {}).get("name")=="internal_get_database"
                           for m in conv if isinstance(m,dict) for tc in (m.get("tool_calls") or []))
            if called_idb:
                offered = (prompt is not None and "internal_get_database" in prompt)
                # also check the tool RESPONSE: did it return data or an error (not-a-tool)?
                resp=None
                for m in conv:
                    if isinstance(m,dict) and m.get("tool_name")=="internal_get_database":
                        resp=str(m.get("content"))[:80]; break
                print(f"  {fp.split('/')[-1][:40]} goal={e['task']['user_goal']} OFFERED_in_prompt={offered} resp={resp}")
                break
    break  # one file is enough for the offered-vs-hallucinated question

print("\n=== (B) OUR loginfirst run: transfer_funds should_T — cred state, BOTH, how passed ===")
CANON="/home/woori/scratch/SOPBench/data/bank_tasks.json"
canon=json.load(open(CANON)); cl=[]
if isinstance(canon,dict):
    for v in canon.values(): cl+=v if isinstance(v,list) else [v]
import collections
cmap=collections.defaultdict(list)
for ct in cl:
    if isinstance(ct,dict): cmap[ident(ct)].append(ct.get("user_known") or {})
def credstate(idt):
    cks=cmap.get(idt)
    if not cks: return "?"
    return "cred-present" if all("identification" in u for u in cks) else "cred-absent"

F="/home/woori/scratch/sft_alias_run/eval_t1c_loginfirst/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
d=json.load(open(F))
for e in d:
    t=e["task"]; ev=e["evaluations"][0]
    if t["user_goal"]!="transfer_funds" or not ev.get("action_should_succeed"): continue
    idt=ident(t); cs=credstate(idt)
    both=ev.get("dirgraph_satisfied") and ev.get("action_successfully_called")
    conv=e["interactions"][0]["interaction"]
    calls=[]
    for m in conv:
        if isinstance(m,dict):
            for tc in (m.get("tool_calls") or []):
                calls.append((tc.get("function") or {}).get("name"))
    uk_has_id = "identification" in (t.get("user_known") or {})
    print(f"  cred={cs} BOTH={both} dg={ev.get('dirgraph_satisfied')} cnv={ev.get('constraint_not_violated')} uk_has_id(augmented?)={uk_has_id}")
    print(f"     calls={calls}")
