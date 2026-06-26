#!/usr/bin/env python3
"""When released llama3.1-70b called internal_get_database, was it OFFERED in the prompt's tool
list (intended tool) or hallucinated? And did the call return DB data or an error?"""
import json, glob
fp=sorted(glob.glob("/home/woori/scratch/SOPBench/output/bank/ast_*llama3.1-70b*tool_full*.json"))[0]
print("file:", fp.split("/")[-1])
d=json.load(open(fp))
shown=0
for e in d:
    for il in (e.get("interactions") or []):
        conv=il.get("interaction") if isinstance(il,dict) else None
        prompt=il.get("prompt") if isinstance(il,dict) else None
        if not conv: continue
        # did any assistant message call internal_get_database?
        idb_call=False; resp=None
        for m in conv:
            if not isinstance(m,dict): continue
            for tc in (m.get("tool_calls") or []):
                fn=(tc.get("function") or {}).get("name") if tc.get("function") else tc.get("tool_name")
                if fn=="internal_get_database": idb_call=True
            if m.get("tool_name")=="internal_get_database" and resp is None:
                resp=str(m.get("content"))[:90]
        if idb_call:
            offered = bool(prompt) and ("internal_get_database" in prompt)
            print(f"  goal={e['task']['user_goal']} OFFERED_in_prompt={offered} resp={resp}")
            shown+=1
            if shown>=6: break
    if shown>=6: break
# Also: does the prompt list ANY internal_ tool? sanity on prompt content
if d:
    p=(d[0].get("interactions") or [{}])[0].get("prompt") or ""
    print("\n  prompt mentions internal_get_database:", "internal_get_database" in p)
    print("  prompt mentions internal_check_username_exist:", "internal_check_username_exist" in p)
    print("  prompt length:", len(p))
