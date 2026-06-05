#!/usr/bin/env python3
"""Evidence: does ANY released leaderboard model ever CALL internal_get_database?
(Code shows it's removed from agent tool specs unless provide_database_getter=True, which is
never set.) Scan all released output/bank/*.json for internal_get_database tool calls.
Also report our own runs (which DRIVE it via the gate) for contrast."""
import json, glob, collections
def count_idb_calls(path):
    try: d=json.load(open(path))
    except Exception: return None
    if not isinstance(d,list): return None
    n_calls=0; n_tasks_with=0; n_tasks=0
    for e in d:
        if not isinstance(e,dict): continue
        inter=e.get("interactions") or []
        seen=False
        for il in inter:
            conv=il.get("interaction") if isinstance(il,dict) else None
            if not conv: continue
            n_tasks+=1
            for m in conv:
                if not isinstance(m,dict): continue
                for tc in (m.get("tool_calls") or []):
                    fn=(tc.get("function") or {}).get("name") or tc.get("tool_name")
                    if fn=="internal_get_database": n_calls+=1; seen=True
        if seen: n_tasks_with+=1
    return (n_calls, n_tasks_with)

print("=== RELEASED output/bank/*.json ===")
tot=0
for fp in sorted(glob.glob("/home/woori/scratch/SOPBench/output/bank/ast_*.json")):
    r=count_idb_calls(fp)
    if r is None: continue
    name=fp.split("/")[-1].replace("ast_","").split("-mode")[0]
    mode="".join([s for s in fp.split("-") if "tool_" in s])
    if r[0]>0:
        print(f"  {name} {mode}: internal_get_database calls={r[0]} tasks={r[1]}")
        tot+=r[0]
print(f"RELEASED total internal_get_database calls across all files: {tot}")

print("\n=== OUR runs (gate drives it) ===")
for fp in ["/home/woori/scratch/sft_alias_run/eval_t1c_loginfirst/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json",
           "/home/woori/scratch/sft_alias_run/eval_t1c_dggate/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"]:
    r=count_idb_calls(fp)
    print(f"  {fp.split('/')[-3]}: internal_get_database calls={r[0] if r else 'NA'}")
