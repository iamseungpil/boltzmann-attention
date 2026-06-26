#!/usr/bin/env python3
"""Check task_sig (content-hash) collisions among the 48 should_T (handoff claimed 0)."""
import json, hashlib, collections
p="/home/woori/scratch/sft_alias_run/eval_t1c_dggate/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
d=json.load(open(p))
def sig(t): return hashlib.md5(json.dumps([t.get("user_goal"),t.get("constraints"),t.get("user_known")],sort_keys=True,default=str).encode()).hexdigest()[:12]
sigs=collections.Counter(); ent=collections.defaultdict(list)
for i,e in enumerate(d):
    if e["evaluations"][0].get("action_should_succeed"):
        s=sig(e["task"]); sigs[s]+=1; ent[s].append(i)
dups={s:c for s,c in sigs.items() if c>1}
print("should_T entries:",sum(sigs.values()),"unique sigs:",len(sigs),"DUP:",dups)
for s in dups:
    print("\nsig",s)
    for i in ent[s]:
        t=d[i]["task"]; ev=d[i]["evaluations"][0]
        idb=hashlib.md5(json.dumps(t["initial_database"],sort_keys=True,default=str).encode()).hexdigest()[:8]
        cp=hashlib.md5(json.dumps(t["constraint_parameters"],sort_keys=True,default=str).encode()).hexdigest()[:8]
        same_uk = t["user_known"]
        print("  idx",i,"goal",t["user_goal"],"idb",idb,"cparam",cp,
              "acc",ev.get("action_successfully_called"),"dg",ev.get("dirgraph_satisfied"),
              "cnv",ev.get("constraint_not_violated"),"uk",same_uk)
