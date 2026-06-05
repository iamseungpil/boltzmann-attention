#!/usr/bin/env python3
"""Pinpoint WHY the first goal-call already fails cnv & dg (single-call truncation).
Replicates the evaluator's strict-replay (cnv) and dirgraph traversal (dg) with
per-call logging, so we see the exact mismatching call (cnv) and missing prereq node (dg)."""
import json, hashlib, sys, copy
sys.path.insert(0, "/home/woori/scratch/SOPBench")
from env.variables import domain_keys, domain_assistant_keys
from env.task import get_default_dep_full
from env.helpers import get_action_parameters, get_ifg_connections_invnodes, dfsgather_ifg_func
from env.evaluator import dfsconvert_tuple_to_list, dfsconvert_list_to_tuple

EVAL = "/home/woori/scratch/sft_alias_run/eval_t1c_dggate/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
DOMAIN = "bank"
OPT = "full"

def try_eval(x):
    try: return eval(x) if isinstance(x, str) else x
    except Exception: return x

def sig(t):
    return hashlib.md5(json.dumps([t.get("user_goal"),t.get("constraints"),t.get("user_known")],
        sort_keys=True, default=str).encode()).hexdigest()[:12]

def extract_func_calls(interaction):
    fc=[]
    for i in range(len(interaction)-1):
        if interaction[i].get("tool_calls", []):
            tcs=[tc for tc in interaction[i]["tool_calls"] if tc["function"]["name"].lower() not in ["n/a","na","none","null"]]
            if tcs:
                fc.append({"tool_name": interaction[i+1]["tool_name"],
                           "arguments": try_eval(tcs[0]["function"]["arguments"]),
                           "content": try_eval(interaction[i+1]["content"])})
    return fc

d = json.load(open(EVAL))

for e in d:
    t=e["task"]; ev=e["evaluations"][0]
    if not (ev.get("action_should_succeed") and ev.get("action_successfully_called") and not ev.get("dirgraph_satisfied")):
        continue
    s=sig(t); goal=t["user_goal"]
    fc=extract_func_calls(e["interactions"][0]["interaction"])
    # truncate to first goal call
    trunc=[]
    for c in fc:
        trunc.append(c)
        if c["tool_name"]==goal: break

    print(f"\n========== [{goal}] sig={s} | constraints_orig={json.dumps(t['constraints_original'])}")

    # ---- CNV: strict replay ----
    dep_innate_full = domain_assistant_keys[DOMAIN].action_innate_dependencies
    default_dep_full = get_default_dep_full(DOMAIN, OPT)
    default_dep_full[goal] = t["constraints"]
    dss = domain_keys[DOMAIN+"_strict"](copy.deepcopy(t["initial_database"]), dep_innate_full, default_dep_full, t["constraint_parameters"])
    ds = dss.evaluation_get_domain_system()
    print("  -- CNV per-call (agent_resp vs strict gt) --")
    for c in trunc:
        fn, ar = c["tool_name"], c["arguments"]
        try:
            if hasattr(dss, fn) and hasattr(ds, fn):
                gt=(0, copy.deepcopy(getattr(dss, fn)(**ar)))
            elif not hasattr(dss, fn) and hasattr(ds, fn):
                gt=(1, None)
            else:
                gt=(2, None)
        except Exception as ex:
            gt=(2, f"EXC:{ex}")
        agent=c["content"]
        fl=dfsconvert_tuple_to_list(agent); gl=dfsconvert_tuple_to_list(gt)
        match = (fl==gl) or ([True, fl]==gl)
        match = match if gt[0]==0 else True
        flag = "" if match else "  <<< CNV MISMATCH"
        print(f"     {fn}{ar}  agent={agent!r}  strict={gt!r}  match={match}{flag}")

    # ---- DG: dirgraph traversal for the goal node ----
    ifcg = copy.deepcopy(t["directed_action_graph"])
    nodes_task = ifcg["nodes"]
    conns_task, invn_task = get_ifg_connections_invnodes(ifcg)
    action_parameters = get_action_parameters(ds, domain_assistant_keys[DOMAIN])
    print(f"  -- DG: goal '{goal}' node in dirgraph? {goal in invn_task} ; dirgraph nodes={nodes_task}")
    print(f"     connections={conns_task}")
    if goal in invn_task:
        gi=invn_task[goal]
        print(f"     goal node idx={gi} prereq-neighbors={conns_task.get(gi)}")
        for ni in conns_task.get(gi, []):
            print(f"       prereq node[{ni}]={nodes_task[ni]} -> its neighbors {conns_task.get(ni)}")
