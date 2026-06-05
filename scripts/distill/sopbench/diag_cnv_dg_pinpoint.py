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
                status_id, gtval = 0, copy.deepcopy(getattr(dss, fn)(**ar))
            elif not hasattr(dss, fn) and hasattr(ds, fn):
                status_id, gtval = 1, None
            else:
                status_id, gtval = 2, None
        except Exception as ex:
            status_id, gtval = 2, f"EXC:{ex}"
        agent=c["content"]
        fl=dfsconvert_tuple_to_list(agent); gl=dfsconvert_tuple_to_list(gtval)
        match = (fl==gl) or ([True, fl]==gl)
        match = match if status_id==0 else True
        flag = "" if match else "  <<< CNV MISMATCH"
        print(f"     {fn}{ar}  agent={agent!r}  strict_val={gtval!r}  match={match}{flag}")

    # ---- DG: dirgraph traversal for the goal node ----
    ifcg = copy.deepcopy(t["directed_action_graph"])
    nodes_task = ifcg["nodes"]
    conns_task, invn_task = get_ifg_connections_invnodes(ifcg)
    action_parameters = get_action_parameters(ds, domain_assistant_keys[DOMAIN])
    called = set(c["tool_name"] for c in trunc)
    print(f"  -- DG: called funcs (trunc) = {sorted(called)}")
    print(f"     dirgraph nodes={nodes_task}")
    def cg(i):
        x = conns_task[i] if isinstance(conns_task, list) else conns_task.get(i)
        return x or set()
    def describe(ni, depth=0):
        node = nodes_task[ni]
        ind = "  " * depth
        if isinstance(node, str):  # and/or
            sub = cg(ni)
            print(f"     {ind}[{ni}] {node} -> {sorted(sub)}")
            for s in sorted(sub):
                describe(s, depth+1)
        else:
            fn = node[0]
            ok = fn in called
            print(f"     {ind}[{ni}] FUNC {fn}  called={ok}{'' if ok else '  <<< NOT CALLED'}")
    if goal in invn_task:
        gi=invn_task[goal]
        # goal node points to its prereq subtree
        for ni in sorted(cg(gi)):
            describe(ni, 1)
