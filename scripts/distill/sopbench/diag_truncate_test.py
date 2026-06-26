#!/usr/bin/env python3
"""Counterfactual: for each premature task, truncate func_calls so the GOAL is
called exactly once (drop the repeated goal-calls + anything after the first goal),
then re-run the REAL evaluator. If cnv & dg flip to True, non-termination (repeated
goal-calls / no STOP after success) is the confirmed root cause and STOP-after-success
is the fix. Zero-cost (offline replay of the authoritative evaluator)."""
import json, hashlib, sys, copy
sys.path.insert(0, "/home/woori/scratch/SOPBench")
from env.evaluator import evaluator_function_directed_graph

EVAL = "/home/woori/scratch/sft_alias_run/eval_t1c_dggate/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"

def try_eval(x):
    try: return eval(x) if isinstance(x, str) else x
    except Exception: return x

def sig(t):
    return hashlib.md5(json.dumps(
        [t.get("user_goal"), t.get("constraints"), t.get("user_known")],
        sort_keys=True, default=str).encode()).hexdigest()[:12]

def extract_func_calls(interaction):
    fc = []
    for i in range(len(interaction)-1):
        if interaction[i].get("tool_calls", []):
            tcs = [tc for tc in interaction[i]["tool_calls"]
                   if tc["function"]["name"].lower() not in ["n/a","na","none","null"]]
            if tcs:
                fc.append({
                    "tool_name": interaction[i+1]["tool_name"],
                    "arguments": try_eval(tcs[0]["function"]["arguments"]),
                    "content": try_eval(interaction[i+1]["content"]),
                })
    return fc

d = json.load(open(EVAL))
print(f"{'goal':<28} {'sig':<13} ORIG(cnv,dg,acc,dbm) -> TRUNC(cnv,dg,acc,dbm)  FLIP?")
print("="*100)
n_flip = 0
for e in d:
    t = e["task"]; ev = e["evaluations"][0]
    if not (ev.get("action_should_succeed") and ev.get("action_successfully_called") and not ev.get("dirgraph_satisfied")):
        continue
    s = sig(t); goal = t["user_goal"]
    inter = e["interactions"][0]["interaction"]
    fc = extract_func_calls(inter)
    # truncate: keep up to and including FIRST goal call
    trunc = []
    for c in fc:
        trunc.append(c)
        if c["tool_name"] == goal:
            break
    # re-run evaluator on truncated. final_database: recompute via strict replay is internal;
    # we pass the ORIGINAL final db (only affects database_match). For idempotent goals this is valid;
    # for others database_match may be unreliable but cnv/dg/acc are exact for the truncated sequence.
    res = {"final_database": e["interactions"][0]["database"]}
    try:
        r = evaluator_function_directed_graph(
            domain_str=e["domain"], task=t, log_msg_fcall=inter,
            func_calls=trunc, results=res, default_constraint_option="full")
    except Exception as ex:
        print(f"{goal:<28} {s:<13} EVAL ERROR: {ex}")
        continue
    o = (ev.get("constraint_not_violated"), ev.get("dirgraph_satisfied"),
         ev.get("action_successfully_called"), ev.get("database_match"))
    n = (r.get("constraint_not_violated"), r.get("dirgraph_satisfied"),
         r.get("action_successfully_called"), r.get("database_match"))
    flip = n[0] and n[1] and n[2]  # cnv & dg & acc all true after truncation
    if flip: n_flip += 1
    print(f"{goal:<28} {s:<13} {str(o):<24} -> {str(n):<24} {'YES (cnv&dg&acc)' if flip else ''}  ncalls {len(fc)}->{len(trunc)}")

print(f"\n{n_flip} / 11 premature flip to cnv&dg&acc=True when goal called once.")
