#!/usr/bin/env python3
"""Robust per-task evaluation: load a sim output JSON, compute per-task `evaluations` via the
authoritative evaluator_function_directed_graph, and SAVE back. Bypasses run_evaluation.py's
goal-level statistics aggregation which crashes with ZeroDivisionError (total_interactions=0 for
some user_goal) BEFORE saving -> leaving healthcare/hotel/online_market with no evaluations (=0%).
Usage: eval_tasks.py <eval_output.json>"""
import json, sys
sys.path.insert(0, "/home/woori/scratch/SOPBench")
from env.evaluator import evaluator_function_directed_graph

def try_eval(x):
    try: return eval(x) if isinstance(x, str) else x
    except Exception: return x

path = sys.argv[1]
d = json.load(open(path))
n = 0
for ts in d:
    inter = ts.get("interactions") or []
    if not inter:
        ts["evaluations"] = [{}]; continue
    il = inter[0]
    results = {"final_database": il["database"]}
    interaction = il["interaction"]
    func_calls = []
    for i in range(len(interaction) - 1):
        if interaction[i].get("tool_calls", []):
            tcs = [tc for tc in interaction[i]["tool_calls"]
                   if tc["function"]["name"].lower() not in ["n/a", "na", "none", "null"]]
            if tcs:
                func_calls.append({
                    "tool_name": interaction[i + 1]["tool_name"],
                    "arguments": try_eval(tcs[0]["function"]["arguments"]),
                    "content": try_eval(interaction[i + 1]["content"]),
                })
    try:
        ev = evaluator_function_directed_graph(
            domain_str=ts["domain"], task=ts["task"], log_msg_fcall=interaction,
            func_calls=func_calls, results=results, default_constraint_option="full")
    except Exception as ex:
        ev = {"success": False, "_eval_error": str(ex)}
    ts["evaluations"] = [ev]
    n += 1
json.dump(d, open(path, "w"), indent=2)
# quick official-success print
s = sum(1 for x in d if (x.get("evaluations") or [{}])[0].get("success"))
sT = sum(1 for x in d if (x.get("evaluations") or [{}])[0].get("success") and (x.get("evaluations") or [{}])[0].get("action_should_succeed"))
nT = sum(1 for x in d if (x.get("evaluations") or [{}])[0].get("action_should_succeed"))
print(f"eval_tasks: {path.split('/')[-3] if '/' in path else path}  official={s}/{len(d)}={100*s/len(d):.2f}%  shouldT={sT}/{nT}")
