#!/usr/bin/env python3
"""v4: full interleaved trace WITH tool return values for the 11 premature.
Focus: did the FIRST goal-call succeed? are repeated goal-calls the cnv/dg killer?
what did each establishing getter return?"""
import json, hashlib, collections

EVAL = "/home/woori/scratch/sft_alias_run/eval_t1c_dggate/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"

def sig(t):
    return hashlib.md5(json.dumps(
        [t.get("user_goal"), t.get("constraints"), t.get("user_known")],
        sort_keys=True, default=str).encode()).hexdigest()[:12]

d = json.load(open(EVAL))
premature = []
for e in d:
    t = e["task"]; ev = e["evaluations"][0]
    if ev.get("action_should_succeed") and ev.get("action_successfully_called") and not ev.get("dirgraph_satisfied"):
        premature.append(e)

for e in premature:
    t = e["task"]; ev = e["evaluations"][0]; s = sig(t); goal = t["user_goal"]
    print(f"\n===== [{goal}] sig={s}  cnv={ev.get('constraint_not_violated')} dg={ev.get('dirgraph_satisfied')} dbm={ev.get('database_match')} =====")
    conv = e["interactions"][0]["interaction"]
    # interleave: assistant tool_calls then matching tool responses
    pending = {}
    step = 0
    for m in conv:
        if not isinstance(m, dict):
            continue
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = (tc.get("function") or {}).get("name") or tc.get("tool_name")
                ar = (tc.get("function") or {}).get("arguments") or tc.get("arguments")
                tcid = tc.get("id") or tc.get("tool_call_id")
                pending[tcid] = (fn, ar)
                step += 1
                mark = " <<GOAL" if fn == goal else ""
                print(f"  {step:2d}. CALL {fn} {ar}{mark}")
        elif m.get("tool_name") is not None or m.get("tool_call_id") is not None:
            fn = m.get("tool_name")
            ret = m.get("content")
            print(f"        -> [{fn}] {str(ret)[:160]}")
