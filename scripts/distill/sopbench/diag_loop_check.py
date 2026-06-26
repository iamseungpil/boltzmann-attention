#!/usr/bin/env python3
"""For should_T that are BOTH(dg&acc) but NOT full success in a run, count goal-action calls
(looping?) and show cnv/dbm. Tests whether goal-call looping is the cnv/dbm killer. argv[1]=json."""
import json, sys, collections
p=sys.argv[1]; d=json.load(open(p))
for e in d:
    t=e["task"]; ev=e["evaluations"][0]; goal=t["user_goal"]
    if not ev.get("action_should_succeed"): continue
    both=ev.get("dirgraph_satisfied") and ev.get("action_successfully_called")
    full=ev.get("success")
    if not (both and not full): continue
    conv=e["interactions"][0]["interaction"]
    gc=0; total=0
    for m in conv:
        if isinstance(m,dict):
            for tc in (m.get("tool_calls") or []):
                fn=(tc.get("function") or {}).get("name") if tc.get("function") else tc.get("tool_name")
                total+=1
                if fn==goal: gc+=1
    print(f"  {goal:<24} goal_calls={gc} total_calls={total} cnv={ev.get('constraint_not_violated')} dbm={ev.get('database_match')} ntce={ev.get('no_tool_call_error')}")
