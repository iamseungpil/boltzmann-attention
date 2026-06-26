#!/usr/bin/env python3
"""Quick BOTH/premature/deny count for a given eval JSON (arg1). Reports transfer 047d status."""
import json, hashlib, sys, collections
p=sys.argv[1]
d=json.load(open(p))
def sig(t): return hashlib.md5(json.dumps([t.get("user_goal"),t.get("constraints"),t.get("user_known")],sort_keys=True,default=str).encode()).hexdigest()[:12]
both=prem=deny=0; bygoal=collections.Counter(); prem_sigs=[]; both_sigs=set()
for e in d:
    t=e["task"]; ev=e["evaluations"][0]
    if not ev.get("action_should_succeed"): continue
    s=sig(t)
    if ev.get("dirgraph_satisfied") and ev.get("action_successfully_called"):
        both+=1; bygoal[t["user_goal"]]+=1; both_sigs.add(s)
    elif ev.get("action_successfully_called"):
        prem+=1; prem_sigs.append((t["user_goal"],s))
    else:
        deny+=1
print(f"{p.split('/')[-3] if '/' in p else p}")
print(f"  should_T BOTH={both} premature={prem} deny={deny}")
print(f"  BOTH by goal: {dict(bygoal)}")
print(f"  transfer 047ddc88900c in BOTH? {'047ddc88900c' in both_sigs}")
print(f"  premature sigs: {prem_sigs}")
