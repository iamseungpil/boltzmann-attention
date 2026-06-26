#!/usr/bin/env python3
"""should_T/should_F + failure-locus breakdown for SOPBench arm-4a outputs.
Usage: python _v2_breakdown.py <eval_json> [label]
"""
import json, sys, glob, os

GATES = ["no_tool_call_error", "constraint_not_violated", "database_match",
         "action_successfully_called", "dirgraph_satisfied", "action_called_correctly"]

def load(path):
    if os.path.isdir(path):
        cands = glob.glob(os.path.join(path, "*full*shuffle_False.json"))
        path = sorted(cands)[0]
    return json.load(open(path)), path

def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(path)
    data, resolved = load(path)
    n = 0; succ = 0
    buckets = {True: {"n": 0, "succ": 0, "fail_gate": {g: 0 for g in GATES}},
               False: {"n": 0, "succ": 0, "fail_gate": {g: 0 for g in GATES}}}
    for rec in data:
        for ev in rec.get("evaluations", []):
            n += 1
            ss = bool(ev.get("action_should_succeed"))
            ok = bool(ev.get("success"))
            buckets[ss]["n"] += 1
            if ok:
                succ += 1; buckets[ss]["succ"] += 1
            else:
                for g in GATES:
                    if ev.get(g) is False:
                        buckets[ss]["fail_gate"][g] += 1
    print(f"=== {label}  ({resolved}) ===")
    print(f"overall pass@1: {succ}/{n} = {100*succ/n:.1f}%")
    for ss, name in [(True, "should_T (perform)"), (False, "should_F (refuse)")]:
        b = buckets[ss]
        tot = b["n"]; s = b["succ"]
        print(f"  {name}: {s}/{tot} = {100*s/tot:.1f}%" if tot else f"  {name}: 0/0")
        fg = {g: c for g, c in b["fail_gate"].items() if c}
        if fg:
            print(f"      fail-locus (gate=False count among {tot-s} fails): {fg}")

if __name__ == "__main__":
    main()
