#!/usr/bin/env python3
"""Full census of should_T (and should_F) failures for an arm-4a output.

For each record, reconstruct the ordered tool-call sequence (func, args, result) and the
goal's ABox precondition requirements, then classify WHY it failed against the dirgraph /
success gates. Outputs per-record lines + aggregate tallies.

Usage: python _census_shouldT.py <output_dir_or_json> <ontology_dir>
"""
import json, sys, glob, os, ast

def try_parse(s):
    if not isinstance(s, str): return s
    for fn in (json.loads, ast.literal_eval):
        try: return fn(s)
        except Exception: pass
    return s

def load_eval(path):
    if os.path.isdir(path):
        path = sorted(glob.glob(os.path.join(path, "*full*shuffle_False.json")))[0]
    return json.load(open(path)), path

def walk_leaves(tree, acc):
    """Collect (pred_name, param_map, negated) leaves in precondition order."""
    if not tree: return
    if isinstance(tree, (list, tuple)) and tree:
        head = tree[0]
        if head == "single":
            name = tree[1]; pm = tree[2] if len(tree) > 2 else {}
            neg = name.startswith("not ")
            acc.append((name[4:] if neg else name, pm, neg))
        elif head in ("and", "or", "chain", "gate"):
            for sub in tree[1]: walk_leaves(sub, acc)

def calls_of(record):
    """Ordered [(func, args_dict, result_obj)] from the interaction thread."""
    out = []
    for inter in record.get("interactions", []):
        thread = inter.get("interaction", [])
        pend = []
        for m in thread:
            tcs = m.get("tool_calls")
            if tcs:
                for tc in tcs:
                    f = tc.get("function", {})
                    a = try_parse(f.get("arguments", "{}"))
                    pend.append((f.get("name"), a if isinstance(a, dict) else {}))
            if m.get("tool_name") is not None and "tool_call_id" in m:
                res = try_parse(m.get("content"))
                nm = m.get("tool_name")
                # match to most recent pending call of same name
                for j in range(len(pend) - 1, -1, -1):
                    if pend[j][0] == nm:
                        out.append((nm, pend[j][1], res)); pend.pop(j); break
                else:
                    out.append((nm, {}, res))
    return out

def is_truthy(res):
    v = res
    if isinstance(res, (list, tuple)) and res:
        v = res[0]
    return v is True or (isinstance(v, (int, float)) and v not in (0, False)) or \
           (isinstance(v, str) and v.lower() not in ("false", "none", "", "error"))

def main():
    data, resolved = load_eval(sys.argv[1])
    ontdir = sys.argv[2]
    ont = json.load(open(os.path.join(ontdir, "ontology_bank.json")))
    ops = ont["operators"]; preds = ont["predicates"]
    # callable tool set = union of all tool names actually emitted
    callable_tools = set()
    for rec in data:
        for c in calls_of(rec):
            callable_tools.add(c[0])

    tally = {}
    def bump(k): tally[k] = tally.get(k, 0) + 1

    for which, label in [(True, "should_T")]:
        print(f"\n################## {label} FAILURE CENSUS ##################")
        for idx, rec in enumerate(data):
            ev = rec["evaluations"][0]
            if bool(ev.get("action_should_succeed")) != which: continue
            if ev.get("success"): continue
            goal = rec["task"]["user_goal"]
            uk = rec["task"].get("user_known", {})
            op = ops.get(goal, {})
            leaves = []; walk_leaves(op.get("precondition"), leaves)
            # required explicit calls: callable conditions (internal_*/tool) + establishable -> by-action
            req = []   # (display, func_to_call, param_map, kind)
            for name, pm, neg in leaves:
                info = preds.get(name, {})
                if info.get("kind") == "establishable" and info.get("by"):
                    req.append((name, info["by"], pm, "est"))
                elif name in callable_tools or name.startswith("internal_"):
                    req.append((name, name, pm, "check"))
                else:
                    req.append((name, None, pm, "implied"))  # computed fact, not explicitly callable
            seq = calls_of(rec)
            called = {}  # func -> (args, truthy)
            for nm, a, r in seq:
                # keep last call of each func
                called[nm] = (a, is_truthy(r))
            goal_call = called.get(goal)
            # classify
            problems = []
            for disp, fn, pm, kind in req:
                if kind == "implied": continue
                if fn not in called:
                    problems.append(("SKIP", disp, fn))
                elif not called[fn][1]:
                    problems.append(("FALSE", disp, fn, called[fn][0]))  # called but returned false/err (likely arg halluc)
            goal_ok = bool(goal_call and goal_call[1])
            gates = {k: ev.get(k) for k in ["action_successfully_called","dirgraph_satisfied",
                     "action_called_correctly","constraint_not_violated","database_match","no_tool_call_error"]}
            # bucket
            skip = [p for p in problems if p[0] == "SKIP"]
            false = [p for p in problems if p[0] == "FALSE"]
            if not gates["no_tool_call_error"]:
                cat = "tool_call_error"
            elif false and not skip:
                cat = "mode-b: prereq called but FALSE (arg halluc/login-fail)"
            elif skip and not false:
                cat = "mode-a: prereq SKIPPED"
            elif skip and false:
                cat = "mode-ab: both skip+false"
            elif not problems and not goal_ok:
                cat = "goal-fail (prereqs ok, goal returned false/err)"
            elif not problems and goal_ok and not gates["dirgraph_satisfied"]:
                cat = "dirgraph-other (all called+true, still dirgraph false)"
            elif not problems and goal_ok and not gates["constraint_not_violated"]:
                cat = "constraint-violation (should_T but constraint failed)"
            else:
                cat = "other"
            bump(cat)
            seqstr = " -> ".join(f"{nm}{'' if r else '=F'}" for nm,a,r in
                                 [(nm,a,is_truthy(r)) for nm,a,r in seq])
            print(f"[{idx}] {goal} | {cat}")
            print(f"     req={[ (d,f,k) for d,f,_,k in req if k!='implied']}")
            print(f"     seq={seqstr}")
            if false:
                for p in false:
                    print(f"     FALSE-prereq {p[2]} args={p[3]}")
            print(f"     gates={gates}")
        print(f"\n===== {label} failure category tally =====")
        for k,v in sorted(tally.items(), key=lambda x:-x[1]):
            print(f"  {v:3d}  {k}")
        print(f"  TOTAL failures classified: {sum(tally.values())}")

if __name__ == "__main__":
    main()
