"""
l0_planner.py — arm-2 L0 symbolic means-ends planner for Zekun-SOPBench (no LLM).

The CLEAN ARBITER (code-review + design-review C-2): runs the §11.3 means-ends rule over the
induced ABox (`ontology_<domain>.json`) on the GROUND-TRUTH domain system — zero LLM, zero
parse/IO/tool_choice confounds (C1/C2/N1 absent). So a low number here means the *rule* is
wrong; a high number means the rule is right and the arm-3 0% is an LLM/IO artifact.

Two rules, both run (resolves design open-decision #1 with data, §11.11):
  --rule regression : true means-ends with backward goal-stack regression (design A1).
  --rule greedy     : the old greedy-forward (no subgoal push) — expected to stall on
                      precondition chains (e.g. login before action).

EXECUTION = drive the STRICT domain system (it enforces preconditions and returns the GT
response), so a satisfied call's `content` matches the evaluator's GT replay -> constraint_not
_violated + database_match hold by construction. REFUSE = do not call the goal (N1/§7: refusal
tasks pass iff the target is not successfully called).

ARG SOURCING: action params resolve from the task `user_known` first, else from the user's
account row in `initial_database` (credentials like identification/admin_password live there).

RUN (clone root):  python scripts/l0_planner.py --domain bank --rule regression [--num_tasks N]
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def establishable_leaves(tree, ont, acc):
    """Collect (predicate, param_mapping, establishing_action) for establishable leaves."""
    if not tree:
        return
    if isinstance(tree, (list, tuple)) and tree:
        head = tree[0]
        if head == "single":
            name = tree[1]
            name = name[4:] if name.startswith("not ") else name
            pm = tree[2] if len(tree) > 2 else {}
            info = ont["predicates"].get(name, {})
            if info.get("kind") == "establishable" and info.get("by"):
                acc.append((name, pm, info["by"]))
        elif head in ("and", "or", "chain", "gate"):
            for sub in tree[1]:
                establishable_leaves(sub, ont, acc)


def run_task(task, domain, ont, dep_innate, dep_full, domain_keys, rule):
    """Means-ends drive the strict system; return (func_calls, final_db, refused)."""
    dss = domain_keys[domain + "_strict"](
        copy.deepcopy(task["initial_database"]), dep_innate, dep_full, task["constraint_parameters"])
    de = dss.evaluation_get_dependency_evaluator()
    st = dss.evaluation_get_innate_state_tracker()

    # slot pool: user_known + the acting user's account row (credentials etc.)
    slots = dict(task.get("user_known", {}))
    accounts = task["initial_database"].get("accounts", {})
    uname = slots.get("username")
    if uname in accounts and isinstance(accounts[uname], dict):
        for k, v in accounts[uname].items():
            slots.setdefault(k, v)

    func_calls = []

    def resolve_args(action):
        amap = ont["operators"][action]["args"]
        return {p: slots[s] for p, s in amap.items() if s in slots}

    def execute(action):
        args = resolve_args(action)
        try:
            content = getattr(dss, action)(**args)
        except Exception as e:
            content = f"{e.__class__.__name__}: {e}"
        func_calls.append({"tool_name": action, "arguments": args, "content": content})
        return content

    def precond_ok(action):
        try:
            return bool(de.process(action, **slots))
        except Exception:
            return False

    def state_true(pred, pm):
        try:
            fargs = {k: slots[pm[k]] for k in pm if pm[k] in slots}
            r = getattr(st, pred)(**fargs)
            return r if isinstance(r, bool) else bool(r[1])
        except Exception:
            return False

    visiting = set()

    def satisfy(action, depth=0):
        """Ensure `action`'s preconditions by calling establishable subgoals (regression).
        Returns True iff `action`'s full precondition holds afterwards."""
        if depth > 12 or action in visiting:
            return precond_ok(action)
        visiting.add(action)
        if rule == "regression":
            leaves = []
            establishable_leaves(ont["operators"][action]["precondition"], ont, leaves)
            for pred, pm, by in leaves:
                if not state_true(pred, pm):
                    satisfy(by, depth + 1)          # recurse on subgoal's own preconditions
                    if precond_ok(by):
                        execute(by)                 # establish the state predicate
        visiting.discard(action)
        return precond_ok(action)

    goal = task["user_goal"]
    refused = False
    if satisfy(goal) and precond_ok(goal):
        execute(goal)
    else:
        refused = True                              # N1/§7: refuse = do not call the goal
    return func_calls, dss.evaluation_get_database(), refused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="bank")
    ap.add_argument("--rule", default="regression", choices=["regression", "greedy"])
    ap.add_argument("--num_tasks", type=int, default=None)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--ont_dir", default="./induced")
    ap.add_argument("--default_constraint_option", default="full")
    args = ap.parse_args()

    from env.variables import domain_assistant_keys, domain_keys
    from env.task import get_default_dep_full
    from env.evaluator import evaluator_function_directed_graph

    ont = json.load(open(f"{args.ont_dir}/ontology_{args.domain}.json"))
    dep_innate = domain_assistant_keys[args.domain].action_innate_dependencies
    dep_full = get_default_dep_full(args.domain, args.default_constraint_option)

    raw = json.load(open(f"{args.data_dir}/{args.domain}_tasks.json"))
    tasks = []
    for g in raw:
        for t in raw[g]:
            t["user_goal"] = g
            tasks.append(t)
    if args.num_tasks:
        tasks = tasks[:args.num_tasks]

    n = succ = ref_correct = succ_should = 0
    n_should_true = n_should_false = 0
    for t in tasks:
        fcs, final_db, refused = run_task(t, args.domain, ont, dep_innate, dep_full, domain_keys, args.rule)
        ev = evaluator_function_directed_graph(
            args.domain, t, [], fcs, {"final_database": final_db}, args.default_constraint_option)
        n += 1
        ok = ev.get("success", False)
        succ += int(ok)
        should = t.get("action_should_succeed")
        if should:
            n_should_true += 1; succ_should += int(ok)
        else:
            n_should_false += 1; ref_correct += int(ok)

    print(f"=== L0 ({args.rule}) {args.domain}  N={n} ===")
    print(f"  pass@1            : {succ/n:.4f}  ({succ}/{n})")
    print(f"  should_succeed=T  : {succ_should}/{n_should_true}"
          + (f" ({succ_should/n_should_true:.3f})" if n_should_true else ""))
    print(f"  should_succeed=F  : {ref_correct}/{n_should_false}"
          + (f" ({ref_correct/n_should_false:.3f})" if n_should_false else ""))


if __name__ == "__main__":
    main()
