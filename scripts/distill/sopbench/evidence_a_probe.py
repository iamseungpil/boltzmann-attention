"""evidence_a_probe.py — decisive evidence-A for the bank "never-passed" instances.

For each bank should_succeed=True task instance that evidence-B found is never passed by any
released model, test directly on the STRICT domain system (the authors' own oracle env):
  1. build the strict system from the task's initial_database + constraints,
  2. satisfy the goal's dependency chain by calling each establishable precondition action
     (login_user / authenticate_admin_password / internal_* ) with GT-resolved args,
  3. call the GOAL action with its GT args, capture the return value,
  4. score the whole call sequence with the UNCHANGED evaluator_function_directed_graph.

If the GOAL action returns False (or the evaluator's action_successfully_called stays False)
*even with every precondition satisfied and correct args*, the task is unpassable by ANY
agent => benchmark defect (not difficulty). This does NOT depend on replay ORDER of the
directed_action_graph (the source of the unreliable MRE's spurious dirgraph failures): we
satisfy deps explicitly, then call the goal.

RUN (clone root):  python scripts/evidence_a_probe.py --domain bank
"""
import argparse, copy, json, os, sys, collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# bank auth/precondition actions, in the order they must be satisfied
PRECOND_ORDER = ["internal_get_database", "internal_check_username_exist",
                 "login_user", "authenticate_admin_password"]


def resolve_args(action_argmap, slots):
    return {p: slots[s] for p, s in action_argmap.items() if s in slots}


def gt_argmaps(task):
    """{action -> {param->slot}} from the task's directed_action_graph nodes."""
    out = {}
    for n in task.get("directed_action_graph", {}).get("nodes", []):
        if isinstance(n, (list, tuple)) and len(n) == 2 and isinstance(n[0], str):
            out[n[0]] = n[1] or {}
    return out


def probe_instance(task, domain, dep_innate, dep_full, domain_keys):
    task_dep = dict(dep_full)
    task_dep[task["user_goal"]] = task["constraints"]
    dss = domain_keys[domain + "_strict"](
        copy.deepcopy(task["initial_database"]), dep_innate, task_dep,
        task["constraint_parameters"])

    slots = dict(task.get("user_known", {}))
    accounts = task["initial_database"].get("accounts", {})
    uname = slots.get("username")
    if uname in accounts and isinstance(accounts[uname], dict):
        for k, v in accounts[uname].items():
            slots.setdefault(k, v)

    argmaps = gt_argmaps(task)
    goal = task["user_goal"]
    func_calls = []

    def call(action):
        amap = argmaps.get(action, {})
        args = resolve_args(amap, slots)
        try:
            content = getattr(dss, action)(**args)
        except Exception as e:
            content = f"{e.__class__.__name__}: {e}"
        func_calls.append({"tool_name": action, "arguments": args, "content": content})
        return content

    # 1) satisfy preconditions that appear in this task's GT graph, in dep order
    for a in PRECOND_ORDER:
        if a in argmaps and a != goal:
            call(a)
    # 2) call the goal action with GT args
    goal_ret = call(goal)
    return func_calls, dss.evaluation_get_database(), goal_ret, slots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="bank")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--default_constraint_option", default="full")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from env.variables import domain_assistant_keys, domain_keys
    from env.task import get_default_dep_full
    from env.evaluator import evaluator_function_directed_graph

    dep_innate = domain_assistant_keys[args.domain].action_innate_dependencies
    dep_full = get_default_dep_full(args.domain, args.default_constraint_option)
    raw = json.load(open(f"{args.data_dir}/{args.domain}_tasks.json"))

    rows = []
    for g in raw:
        per = 0
        for t in raw[g]:
            t["user_goal"] = g
            if not t.get("action_should_succeed"):
                continue
            idx = per; per += 1
            fcs, final_db, goal_ret, slots = probe_instance(
                t, args.domain, dep_innate, dep_full, domain_keys)
            ev = evaluator_function_directed_graph(
                args.domain, t, [], fcs, {"final_database": final_db},
                args.default_constraint_option)
            # goal return truthiness (bank methods return bool or (bool, ...))
            gr = goal_ret
            gr_ok = (gr is True) or (isinstance(gr, (list, tuple)) and gr and gr[0] is True)
            rows.append({
                "goal": g, "idx": idx,
                "goal_return": str(goal_ret)[:60],
                "goal_returned_true": bool(gr_ok),
                "action_successfully_called": ev.get("action_successfully_called"),
                "success": ev.get("success"),
                "n_calls": len(fcs),
                "cc_type": type(t["initial_database"]["accounts"]
                                .get(slots.get("username"), {})
                                .get("credit_cards")).__name__
                           if isinstance(t["initial_database"]["accounts"]
                                         .get(slots.get("username"), {}), dict) else "NA",
            })

    # summary: among should=True, how many can the ORACLE not make succeed?
    n = len(rows)
    oracle_fail = [r for r in rows if not r["action_successfully_called"]]
    by_goal = collections.Counter(r["goal"] for r in oracle_fail)
    print(f"=== evidence-A (oracle dep-satisfied call) {args.domain}  should=True N={n} ===")
    print(f"  ORACLE could NOT successfully call the goal: {len(oracle_fail)} instances "
          f"/ {len(by_goal)} goals")
    for gname, c in sorted(by_goal.items()):
        ex = next(r for r in oracle_fail if r["goal"] == gname)
        print(f"    {gname}: {c}  e.g. goal_return={ex['goal_return']} cc_type={ex['cc_type']}")
    print("  --- per-instance (oracle-fail only) ---")
    for r in oracle_fail:
        print(f"    {r['goal']}[{r['idx']}] ret={r['goal_return']} "
              f"asc={r['action_successfully_called']} success={r['success']} cc={r['cc_type']}")
    if args.out:
        json.dump({"domain": args.domain, "n_should_true": n,
                   "oracle_cannot_succeed": len(oracle_fail),
                   "by_goal": dict(by_goal), "rows": rows}, open(args.out, "w"), indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
