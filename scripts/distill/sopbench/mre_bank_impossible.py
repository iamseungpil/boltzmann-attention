"""
mre_bank_impossible.py — Minimal reproducible evidence that a subset of SOPBench (Leezekun)
`bank` should_succeed=True tasks are STRUCTURALLY UNSOLVABLE: the author's own oracle answer
(the task `directed_action_graph`) does not pass the author's own evaluator.

This is the strongest possible bug evidence — it does NOT depend on our planner, our induced
ontology, or any LLM. It replays the GROUND-TRUTH call graph on the strict domain system and
scores it with the UNCHANGED `evaluator_function_directed_graph`. If the oracle trajectory
fails on should_succeed=True, the task cannot be passed by ANY agent => benchmark defect.

It produces three artifacts for a GitHub bug report (Leezekun/SOPBench):
  (A) ORACLE-REPLAY TABLE: per should_succeed=True task, success T/F + which sub-check failed
      (no_tool_call_error / constraint_not_violated / database_match / action_called_correctly
      / dirgraph_satisfied). The set of (oracle-replay == False) tasks = the impossible set.
  (B) CROSS-CHECK vs author trajectories: for each impossible task, whether ANY model in
      output/bank/*.json ever passed it (expected: none) — corroborating, optional.
  (C) ROOT-CAUSE PROBE: dump the type of `credit_cards` in a bank account row from
      initial_database (list vs dict) and grep the bank source for dict-style indexing of
      credit_cards / card_num so the report can cite file:line.

RUN (in the SOPBench clone root, py>=3.10 env, e.g. seka_env):
    python scripts/mre_bank_impossible.py --domain bank
    python scripts/mre_bank_impossible.py --domain bank --crosscheck --out mre_bank.json

NOTE: replay walks `directed_action_graph` nodes in listed order (the graph is a chain/DAG of
the correct calls). If a domain needs strict topological order, set --toposort (uses edges).
Args for each node come from the node's {param->slot} binding, resolved from user_known then
the acting account row — identical sourcing to l0_planner.py (verified path).
"""
import argparse
import copy
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SUBCHECKS = ["no_tool_call_error", "constraint_not_violated", "database_match",
             "action_called_correctly", "dirgraph_satisfied"]


def _toposort(nodes, edges):
    """Return node-action order from edges if a valid DAG order exists, else listed order."""
    names = [n[0] for n in nodes if isinstance(n, (list, tuple)) and n]
    indeg = {a: 0 for a in names}
    adj = {a: [] for a in names}
    for e in edges or []:
        if isinstance(e, (list, tuple)) and len(e) == 2 and e[0] in indeg and e[1] in indeg:
            adj[e[0]].append(e[1]); indeg[e[1]] += 1
    queue = [a for a in names if indeg[a] == 0]
    order, seen = [], set()
    while queue:
        a = queue.pop(0)
        if a in seen:
            continue
        seen.add(a); order.append(a)
        for b in adj[a]:
            indeg[b] -= 1
            if indeg[b] == 0:
                queue.append(b)
    return order if len(order) == len(names) else names


def replay_oracle(task, domain, dep_innate, dep_full, domain_keys, toposort):
    """Execute the task's directed_action_graph on the strict system; return func_calls+db."""
    # mirror evaluator: the goal's dependency is overridden by the task constraints
    task_dep = dict(dep_full)
    task_dep[task["user_goal"]] = task["constraints"]
    dss = domain_keys[domain + "_strict"](
        copy.deepcopy(task["initial_database"]), dep_innate, task_dep,
        task["constraint_parameters"])

    # arg slot pool: user_known + acting account row (credentials etc.) — same as l0_planner
    slots = dict(task.get("user_known", {}))
    accounts = task["initial_database"].get("accounts", {})
    uname = slots.get("username")
    if uname in accounts and isinstance(accounts[uname], dict):
        for k, v in accounts[uname].items():
            slots.setdefault(k, v)

    dag = task.get("directed_action_graph", {}) or {}
    nodes = dag.get("nodes", []) or []
    argmap = {}
    for n in nodes:
        if isinstance(n, (list, tuple)) and len(n) == 2:
            argmap[n[0]] = n[1] or {}
    order = _toposort(nodes, dag.get("edges")) if toposort else [
        n[0] for n in nodes if isinstance(n, (list, tuple)) and n]

    func_calls = []
    for action in order:
        amap = argmap.get(action, {})
        args = {p: slots[s] for p, s in amap.items() if s in slots}
        try:
            content = getattr(dss, action)(**args)
        except Exception as e:
            content = f"{e.__class__.__name__}: {e}"
        func_calls.append({"tool_name": action, "arguments": args, "content": content})
    return func_calls, dss.evaluation_get_database()


def root_cause_probe(domain, data_dir):
    """Dump credit_cards type in a bank account + grep bank source for dict-style indexing."""
    out = {"credit_cards_observed": [], "source_hits": []}
    try:
        raw = json.load(open(f"{data_dir}/{domain}_tasks.json"))
        for g in raw:
            for t in raw[g]:
                for uname, row in (t["initial_database"].get("accounts", {}) or {}).items():
                    if isinstance(row, dict) and "credit_cards" in row:
                        cc = row["credit_cards"]
                        out["credit_cards_observed"].append(
                            {"task_goal": g, "user": uname, "type": type(cc).__name__,
                             "sample": cc[:1] if isinstance(cc, list) else cc})
                        break
                if out["credit_cards_observed"]:
                    break
            if out["credit_cards_observed"]:
                break
    except Exception as e:
        out["credit_cards_observed"] = f"probe error: {e.__class__.__name__}: {e}"

    # grep bank domain source for dict-style credit card access
    pat = re.compile(r"(credit_cards\s*\[|card_num\b|\['card_num'\]|\[\"card_num\"\])")
    roots = [os.path.join("env"), os.path.join("env", "domains", domain), "."]
    seen = set()
    for r in roots:
        for path in glob.glob(os.path.join(r, "**", "*.py"), recursive=True):
            if domain not in path.lower() and "evaluator" not in path.lower():
                continue
            if path in seen:
                continue
            seen.add(path)
            try:
                for i, line in enumerate(open(path, encoding="utf-8"), 1):
                    if pat.search(line):
                        out["source_hits"].append({"file": path, "line": i,
                                                    "code": line.strip()[:160]})
            except Exception:
                pass
    return out


def crosscheck_author_outputs(domain, impossible_goals, out_root="output"):
    """For each impossible goal, did ANY author trajectory ever pass it? (corroboration)"""
    res = {g: {"passed_by": [], "files_seen": 0} for g in impossible_goals}
    for path in glob.glob(os.path.join(out_root, domain, "**", "*.json"), recursive=True):
        try:
            data = json.load(open(path))
        except Exception:
            continue
        model = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
        records = data if isinstance(data, list) else data.get("results", data.get("tasks", []))
        if not isinstance(records, list):
            continue
        for rec in records:
            if not isinstance(rec, dict):
                continue
            g = rec.get("user_goal") or rec.get("goal")
            ev = rec.get("evaluation") or rec.get("eval") or {}
            if g in res:
                res[g]["files_seen"] += 1
                if isinstance(ev, dict) and ev.get("success"):
                    if model not in res[g]["passed_by"]:
                        res[g]["passed_by"].append(model)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="bank")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--default_constraint_option", default="full")
    ap.add_argument("--toposort", action="store_true",
                    help="order calls by directed_action_graph edges (default: listed order)")
    ap.add_argument("--crosscheck", action="store_true",
                    help="cross-check impossible goals against output/<domain>/*.json")
    ap.add_argument("--out", default=None, help="write full JSON artifact here")
    args = ap.parse_args()

    from env.variables import domain_assistant_keys, domain_keys
    from env.task import get_default_dep_full
    from env.evaluator import evaluator_function_directed_graph

    dep_innate = domain_assistant_keys[args.domain].action_innate_dependencies
    dep_full = get_default_dep_full(args.domain, args.default_constraint_option)
    raw = json.load(open(f"{args.data_dir}/{args.domain}_tasks.json"))

    rows = []
    impossible = []          # (goal-keyed) should_succeed=True but oracle replay fails
    n_should_true = 0
    for g in raw:
        for t in raw[g]:
            t["user_goal"] = g
            if not t.get("action_should_succeed"):
                continue
            n_should_true += 1
            fcs, final_db = replay_oracle(t, args.domain, dep_innate, dep_full,
                                          domain_keys, args.toposort)
            ev = evaluator_function_directed_graph(
                args.domain, t, [], fcs, {"final_database": final_db},
                args.default_constraint_option)
            ok = bool(ev.get("success"))
            failed = [k for k in SUBCHECKS
                      if k in ev and not ev.get(k)] if isinstance(ev, dict) else []
            rows.append({"goal": g, "oracle_success": ok, "failed_subchecks": failed,
                         "n_calls": len(fcs)})
            if not ok:
                impossible.append(g)

    # de-dup goals for the headline count, but keep per-task rows
    impossible_goals = sorted(set(impossible))
    n_impossible_tasks = sum(1 for r in rows if not r["oracle_success"])

    print(f"=== MRE oracle-replay  {args.domain}  (should_succeed=True) ===")
    print(f"  tasks (should_succeed=True)         : {n_should_true}")
    print(f"  oracle-replay FAILED (impossible)   : {n_impossible_tasks} tasks  "
          f"/ {len(impossible_goals)} unique goals")
    print(f"  => effective ceiling on should_succeed=True ≈ "
          f"{(n_should_true - n_impossible_tasks)/n_should_true:.1%}" if n_should_true else "")
    print("  --- impossible goals + dominant failing sub-check ---")
    from collections import Counter
    by_goal = {}
    for r in rows:
        if not r["oracle_success"]:
            by_goal.setdefault(r["goal"], Counter())
            by_goal[r["goal"]].update(r["failed_subchecks"] or ["<none-flagged>"])
    for g in impossible_goals:
        print(f"    {g}: {dict(by_goal[g])}")

    probe = root_cause_probe(args.domain, args.data_dir)
    print("  --- root-cause probe ---")
    print(f"    credit_cards observed: {probe['credit_cards_observed']}")
    for h in probe["source_hits"][:20]:
        print(f"    {h['file']}:{h['line']}: {h['code']}")

    artifact = {"domain": args.domain, "n_should_true": n_should_true,
                "n_impossible_tasks": n_impossible_tasks,
                "impossible_goals": impossible_goals, "rows": rows, "probe": probe}

    if args.crosscheck:
        cc = crosscheck_author_outputs(args.domain, impossible_goals)
        artifact["crosscheck"] = cc
        print("  --- cross-check vs author output/ (passed_by should be empty) ---")
        for g in impossible_goals:
            print(f"    {g}: passed_by={cc[g]['passed_by']} (files={cc[g]['files_seen']})")

    if args.out:
        json.dump(artifact, open(args.out, "w"), indent=2, default=str)
        print(f"  wrote artifact -> {args.out}")


if __name__ == "__main__":
    main()
