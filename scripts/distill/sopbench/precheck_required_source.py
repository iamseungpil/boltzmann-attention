"""precheck_required_source.py — RUNG1 §2 GATE (run BEFORE any T1 teacher edit).

Question (handoff §2): is `task["constraints"]` the AUTHORITATIVE required_set, or do some
goals require an establishable (login/auth) via the evaluator's *innate / default-induced*
goal precondition that is NOT present in task["constraints"]?

  - constraints-establishable   = establishable leaves found IN task["constraints"].
  - authoritative-establishable = establishable leaves the evaluator's dependency requires for
                                   the goal = union of
                                     (i)  dep_innate[goal]   (action_innate_dependencies), and
                                     (ii) the goal's DEFAULT induced ontology precondition
                                          (ops[goal]["precondition"]) establishable leaves.
                                   This is exactly what run_scripted mode=abc's C-phase follows
                                   (run_scripted.py L124-139) and what evaluator_function_directed_graph
                                   scores against.

BRANCH (handoff §2):
  - If for EVERY task  authoritative-establishable  ⊆  constraints-establishable
      => task["constraints"] IS the authoritative required_set. SAME.
      => T1 uses task["constraints"] leaves directly.
  - If SOME task has an authoritative establishable NOT in constraints (login required innately
    but omitted from constraints) => DIFFERENT.
      => required_set source = the evaluator-authoritative synthesized dependency
         (constraints leaves UNION goal-default establishable). Still uniform, login non-special.

This script ONLY measures; it changes nothing. Run from the SOPBench clone root with seka_env:
  cd /home/woori/scratch/SOPBench
  /home/woori/venvs/seka_env/bin/python scripts/precheck_required_source.py
"""
import sys, os, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.task import task_default_dep_full, get_default_dep_full
from env.variables import domain_assistant_keys

DOMAINS = ["bank", "dmv", "healthcare", "hotel", "library", "online_market", "university"]


def walk(tree, acc):
    """Flatten a dep/constraint tree to leaf predicate names (strip 'not ')."""
    if not tree or not isinstance(tree, (list, tuple)) or not tree:
        return
    h = tree[0]
    if h == "single":
        nm = tree[1]
        acc.append(nm[4:] if nm.startswith("not ") else nm)
    elif h in ("and", "or", "chain", "gate"):
        for s in tree[1]:
            walk(s, acc)


def est_leaves(tree, preds):
    """Establishable-kind leaves of a tree."""
    acc = []
    walk(tree, acc)
    return {p for p in acc if preds.get(p, {}).get("kind") == "establishable"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--ont_dir", default="./induced")
    ap.add_argument("--limit", type=int, default=0, help="per-domain task cap (0=all)")
    args = ap.parse_args()

    grand_total = 0
    grand_diff = 0
    print("domain\tgoal\tn_tasks\tn_diff\t(diff = goal needs establishable NOT in constraints)")
    for domain in DOMAINS:
        try:
            ont = json.load(open(f"{args.ont_dir}/ontology_{domain}.json"))
            preds = ont.get("predicates", {})
            ops = ont.get("operators", {})
            dep_innate = domain_assistant_keys[domain].action_innate_dependencies
            dep_full_raw = get_default_dep_full(domain, "full")
            raw = json.load(open(f"{args.data_dir}/{domain}_tasks.json"))
        except Exception as e:
            print(f"[{domain}] LOAD FAILED: {e.__class__.__name__}: {e}", file=sys.stderr)
            continue

        per_goal = {}
        for goal in raw:
            tasks = raw[goal]
            if args.limit:
                tasks = tasks[:args.limit]
            for task in tasks:
                grand_total += 1
                # constraints-establishable
                con_est = est_leaves(task.get("constraints"), preds)
                # authoritative-establishable = innate dep[goal] U goal default-induced precond
                auth_est = set()
                auth_est |= est_leaves(dep_innate.get(goal), preds)
                auth_est |= est_leaves(dep_full_raw.get(goal), preds)
                auth_est |= est_leaves(ops.get(goal, {}).get("precondition"), preds)
                missing = auth_est - con_est   # required by evaluator, absent from constraints
                if missing:
                    grand_diff += 1
                    g = per_goal.setdefault(goal, {"n": 0, "diff": 0, "ex": None})
                    g["diff"] += 1
                    if g["ex"] is None:
                        g["ex"] = sorted(missing)
                per_goal.setdefault(goal, {"n": 0, "diff": 0, "ex": None})["n"] += 1
        for goal, g in sorted(per_goal.items()):
            ex = f"\tmissing_e.g.={g['ex']}" if g["ex"] else ""
            print(f"{domain}\t{goal}\t{g['n']}\t{g['diff']}{ex}")

    print("=" * 60)
    print(f"TOTAL tasks={grand_total}  tasks_with_authoritative_establishable_NOT_in_constraints={grand_diff}")
    if grand_diff == 0:
        print("GATE=SAME  -> required_set = task['constraints'] leaves (use constraints directly).")
    else:
        print("GATE=DIFFERENT  -> required_set = constraints leaves UNION goal-default establishable "
              "(evaluator-authoritative synthesized dependency); login still uniform/non-special.")


if __name__ == "__main__":
    main()
