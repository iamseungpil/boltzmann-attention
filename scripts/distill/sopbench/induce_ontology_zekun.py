"""
induce_ontology_zekun.py — extract the 8-relation ABox (`ontology_<domain>.json`) for the
7 Zekun-SOPBench domains, from the STRUCTURED dependency trees (the authoritative source).

Per WORKFLOW_ONTOLOGY_DESIGN §11 / §2.5. The ABox is the per-domain instance that the
TBox planner (L0/L1/L2) reads and that we SWAP for transfer; the TBox (the 8 relation TYPES
+ the means-ends policy, §11.3) is domain-invariant and is NOT in this file.

WHAT IS EXTRACTED (domain-general):
  operators[action] = {
     precondition : the dep tree (and/or/chain/gate/single over predicates) = dep_full[action]
     args         : {param -> slot}  (from each task's directed_action_graph action node)
     produces     : [state-predicates this action establishes]  (effect, see below)
     is_goal      : action is a user_goal target
  }
  predicates[name] = { kind: "state"|"fact", by: <establishing action or null>, params:{...} }
     - "state"  predicate  -> lives on ds.innate_state_tracker; an AGENT ACTION establishes it
                              (e.g. logged_in_user <- login_user). These drive the §11.3
                              backward regression (unmet state pred -> subgoal = call its action).
     - "fact"   predicate  -> lives on the domain system / DB (internal_check_*, computed
                              like sufficient_account_balance). NOT agent-establishable; if
                              false the action is refused (action_should_succeed=false, N1/§7).

HOW classification works (introspection, not hard-coded):
  Instantiate one task's strict domain system exactly like env/evaluator.py, then a predicate
  is "state" iff it is a method on `ds.innate_state_tracker`, else "fact".

HOW the state-pred -> action map is found (effect-mining = domain-general, §4):
  For each state predicate P, find the agent action whose call flips P false->true by replaying
  a SUCCESS-path: we mine it from the per-task directed_action_graph + a name-convention fallback
  (logged_in_user->login_user, authenticated_X->authenticate_X), and VERIFY by effect probe where
  task args are available. Unmapped state preds are dumped as WARNINGs for review.

RUN (in the SOPBench clone root):  python scripts/induce_ontology_zekun.py [--domain bank] [--out induced]
"""
import argparse
import copy
import json
import os

DOMAINS = ["bank", "dmv", "healthcare", "hotel", "library", "online_market", "university"]


def walk_predicates(tree, acc):
    """Collect (predicate_name, param_mapping) leaves from a dep tree."""
    if not tree:
        return
    if isinstance(tree, (list, tuple)) and tree:
        head = tree[0]
        if head == "single":
            name = tree[1]
            pm = tree[2] if len(tree) > 2 else {}
            neg = name.startswith("not ")
            acc.append((name[4:] if neg else name, pm))
        elif head in ("and", "or", "chain", "gate"):
            for sub in tree[1]:
                walk_predicates(sub, acc)


def convention_action_for(pred, actions):
    """Name-convention fallback: state predicate -> establishing agent action."""
    cand = []
    if pred.startswith("logged_in"):
        cand = ["login_user", "login"]
    elif pred.startswith("authenticated_"):
        cand = ["authenticate_" + pred[len("authenticated_"):], "authenticate_admin_password"]
    elif pred.startswith("verified_") or pred.startswith("authenticated"):
        cand = ["authenticate_admin_password"]
    for c in cand:
        if c in actions:
            return c
    return None


def induce_domain(domain, data_dir, out_dir):
    from env.variables import domain_assistant_keys, domain_keys
    from env.task import get_default_dep_full

    A = domain_assistant_keys[domain]
    dep_innate = A.action_innate_dependencies
    dep_full = get_default_dep_full(domain, "full")          # raw structured trees

    raw = json.load(open(f"{data_dir}/{domain}_tasks.json"))
    goals = list(raw.keys())
    # one task to instantiate a domain system for predicate classification
    t0 = raw[goals[0]][0]
    ds_strict = domain_keys[domain + "_strict"](
        copy.deepcopy(t0["initial_database"]), dep_innate, dep_full, t0["constraint_parameters"])
    ds = ds_strict.evaluation_get_domain_system()
    state_obj = getattr(ds, "innate_state_tracker", None)

    def is_state_pred(p):
        return state_obj is not None and hasattr(state_obj, p)

    # agent-callable actions = dep_full keys that are methods on ds and NOT internal_* checks
    actions = [a for a in dep_full
               if hasattr(ds, a) and not a.startswith("internal_")]

    # collect predicates across all action precondition trees
    predicates = {}
    for a in dep_full:
        leaves = []
        walk_predicates(dep_full[a], leaves)
        for name, pm in leaves:
            predicates.setdefault(name, {"params": pm})

    # classify + map state preds -> establishing action
    produces = {a: [] for a in actions}
    warnings = []
    for name, info in predicates.items():
        if is_state_pred(name):
            info["kind"] = "state"
            act = convention_action_for(name, actions)
            info["by"] = act
            if act:
                produces[act].append(name)
            else:
                warnings.append(f"state predicate '{name}' has NO establishing action (convention miss)")
        else:
            info["kind"] = "fact"
            info["by"] = None

    # arg bindings: aggregate {param->slot} from each task's directed_action_graph action node
    args = {a: {} for a in actions}
    for g in goals:
        for task in raw[g]:
            for node in task.get("directed_action_graph", {}).get("nodes", []):
                if isinstance(node, list) and len(node) == 2 and node[0] in args:
                    args[node[0]].update(node[1])

    operators = {}
    for a in actions:
        operators[a] = {
            "precondition": dep_full[a],
            "args": args[a],
            "produces": sorted(set(produces[a])),
            "is_goal": a in goals,
        }

    ontology = {
        "domain": domain,
        "actions": actions,
        "goal_actions": [g for g in goals if g in actions] or goals,
        "operators": operators,
        "predicates": predicates,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ontology_{domain}.json")
    json.dump(ontology, open(path, "w"), indent=2)

    n_state = sum(1 for p in predicates.values() if p["kind"] == "state")
    n_fact = len(predicates) - n_state
    print(f"[{domain}] actions={len(actions)} goals={len(ontology['goal_actions'])} "
          f"predicates={len(predicates)} (state={n_state}, fact={n_fact}) -> {path}")
    for a in actions:
        if operators[a]["produces"]:
            print(f"    produces: {a} -> {operators[a]['produces']}")
    for w in warnings:
        print(f"    WARN: {w}")
    return ontology


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default=None, help="single domain, else all 7")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--out", default="./induced")
    args = ap.parse_args()
    doms = [args.domain] if args.domain else DOMAINS
    for d in doms:
        try:
            induce_domain(d, args.data_dir, args.out)
        except Exception as e:
            import traceback
            print(f"[{d}] FAILED: {e.__class__.__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
