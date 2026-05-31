"""
build_tbox_planner_sft.py — arm-4a SFT data: teach the gating rule (TBox) the 7B IGNORED
in-context (Exp-3v2). Replaces the entangled build_abstract_sft (§11.0/§11.4).

For each task we derive the CORRECT means-ends decision sequence over the induced ABox on the
GROUND-TRUTH domain system (login/authenticate to clear BLOCKED preconditions, then call the
goal; STOP=refuse when an unsatisfiable FACT blocks it). Each decision becomes ONE SFT example:
  input  = the EXACT arm-3v2 planner prompt (shared build_v2_prompt -> train/test identical),
           with READY/BLOCKED computed for the step's established-set.
  target = the next operator NAME (copy-grounded into the in-context tool list) or "STOP".
Only the (short) target is the assistant turn -> supervise target only, mask the prompt
(standard chat SFT). Tool order is SHUFFLED per example (block positional memorization, §11.4);
operator-name ALIAS (lexical guard, design-review B1) is a TODO follow-up (--alias stub).

Anti-entanglement: the target is ALWAYS a name present in the in-context tool list; the model
must read the READY/BLOCKED structure, not memorize "login->apply for bank". ABox stays in the
prompt (swappable) — never baked into weights. Cross-domain: emit per-domain; LODO holds out 1
domain at train time.

RUN (clone root):  python scripts/build_tbox_planner_sft.py --out sft_tbox  [--domain bank]
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from two_stage_client import build_v2_prompt, _render_precond_mod   # shared prompt builder

DOMAINS = ["bank", "dmv", "healthcare", "hotel", "library", "online_market", "university"]


def collect_pred_params(tree, acc):
    """{predicate -> param_mapping} from a dep tree (for resolving internal-check args)."""
    if not tree:
        return
    if isinstance(tree, (list, tuple)) and tree:
        if tree[0] == "single":
            name = tree[1]
            name = name[4:] if name.startswith("not ") else name
            acc[name] = tree[2] if len(tree) > 2 else {}
        elif tree[0] in ("and", "or", "chain", "gate"):
            for s in tree[1]:
                collect_pred_params(s, acc)


def establishable_of(action, ont):
    """{pred -> establishing action} for action's establishable precondition leaves."""
    leaves, est = [], {}
    op = ont["operators"].get(action)
    if op:
        _render_precond_mod(op.get("precondition"), ont.get("predicates", {}), leaves, est)
    return est


def build_domain(domain, data_dir, ont_dir, shuffle_seed_base):
    from env.variables import domain_assistant_keys, domain_keys
    from env.task import task_default_dep_full, task_initializer, get_default_dep_full
    from swarm.util import function_to_json

    ont = json.load(open(f"{ont_dir}/ontology_{domain}.json"))
    di, dfu, dd = task_default_dep_full(domain, "full", "structured", dependency_verb_dep_orig=True)
    dep_innate = domain_assistant_keys[domain].action_innate_dependencies
    dep_full_raw = get_default_dep_full(domain, "full")
    raw = json.load(open(f"{data_dir}/{domain}_tasks.json"))

    def _exit():
        return "Conversation ended."

    examples = []
    idx = 0
    for goal in raw:
        for task in raw[goal]:
            task = dict(task, user_goal=goal)
            _, ui, ai, _ = task_initializer(domain, task, di, dfu, dd, None, "prompt", False, "structured")
            tool_specs = ai["tools"] + [function_to_json(_exit)]
            tool_names = [t["function"]["name"] for t in tool_specs]
            op_descs = {t["function"]["name"]: t["function"].get("description", "") for t in tool_specs}
            user_req = str(task.get("user_prompt", ""))[:300]
            policy = str(ai["instructions"])[:600]

            # GT system: override goal dep with task constraints (match evaluator)
            task_dep = dict(dep_full_raw)
            task_dep[goal] = task["constraints"]
            dss = domain_keys[domain + "_strict"](
                copy.deepcopy(task["initial_database"]), dep_innate, task_dep, task["constraint_parameters"])
            de = dss.evaluation_get_dependency_evaluator()
            slots = dict(task.get("user_known", {}))
            accounts = task["initial_database"].get("accounts", {})
            uname = slots.get("username")
            if uname in accounts and isinstance(accounts[uname], dict):
                for k, v in accounts[uname].items():
                    slots.setdefault(k, v)

            # param_mapping for every predicate in the goal constraint (for check-tool args)
            fact_pm = {}
            collect_pred_params(task["constraints"], fact_pm)

            def resolve_args(a):
                am = ont["operators"].get(a, {}).get("args") or fact_pm.get(a, {})
                return {p: slots[s] for p, s in am.items() if s in slots}

            # goal's FACT preconditions that are directly checkable (callable tool, same name) — v2
            gleaves, gest = [], {}
            _render_precond_mod(task["constraints"], ont.get("predicates", {}), gleaves, gest)
            goal_fact_checkable = [p for p in dict.fromkeys(gleaves)
                                   if p not in gest and p in tool_names]

            established, history, executed, observed = set(), [], set(), {}

            def next_decision():
                # 1. gather: verify any unobserved checkable fact of the goal first
                for p in goal_fact_checkable:
                    if p not in observed and p not in executed:
                        return p
                # 2. a verified fact is FALSE -> refuse
                if any(observed.get(p) is False for p in goal_fact_checkable):
                    return "STOP"
                # 3. goal reachable?
                try:
                    if de.process(goal, **slots):
                        return goal
                except Exception:
                    pass
                # 4. BFS for a callable establishing action that advances an unmet precond
                frontier, seen = [goal], set()
                while frontier:
                    a = frontier.pop(0)
                    if a in seen:
                        continue
                    seen.add(a)
                    for pred, by in establishable_of(a, ont).items():
                        if pred in established or by in executed:
                            continue
                        try:
                            ok = de.process(by, **slots)
                        except Exception:
                            ok = False
                        if ok:
                            return by
                        frontier.append(by)
                return "STOP"

            for _ in range(12):
                # SHUFFLE tool order per step (deterministic from idx; no Math.random in env)
                order = list(range(len(tool_names)))
                s = (shuffle_seed_base + idx * 7 + len(history)) % max(1, len(order))
                shown = tool_names[s:] + tool_names[:s]
                prompt = build_v2_prompt(ont, shown, established, user_req, policy,
                                         list(history), set(slots.keys()), op_descs, observed)
                target = next_decision()
                examples.append({
                    "domain": domain, "goal": goal,
                    "messages": [{"role": "user", "content": prompt},
                                 {"role": "assistant", "content": target}],
                })
                idx += 1
                if target in ("STOP", goal) or target in executed:
                    break
                executed.add(target)
                # advance GT state by executing the establishing action
                try:
                    r = getattr(dss, target)(**resolve_args(target))
                except Exception as e:
                    r = f"{e.__class__.__name__}"
                history.append(f"CALLED: {target}")
                history.append(f"RESULT[{target}]: {str(r)[:60]}")
                if r is not False and "Error" not in str(r):
                    established.update(ont["operators"].get(target, {}).get("produces", []))
                # fact-visibility: record a checkable fact's observed bool (v2)
                if target in goal_fact_checkable:
                    v = r[1] if isinstance(r, (list, tuple)) and len(r) >= 2 else r
                    observed[target] = bool(v) if isinstance(v, (bool, int)) else False
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default=None)
    ap.add_argument("--out", default="./sft_tbox")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--ont_dir", default="./induced")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    doms = [args.domain] if args.domain else DOMAINS
    all_ex = []
    for d in doms:
        try:
            ex = build_domain(d, args.data_dir, args.ont_dir, shuffle_seed_base=13)
        except Exception as e:
            import traceback
            print(f"[{d}] FAILED: {e.__class__.__name__}: {e}")
            traceback.print_exc()
            continue
        path = os.path.join(args.out, f"sft_tbox_{d}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for e in ex:
                f.write(json.dumps(e) + "\n")
        from collections import Counter
        tgt = Counter(e["messages"][1]["content"] if e["messages"][1]["content"] in ("STOP",)
                      else ("GOAL" if e["messages"][1]["content"] == e["goal"] else "establish")
                      for e in ex)
        print(f"[{d}] {len(ex)} examples -> {path}  | target mix: {dict(tgt)}")
        all_ex.extend(ex)
    print(f"TOTAL: {len(all_ex)} examples across {len(doms)} domains")


if __name__ == "__main__":
    main()
