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


def collect_leaf_list(tree, acc):
    """ARGS-AWARE leaf list [(pred, param_map, negated)] preserving same-name duplicates
    (e.g. internal_check_username_exist on source AND destination)."""
    if not tree or not isinstance(tree, (list, tuple)) or not tree:
        return
    if tree[0] == "single":
        nm = tree[1]; neg = nm.startswith("not ")
        acc.append((nm[4:] if neg else nm, tree[2] if len(tree) > 2 else {}, neg))
    elif tree[0] in ("and", "or", "chain", "gate"):
        for s in tree[1]:
            collect_leaf_list(s, acc)


# (b §8.5.3) condition predicate -> verifying GETTER tool, per domain. A `condition` predicate
# (kind=condition, by:null) is not directly callable; the agent calls the GETTER and a deterministic
# resolver compares the result to the threshold. Derived from directed_action_graph co-occurrence
# (lever_decomp). bank verified; other domains derived the same way (TODO: auto-derive per domain).
GETTER_BY_DOMAIN = {
    "bank": {
        "minimal_elgibile_credit_score": "internal_get_credit_score",
        "sufficient_account_balance": "get_account_balance",
        "no_credit_card_balance_on_card": "get_credit_card_info",
        "not_over_credit_limit": "get_credit_card_info",
        "internal_check_credit_card_exist": "get_credit_card_info",
        "get_loan_owed_balance_restr": "get_account_owed_balance",
        "pay_loan_account_balance_restr": "get_account_balance",
        "pay_loan_amount_restr": "get_account_balance",
        "safety_box_eligible": "get_account_balance",
    },
}


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

            # ---- (b §8.5) COMPLETE verification gather + reason-based STOP (should_T & should_F) ----
            GETTER = GETTER_BY_DOMAIN.get(domain, {})
            cleaves = []
            collect_leaf_list(task["constraints"], cleaves)        # args-aware constraint leaves
            # required fact/condition verifications: callable check -> itself; condition -> its getter.
            required = []   # (pred, param_map, negated, tool_to_call)
            for pred, pm, neg in cleaves:
                kind = ont["predicates"].get(pred, {}).get("kind")
                if kind == "establishable":
                    continue                                       # login/auth handled in establish phase
                if pred in tool_names:
                    required.append((pred, pm, neg, pred))         # A: args-aware callable check
                elif pred in GETTER and GETTER[pred] in tool_names:
                    required.append((pred, pm, neg, GETTER[pred]))  # B: condition -> getter+compare
            # C: goal's establishable login/auth, CONDITIONAL on credential availability (no halluc)
            gl = []
            collect_leaf_list(ont["operators"].get(goal, {}).get("precondition"), gl)
            ests = []
            for pred, pm, neg in gl:
                info = ont["predicates"].get(pred, {})
                by = info.get("by")
                if info.get("kind") == "establishable" and by and by in tool_names:
                    am = ont["operators"].get(by, {}).get("args") or {}
                    if set(am.values()).issubset(set(slots.keys())):     # creds present in user_known
                        ests.append((pred, by))

            def truth(pred, pm, neg):
                """Authoritative truth of a constraint leaf on the GT strict system (resolver = deterministic)."""
                try:
                    return bool(de._process(("single", ("not " if neg else "") + pred, pm), **slots))
                except Exception:
                    return None

            established, history, executed, observed = set(), [], set(), {}

            def next_decision():
                # 1. gather: call each required fact/condition verification tool (args-aware)
                for pred, pm, neg, tool in required:
                    if tool not in executed:
                        return tool
                # 2. a verified constraint fact/condition is FALSE -> refuse (reason now gathered) [should_F]
                if any(observed.get(p) is False for p, _, _, _ in required):
                    return "STOP"
                # 3. establish login/auth (only when creds available) for the goal
                for pred, by in ests:
                    if pred in established or by in executed:
                        continue
                    return by
                # 4. goal reachable on the GT system? [should_T]
                try:
                    if de.process(goal, **slots):
                        return goal
                except Exception:
                    pass
                return "STOP"

            _seq = []
            for _ in range(16):
                # SHUFFLE tool order per step (deterministic from idx; no Math.random in env)
                order = list(range(len(tool_names)))
                s = (shuffle_seed_base + idx * 7 + len(history)) % max(1, len(order))
                shown = tool_names[s:] + tool_names[:s]
                prompt = build_v2_prompt(ont, shown, established, user_req, policy,
                                         list(history), set(slots.keys()), op_descs, observed)
                target = next_decision()
                _seq.append(target)
                examples.append({
                    "domain": domain, "goal": goal,
                    "messages": [{"role": "user", "content": prompt},
                                 {"role": "assistant", "content": target}],
                })
                idx += 1
                if target in ("STOP", goal) or target in executed:
                    break
                executed.add(target)
                # advance GT state by executing the chosen tool
                try:
                    r = getattr(dss, target)(**resolve_args(target))
                except Exception as e:
                    r = f"{e.__class__.__name__}"
                history.append(f"CALLED: {target}")
                history.append(f"RESULT[{target}]: {str(r)[:60]}")
                if r is not False and "Error" not in str(r):
                    established.update(ont["operators"].get(target, {}).get("produces", []))
                # resolver: reveal the deterministic truth of every constraint leaf this tool verifies
                for pred, pm, neg, tool in required:
                    if tool == target and pred not in observed:
                        observed[pred] = truth(pred, pm, neg)
            if os.environ.get("SFT_TRACE"):
                ss = "T" if task.get("action_should_succeed") else "F"
                print(f"[{ss}] {goal:22} req={[t for _,_,_,t in required]} ests={[b for _,b in ests]} "
                      f"obs={observed} seq={_seq}", file=sys.stderr)
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
