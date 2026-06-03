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
from two_stage_client import (build_v2_prompt, _render_precond_mod,   # shared prompt builder
                              make_alias_map)

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


def treeval_expr(tree, observed, amap=None):
    """v3 grounded tree-eval (RUNG1_V3_TREE_EVAL_LITREVIEW): render the constraint TREE with each
    leaf replaced by its GATHERED truth (observed lookup, not a cold guess) and recursively evaluate
    AND/OR/chain. Returns (expr_str, value). value=None if a leaf truth is missing (ungathered).
    expr e.g. 'AND(credit=true, OR(b=false, c=true))'. amap aliases predicate names (anti-cheat)."""
    A = (lambda n: amap.get(n, n)) if amap else (lambda n: n)
    if not tree or not isinstance(tree, (list, tuple)) or not tree:
        return ("true", True)
    h = tree[0]
    if h == "single":
        nm = tree[1]; neg = nm.startswith("not "); base = nm[4:] if neg else nm
        pm = tree[2] if len(tree) > 2 else {}
        v = observed.get((base, frozenset((pm or {}).items())))   # (pred,args) key: source/dest distinct
        vv = "true" if v is True else ("false" if v is False else "unknown")
        return (f"{A(base)}={vv}", v)
    if h in ("and", "chain", "gate"):
        ch = [treeval_expr(s, observed, amap) for s in tree[1]]
        vals = [c[1] for c in ch]
        val = (all(x is True for x in vals) if vals else True)
        if any(x is None for x in vals) and not any(x is False for x in vals):
            val = None                                  # unknown leaf, not decidable false
        return ("AND(" + ", ".join(c[0] for c in ch) + ")", val)
    if h == "or":
        ch = [treeval_expr(s, observed, amap) for s in tree[1]]
        vals = [c[1] for c in ch]
        if any(x is True for x in vals):
            val = True
        elif any(x is None for x in vals):
            val = None
        else:
            val = False
        return ("OR(" + ", ".join(c[0] for c in ch) + ")", val)
    return ("true", True)


def treeval_reduce(tree, observed, amap=None):
    """v3-INDUCTIVE (litreview §4 / Abbe NeurIPS24 inductive scratchpad + Kim&Suzuki ICML25
    intermediate-step loss): instead of emitting the whole nested expression in ONE step
    (`gate = AND(op_a, AND(op_b, op_c)) = val`, which our Exp-4-rung1-v3-AB retest showed converges
    but gives BOTH==control = single-step globality limit), emit a BOTTOM-UP REDUCTION CHAIN where
    each internal node folds its ALREADY-RESOLVED children into a named partial result. Each `tK=OP(..)=v`
    is a LOCAL step (combines 2-3 known values) and is in the SFT target -> serial compute + supervision.
    Leaves use the GATHERED truth lookup (identical semantics to treeval_expr; no re-derivation).
    Returns (chain_str, value); chain_str e.g. 't1=AND(op_39=true, op_25=true)=true; t2=AND(op_32=false, t1)=false'."""
    A = (lambda n: amap.get(n, n)) if amap else (lambda n: n)
    steps = []
    counter = [0]

    def rec(t):
        # returns (ref, value): ref is a leaf token 'name=val' or a temp 'tK' for an internal node.
        if not t or not isinstance(t, (list, tuple)) or not t:
            return ("true", True)
        h = t[0]
        if h == "single":
            nm = t[1]; neg = nm.startswith("not "); base = nm[4:] if neg else nm
            pm = t[2] if len(t) > 2 else {}
            v = observed.get((base, frozenset((pm or {}).items())))
            vv = "true" if v is True else ("false" if v is False else "unknown")
            return (f"{A(base)}={vv}", v)
        if h in ("and", "chain", "gate", "or"):
            kids = [rec(s) for s in t[1]]
            vals = [k[1] for k in kids]
            if h == "or":
                if any(x is True for x in vals):
                    val = True
                elif any(x is None for x in vals):
                    val = None
                else:
                    val = False
                opn = "OR"
            else:
                val = (all(x is True for x in vals) if vals else True)
                if any(x is None for x in vals) and not any(x is False for x in vals):
                    val = None
                opn = "AND"
            counter[0] += 1
            tk = f"t{counter[0]}"
            vstr = "true" if val is True else ("false" if val is False else "unknown")
            steps.append(f"{tk}={opn}(" + ", ".join(k[0] for k in kids) + f")={vstr}")
            return (tk, val)
        return ("true", True)

    ref, val = rec(tree)
    chain = "; ".join(steps) if steps else ref   # single-leaf tree -> just the leaf truth
    return (chain, val)


# (T1b, handoff §1.5-(2)) DEPRECATED hand-map. The teacher NO LONGER uses this for required_set —
# condition->getter is now the auto-derived `getter_map.json` (GMAP, structural AST parse,
# autoderive_getter_map.py) on ALL 7 domains uniformly. Retained ONLY as the ground-truth that
# autoderive_getter_map.py validates its bank recall against (imported there). Do NOT add domains here.
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


def build_domain(domain, data_dir, ont_dir, shuffle_seed_base, use_alias=False, source=1,
                 use_gate=False, use_scratch=False, use_treeval=False, use_treeval_inductive=False):
    from env.variables import domain_assistant_keys, domain_keys
    from env.task import task_default_dep_full, task_initializer, get_default_dep_full
    from swarm.util import function_to_json

    ont = json.load(open(f"{ont_dir}/ontology_{domain}.json"))
    # auto-derived condition->getter-SET (autoderive_getter_map.py; structural, multi-getter,
    # transfer-clean). Replaces the hand GETTER_BY_DOMAIN single-getter map. Falls back to hand
    # map if absent. See EXPERIMENT_DESIGN Rung1 priority-lock / Exp-4-precheck-FINAL.
    _gmap_path = f"{ont_dir}/getter_map.json"
    GMAP = json.load(open(_gmap_path)).get(domain, {}) if os.path.exists(_gmap_path) else {}
    di, dfu, dd = task_default_dep_full(domain, "full", "structured", dependency_verb_dep_orig=True)
    dep_innate = domain_assistant_keys[domain].action_innate_dependencies
    dep_full_raw = get_default_dep_full(domain, "full")
    raw = json.load(open(f"{data_dir}/{domain}_tasks.json"))

    def _exit():
        return "Conversation ended."

    examples = []
    idx = 0
    task_no = 0
    for goal in raw:
        for task in raw[goal]:
            task = dict(task, user_goal=goal)
            _, ui, ai, _ = task_initializer(domain, task, di, dfu, dd, None, "prompt", False, "structured")
            tool_specs = ai["tools"] + [function_to_json(_exit)]
            tool_names = [t["function"]["name"] for t in tool_specs]
            op_descs = {t["function"]["name"]: t["function"].get("description", "") for t in tool_specs}
            # §8.5.★ ①: per-task alias map (TRAIN salt; eval re-randomizes -> tests generalization).
            amap = None
            if use_alias:
                terms = set(ont["operators"]) | set(ont["predicates"]) | set(tool_names)
                amap = make_alias_map(terms, salt=f"train|{domain}|{task_no}")
            task_no += 1
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

            # ---- T1 (RUNG1): UNIFORM required_set — login is an ORDINARY member, NOT special-cased.
            #  source = task["constraints"] leaves  ∪  goal-default establishable (evaluator-authoritative;
            #  handoff §2 GATE=DIFFERENT: 98/830 goals need login innately, absent from constraints ->
            #  constraints-only would UNDER-login -> goal precond unmet -> fail). The authoritative source
            #  is the SAME one precheck_required_source.py measured: dep_innate[goal] ∪ dep_full_raw[goal]
            #  ∪ ops[goal]["precondition"]. Mapping is ABox-driven & domain-general: check leaf -> itself;
            #  condition -> its getter(s) (auto getter_map only; hand GETTER_BY_DOMAIN dropped = T1b);
            #  establishable -> its `by` action. No creds heuristic, no per-tool/per-domain branch, no
            #  separate establish phase. `is_est` marks establishment tools (satisfied by `produces`, not a
            #  read-truth) so they are GATHERED but excluded from the preconds_verified truth-AND.
            cleaves = []
            collect_leaf_list(task["constraints"], cleaves)        # args-aware constraint leaves
            gleaves = []                                           # goal-default / innate establishable
            collect_leaf_list(dep_innate.get(goal), gleaves)
            collect_leaf_list(dep_full_raw.get(goal), gleaves)
            collect_leaf_list(ont["operators"].get(goal, {}).get("precondition"), gleaves)
            con_preds = {p for p, _, _ in cleaves}
            required = []   # (pred, param_map, negated, tool_to_call, is_est)
            _seen_rt = set()

            def _add_req(pred, pm, neg, tool, is_est):
                if tool in tool_names and (pred, tool) not in _seen_rt:
                    _seen_rt.add((pred, tool))
                    required.append((pred, pm, neg, tool, is_est))

            # 1. establishment tools FIRST (login/auth before the reads that depend on them): constraint
            #    establishables + goal-default establishables absent from constraints (evaluator-authoritative).
            for pred, pm, neg in list(cleaves) + [g for g in gleaves if g[0] not in con_preds]:
                info = ont["predicates"].get(pred, {})
                if info.get("kind") == "establishable" and info.get("by"):
                    _add_req(pred, pm, neg, info["by"], True)
            # 2. then constraint checks / condition-getters (uniform routing).
            argonly = []   # conditions evaluable from the REQUEST (no tool/getter): autoderive arg-only.
            for pred, pm, neg in cleaves:
                info = ont["predicates"].get(pred, {})
                if info.get("kind") == "establishable":
                    continue                                       # already added as establishment (step 1)
                if pred in tool_names:
                    _add_req(pred, pm, neg, pred, False)           # A: args-aware callable check
                elif pred in GMAP and any(g in tool_names for g in GMAP[pred]):
                    for g in GMAP[pred]:                            # B: condition -> getter-SET (auto, multi)
                        _add_req(pred, pm, neg, g, False)
                else:
                    argonly.append((pred, pm, neg))                # C: arg-only -> known from request (inject truth)

            def truth(pred, pm, neg):
                """Authoritative truth of a constraint leaf on the GT strict system (resolver = deterministic)."""
                try:
                    return bool(de._process(("single", ("not " if neg else "") + pred, pm), **slots))
                except Exception:
                    return None

            established, history, executed, observed = set(), [], set(), {}
            # Point 1: inject arg-only condition truths (no tool needed; model evaluates from request)
            # so treeval grounds them instead of seeing 'unknown'. (autoderive: 100% getter-set OR arg-only.)
            for _p, _pm, _neg in argonly:
                observed[(_p, frozenset((_pm or {}).items()))] = truth(_p, _pm, _neg)
            observed_goal_ok = False                              # T2: goal called AND succeeded (post-success exit)
            est_tools = {t for _p, _pm, _n, t, ie in required if ie}   # v3: establishment tool names
            est_failed = False                                    # v3: any required establishable failed to establish
            _tv = {}                                              # v3 treeval: terminal (expr, value) for emit
            should_succeed = bool(task.get("action_should_succeed", True))

            # ★COMPLETENESS CENSUS (v3 gate, SFT_CENSUS=1): does AND(gatherable conditions) ∧ reachable
            #  == should_succeed? Mismatch = the set wait CANNOT fix (ungatherable pure-policy rule / OR
            #  structure / missing condition) = v3 ceiling. Dual of v2's preconds_verified=false;ACT==0.
            if os.environ.get("SFT_CENSUS"):
                _lt = [truth(p, pm, neg) for p, pm, neg, _t, ie in required if not ie]
                _nest = sum(1 for *_x, ie in required if ie)
                try:
                    _reach = bool(de.process(goal, **slots))
                except Exception:
                    _reach = False
                # NOTE: _reach = de.process(goal) on the FROZEN pre-login state -> spurious 0 for
                # login-needing tasks (census diagnostic limitation; needs establishment-simulation to
                # be meaningful). Teacher decision no longer uses this (terminal = should_succeed).
                _modeled = (all(t is True for t in _lt) if _lt else True) and _reach
                print(f"CENSUS\t{domain}\t{goal}\tshould={int(should_succeed)}\tmodeled={int(_modeled)}"
                      f"\treach={int(_reach)}\tnleaf={len(_lt)}\tnest={_nest}\tlt={_lt}",
                      file=sys.stderr)
                continue

            def next_decision():
                # 1. gather: call each required tool (establishment + checks), args-aware, UNIFORM
                #    (login is just an early required member now — no separate establish phase).
                for pred, pm, neg, tool, is_est in required:
                    if tool not in executed:
                        return tool
                # 2. T2 (R3): post-success termination. goal already called & succeeded -> exit (DONE).
                if observed_goal_ok:
                    return "DONE"
                # 3. ★v3 TERMINAL = GROUNDED tree-eval over GATHERED leaf truths (RUNG1_V3 litreview#1).
                #    permitted is no longer a cold should_succeed guess -> it is AND/OR/chain over the
                #    OBSERVED constraint-leaf truths (+ establishment status), so it cannot cold-collapse
                #    and OR is handled correctly. gate_val SHOULD == should_succeed (SFT_TRACE census).
                if use_treeval:
                    obs2 = dict(observed)
                    for _p, _pm, _n, _t, _ie in required:
                        _k = (_p, frozenset((_pm or {}).items()))
                        if _ie and _k not in obs2:
                            # establishable leaf: on should_T trust GT (creds establishable; sim cred
                            # artifacts ignored); on should_F use real established status.
                            obs2[_k] = True if should_succeed else (_p in established)
                    expr, tv = treeval_expr(task["constraints"], obs2, amap)
                    disp = (tv is True) and (not est_failed)     # grounded gate value (for display)
                    _tv["expr"], _tv["val"] = expr, disp
                    if use_treeval_inductive:
                        # bottom-up reduction chain (each internal node folded as a supervised step).
                        # Keep chain_val (treeval_reduce's final fold) so the emit can guarantee the
                        # chain's last value == the stated gate (avoids the est_failed-flip inconsistency
                        # where disp!=tv would print a chain ending =true under gate=false).
                        chain, chain_val = treeval_reduce(task["constraints"], obs2, amap)
                        _tv["chain"], _tv["chain_val"] = chain, chain_val
                    return "ACT" if should_succeed else "STOP"   # decision = authoritative GT label
                # 3'. (legacy non-treeval) should_F gathered-false -> STOP; terminal = should_succeed label.
                if not should_succeed and any(observed.get((p, frozenset((pm or {}).items()))) is False
                                              for p, pm, _, _, ie in required if not ie):
                    return "STOP"
                if use_gate:
                    return "ACT" if should_succeed else "STOP"
                return goal if should_succeed else "STOP"

            _seq = []
            for _ in range(16):
                # SHUFFLE tool order per step (deterministic from idx; no Math.random in env)
                order = list(range(len(tool_names)))
                s = (shuffle_seed_base + idx * 7 + len(history)) % max(1, len(order))
                shown = tool_names[s:] + tool_names[:s]
                prompt = build_v2_prompt(ont, shown, established, user_req, policy,
                                         list(history), set(slots.keys()), op_descs, observed,
                                         alias_map=amap, source=source, gate_token=use_gate,
                                         scratchpad=use_scratch)
                target = next_decision()
                _seq.append(target)
                # §8.7 Rung1 educated scratchpad: terminal target = TWO-gate AND + branch.
                # gather steps unchanged. (review B-3/B-4: a single all_verified conflated checks
                # and policy; split into preconds_verified + permitted, ACT iff both.)
                if use_scratch:
                    # §3 Rung1 ①: per-step readiness gate. tool step -> ready=false; <tool> (never ACT);
                    # terminal -> ready=true; preconds_verified=<checks>; permitted=<policy>; <ACT|STOP>.
                    if target == "DONE":
                        # T2 (R3) post-success exit. Distinct from refuse-STOP via the `done` flag so the
                        # model learns "goal succeeded in history -> terminate" (not a policy refusal).
                        tgt_out = "ready=true; done=true; STOP"
                    elif use_treeval and target in ("ACT", "STOP"):
                        # ★v3 grounded derivation: per-leaf truths aggregated by the constraint TREE
                        # (AND/OR/chain) over GATHERED values -> gate. Not a cold guess. (litreview#1)
                        # Emit grounded form ONLY if it matches the GT decision; else (slot/cred-confounded
                        # tail ~14%) fall back to the non-grounded permitted token to avoid contradiction.
                        if use_treeval_inductive:
                            # §4 inductive: emit the bottom-up reduction chain ONLY when its final fold
                            # value matches the GT decision (ACT<->true, STOP<->false) — guarantees the
                            # chain ends at the stated gate (internally consistent). Else (chain_val None/
                            # unknown, or est_failed/policy diverges) fall back to the non-grounded token.
                            cv = _tv.get("chain_val")
                            if (cv is True and target == "ACT") or (cv is False and target == "STOP"):
                                _ch = _tv.get("chain", "true")
                                _seg = f"{_ch}; " if "=" in _ch else ""   # drop bare trivial literal (empty-constraint task)
                                tgt_out = (f"ready=true; {_seg}"
                                           f"gate={'true' if cv is True else 'false'}; {target}")
                            else:
                                tgt_out = (f"ready=true; preconds_verified=true; "
                                           f"permitted={'true' if target == 'ACT' else 'false'}; {target}")
                        elif bool(_tv.get("val")) == (target == "ACT"):
                            tgt_out = (f"ready=true; gate = {_tv.get('expr', 'true')} = "
                                       f"{'true' if _tv.get('val') else 'false'}; {target}")
                        else:
                            tgt_out = (f"ready=true; preconds_verified=true; "
                                       f"permitted={'true' if target == 'ACT' else 'false'}; {target}")
                    elif target in ("ACT", "STOP"):
                        # Two-gate decomposition (review B-3/B-4, 2026-06-01). A single `all_verified`
                        # token cannot represent both gates: refusal happens when a required CHECK
                        # fails AND when checks pass but POLICY forbids. The old av=AND(checks) gave
                        # `all_verified=true; STOP` (380/473) contradicting "ACT iff true"; the naive
                        # av=(target==ACT) fix removed the contradiction but made av a pure echo of the
                        # decision (zero decomposition -> scaffold collapse). Split into two INDEPENDENT
                        # predictions whose AND is the decision; ACT iff (preconds_verified AND permitted).
                        _ro = [observed.get((p, frozenset((pm or {}).items()))) for p, pm, _, _, ie in required
                               if not ie and (p, frozenset((pm or {}).items())) in observed]
                        pv = all(v is True for v in _ro) if _ro else True   # required checks verified (gather-state)
                        # permitted = GT outcome (should_succeed); encodes policy + OR + login-block.
                        # (dropped `and _reach`: de.process on the frozen pre-login state wrongly blocked
                        # login-needing should_T -> permitted=false on should-succeed tasks.)
                        pm = bool(should_succeed)                           # policy permits (GT label)
                        # consistency: ACT iff (pv and pm); STOP explained by which gate is false.
                        tgt_out = (f"ready=true; preconds_verified={'true' if pv else 'false'}; "
                                   f"permitted={'true' if pm else 'false'}; {target}")
                    else:
                        _tool = amap.get(target, target) if amap else target
                        tgt_out = f"ready=false; {_tool}"
                elif target == "DONE":
                    tgt_out = "STOP"                       # non-scratch post-success exit (parser: terminate)
                elif amap is None or target in ("STOP", "ACT"):
                    tgt_out = target                      # constants stay literal
                else:
                    tgt_out = amap.get(target, target)    # alias the tool name
                tgt_kind = ("STOP" if target == "STOP" else "done" if target == "DONE"
                            else "GOAL" if target in (goal, "ACT") else "establish")
                examples.append({
                    "domain": domain, "goal": goal, "target_kind": tgt_kind,
                    "messages": [{"role": "user", "content": prompt},
                                 {"role": "assistant", "content": tgt_out}],
                })
                idx += 1
                if target in ("STOP", "DONE"):
                    break
                # T2: execute the chosen tool (incl. the goal on ACT) and CONTINUE so a post-success DONE
                #     step can be emitted next iter. We no longer break on goal/ACT (that erased the exit).
                exec_target = goal if target == "ACT" else target
                if exec_target in executed:               # infinite-loop guard RETAINED (handoff T2)
                    break
                executed.add(exec_target)
                try:
                    r = getattr(dss, exec_target)(**resolve_args(exec_target))
                except Exception as e:
                    r = f"{e.__class__.__name__}"
                history.append(f"CALLED: {exec_target}")
                history.append(f"RESULT[{exec_target}]: {str(r)[:60]}")
                _ok = (r is not False and "Error" not in str(r))
                if exec_target in est_tools and not _ok and not should_succeed:
                    est_failed = True                            # v3: establishment block counts only for should_F
                if _ok:
                    established.update(ont["operators"].get(exec_target, {}).get("produces", []))
                if exec_target == goal:
                    observed_goal_ok = _ok                # T2: gate the post-success DONE on real success
                # resolver: reveal the deterministic truth of every constraint leaf this tool verifies
                #           (checks only; establishables are tracked via `established`/`produces`).
                for pred, pm, neg, tool, is_est in required:
                    _k = (pred, frozenset((pm or {}).items()))   # (pred,args) key: same-pred-multi-arg distinct
                    if not is_est and tool == exec_target and _k not in observed:
                        observed[_k] = truth(pred, pm, neg)
            # T2 build-time assertion: DONE, if present, must be the terminal step (no tool-call after a
            # successful goal). Guards the loop restructure (handoff T2).
            assert "DONE" not in _seq or _seq.index("DONE") == len(_seq) - 1, \
                f"DONE not terminal in {domain}/{goal}: {_seq}"
            if os.environ.get("SFT_TRACE"):
                ss = "T" if task.get("action_should_succeed") else "F"
                _reqt = [t for _, _, _, t, _ in required]
                _estt = [t for _, _, _, t, ie in required if ie]
                _tvinfo = ""
                if use_treeval and "val" in _tv:
                    _agree = (bool(_tv["val"]) == should_succeed)
                    _tvinfo = f" | treeval={_tv['val']} ss={should_succeed} agree={_agree} expr={_tv['expr']}"
                print(f"[{ss}] {goal:22} req={_reqt} est={_estt} "
                      f"obs={observed} seq={_seq}{_tvinfo}", file=sys.stderr)
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default=None)
    ap.add_argument("--out", default="./sft_tbox")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--ont_dir", default="./induced")
    ap.add_argument("--alias", action="store_true",
                    help="§8.5.★ ①: mask tool/predicate names as opaque per-task aliases (anti-cheat)")
    ap.add_argument("--source", type=int, default=1, choices=[1, 3],
                    help="1=render constraint-derived STATUS (answer key); 3=NL policy only")
    ap.add_argument("--gate-token", action="store_true",
                    help="§8.6: terminal target = constant ACT/STOP (not the varying goal name); "
                         "should_F forced to STOP. Fixes the act-vs-STOP gate asymmetry.")
    ap.add_argument("--scratchpad", action="store_true",
                    help="§8.7 Rung1: educated inductive scratchpad — terminal target emits the "
                         "AND-aggregation token `all_verified=<true|false>; <ACT|STOP>` (implies gate). "
                         "Grounded in Abbe NeurIPS24 / Kim&Suzuki ICML25.")
    ap.add_argument("--treeval", action="store_true",
                    help="§3.10 v3: terminal = GROUNDED tree-eval derivation `gate = AND/OR(leaf=<obs>...) "
                         "= <bool>; <ACT|STOP>` over gathered leaf truths (replaces cold permitted). "
                         "Implies --scratchpad/--gate-token. RUNG1_V3_TREE_EVAL_LITREVIEW.")
    ap.add_argument("--treeval_inductive", action="store_true",
                    help="§4 v3-inductive: terminal = BOTTOM-UP REDUCTION CHAIN `t1=AND(a=t,b=t)=t; "
                         "t2=AND(t1,c=f)=f; gate=f; <ACT|STOP>` — each internal node folded as a supervised "
                         "step (Abbe inductive scratchpad + Kim&Suzuki intermediate-loss). Replaces the "
                         "single-step whole-expression emit (Exp-4-rung1-v3-AB: single-step converges but "
                         "BOTH==control). Implies --treeval.")
    args = ap.parse_args()
    if args.treeval_inductive:
        args.treeval = True       # inductive is a treeval variant (shares the grounded terminal branch)
    if args.treeval:
        args.scratchpad = True    # treeval terminal lives in the scratchpad emit branch
    if args.scratchpad and not args.gate_token:
        args.gate_token = True   # scratchpad terminal is ACT/STOP -> requires gate-token vocabulary
    os.makedirs(args.out, exist_ok=True)
    doms = [args.domain] if args.domain else DOMAINS
    all_ex = []
    for d in doms:
        try:
            ex = build_domain(d, args.data_dir, args.ont_dir, shuffle_seed_base=13,
                              use_alias=args.alias, source=args.source, use_gate=args.gate_token,
                              use_scratch=args.scratchpad, use_treeval=args.treeval,
                              use_treeval_inductive=args.treeval_inductive)
        except Exception as e:
            import traceback
            print(f"[{d}] FAILED: {e.__class__.__name__}: {e}")
            traceback.print_exc()
            continue
        tag = ("_alias" if args.alias else "") + (f"_s{args.source}" if args.source != 1 else "") \
            + ("_gate" if args.gate_token else "") + ("_scratch" if args.scratchpad else "") \
            + ("_treeval" if args.treeval else "") + ("ind" if args.treeval_inductive else "")
        path = os.path.join(args.out, f"sft_tbox_{d}{tag}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for e in ex:
                f.write(json.dumps(e) + "\n")
        from collections import Counter
        tgt = Counter(e.get("target_kind", "?") for e in ex)
        print(f"[{d}] {len(ex)} examples -> {path}  | target mix: {dict(tgt)}")
        all_ex.extend(ex)
    print(f"TOTAL: {len(all_ex)} examples across {len(doms)} domains")


if __name__ == "__main__":
    main()
