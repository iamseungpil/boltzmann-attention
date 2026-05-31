#!/usr/bin/env python3
"""§8.1 offline lever-ceiling decomposition (no model, no retrain — R7 'verify first').

For each should_T FAILURE, decompose the dirgraph-required verification it missed into:
  A (args-aware callable check): the missing check IS a directly-callable tool node
     (bank: internal_check_username_exist / internal_check_foreign_currency_available).
     A duplicated same-name check with different args (transfer_funds source+dest) is the
     canonical A case — fixable by making the gather (name,args)-keyed.
  B (condition->getter mapping): the constraint leaf is a `condition` predicate with
     by:null in the induced ontology (minimal_eligible_credit_score, sufficient_account_balance,
     no_credit_card_balance_on_card, ...). It is NOT callable; the dirgraph verifies it via a
     GETTER (internal_get_credit_score, get_account_balance, get_credit_card_info, ...). The
     induced ontology left `by:null`, so the means-ends teacher cannot gather it at all. Fixable
     by inducing condition->getter into the ontology (a domain-constant HOW fact, NOT oracle).

Output: leaf-class tally, task-level A-only vs needs-B split, and the condition->getter table
(derived from dirgraph co-occurrence) proving B is inducible. Ceiling with A+B = oracle 40/48.

Usage: python lever_decomp.py <baseline_json> <ontology_bank.json>
"""
import json, sys
from collections import Counter, defaultdict

R = "/home/woori/scratch/SOPBench"
BASE = sys.argv[1] if len(sys.argv) > 1 else R + "/output_v4a_v2/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
ONT = sys.argv[2] if len(sys.argv) > 2 else R + "/induced/ontology_bank.json"
b = json.load(open(BASE)); PREDS = json.load(open(ONT))["predicates"]


def walk(t, a):
    if not t: return
    if isinstance(t, (list, tuple)) and t:
        h = t[0]
        if h == "single":
            nm = t[1]; a.append((nm[4:] if nm.startswith("not ") else nm, t[2] if len(t) > 2 else {}))
        elif h in ("and", "or", "chain", "gate"):
            for s in t[1]: walk(s, a)


def dag_tool_nodes(t):
    return [n[0] for n in t.get("directed_action_graph", {}).get("nodes", []) if isinstance(n, list)]


def main():
    callable_nodes = set()
    for rec in b:
        callable_nodes |= set(dag_tool_nodes(rec["task"]))
    A_only, needs_B = [], []
    kindcnt = Counter(); cond_getter = defaultdict(Counter)
    for i, rec in enumerate(b):
        ev = rec["evaluations"][0]
        if not ev.get("action_should_succeed") or ev.get("success"): continue
        t = rec["task"]; leaves = []; walk(t.get("constraints"), leaves)
        nodes = set(dag_tool_nodes(t)); cls_list = []
        for nm, pm in leaves:
            kind = PREDS.get(nm, {}).get("kind", "?")
            if nm in callable_nodes: cls = "A_callable_check"
            elif kind == "establishable": cls = "handled_est(login/auth)"
            elif kind == "condition": cls = "B_condition"
            else: cls = "other:" + kind
            cls_list.append((nm, cls)); kindcnt[cls] += 1
        has_B = any(c == "B_condition" for _, c in cls_list)
        (needs_B if has_B else A_only).append((i, t["user_goal"]))
        if has_B:
            getters = tuple(sorted(n for n in nodes if n.startswith("internal_get") or n.startswith("get_")))
            for nm, c in cls_list:
                if c == "B_condition": cond_getter[nm][getters] += 1
    print("=== leaf-class tally (44 should_T failures) ===")
    for k, v in kindcnt.most_common(): print(f"  {v:3d}  {k}")
    print(f"\n=== task-level lever split ===\n  A-only (args-aware suffices): {len(A_only)}")
    for i, g in A_only: print(f"      [{i}] {g}")
    print(f"  needs B (>=1 condition w/ by:null): {len(needs_B)}")
    for g, c in Counter(g for _, g in needs_B).most_common(): print(f"      {c:2d} x {g}")
    print("\n=== condition -> candidate getter (dirgraph co-occurrence; B is inducible) ===")
    for cond, gett in cond_getter.items(): print(f"  {cond}: {gett.most_common(2)}")


if __name__ == "__main__":
    main()
