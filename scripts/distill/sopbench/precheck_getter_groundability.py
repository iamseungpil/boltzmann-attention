#!/usr/bin/env python3
"""precheck_getter_groundability.py — DECISIVE pre-check (EXPERIMENT_DESIGN Rung1 priority-lock).

Settles "synthetic-first vs data-fix-first" empirically. For every DISCRIMINATING condition
predicate across all 7 SOPBench domains, decide whether an EXISTING getter tool can resolve it:

  A_callable   : the predicate name is itself a callable tool (directly verifiable).
  B_getter     : a getter-like tool (get_*/internal_get*/internal_check*) shares a CONTENT token
                 with the condition name -> auto-derivable condition->getter link (the Track-A fix).
  UNGROUNDABLE : no callable, no token-overlapping getter -> a pure policy rule (e.g. "US resident")
                 = the genuine residual that ONLY synthetic/other can address (= v3/synth scope).

Also VALIDATES the auto-derive rule: for bank, compare the token-overlap map against the
hand-maintained GETTER_BY_DOMAIN (build_tbox_planner_sft) -> precision/recall. High agreement =>
auto-derive is trustworthy => Track A (auto-derive getter map) is the right primary fix.

Outputs per domain: #conditions, #A, #B, #UNGROUNDABLE + the ungroundable list, and the
proposed condition->getter table. Run from the SOPBench clone root (needs env + ./induced).

  RUN (clone root):
    python scripts/distill/sopbench/precheck_getter_groundability.py \
       --ont_dir ./induced --data_dir ./data/tasks   # adjust to remote layout
"""
import argparse, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_tbox_planner_sft import (DOMAINS, GETTER_BY_DOMAIN, collect_leaf_list)

# tokens that carry no discriminative meaning when matching condition<->getter names.
STOP = {
    "minimal", "eligible", "elgibile", "sufficient", "enough", "no", "not", "is", "has",
    "restr", "within", "over", "under", "valid", "exist", "exists", "available", "check",
    "get", "internal", "the", "of", "a", "an", "on", "in", "to", "and", "or", "for",
    "account", "user", "current", "amount", "info", "status", "value", "number",
}


def toks(name):
    return [t for t in name.replace("-", "_").lower().split("_") if t and t not in STOP]


def is_getter(name):
    n = name.lower()
    return n.startswith("get_") or n.startswith("internal_get") or n.startswith("internal_check")


def best_getter(cond, getters):
    """(getter, shared_tokens) maximizing content-token overlap; ('', []) if none share a token."""
    cset = set(toks(cond))
    best, best_sh = "", set()
    for g in getters:
        sh = cset & set(toks(g))
        if len(sh) > len(best_sh):
            best, best_sh = g, sh
    return best, sorted(best_sh)


def build_domain_tools(domain, data_dir):
    """All tool names seen across the domain's tasks (full-tool mode) via the live env."""
    from env.variables import domain_assistant_keys, domain_keys  # noqa
    from env.task import task_default_dep_full, task_initializer, get_default_dep_full  # noqa
    from swarm.util import function_to_json

    di, dfu, dd = task_default_dep_full(domain, "full", "structured", dependency_verb_dep_orig=True)
    raw = json.load(open(f"{data_dir}/{domain}_tasks.json"))
    tools = set()
    tasks = []
    for goal in raw:
        for task in raw[goal]:
            task = dict(task, user_goal=goal)
            _, ui, ai, _ = task_initializer(domain, task, di, dfu, dd, None, "prompt", False, "structured")
            tn = [t["function"]["name"] for t in ai["tools"]]
            tools.update(tn)
            tasks.append(task)
    return tools, tasks


def census_domain(domain, ont_dir, data_dir):
    ont = json.load(open(f"{ont_dir}/ontology_{domain}.json"))
    preds = ont.get("predicates", {})
    tools, tasks = build_domain_tools(domain, data_dir)
    getters = sorted(t for t in tools if is_getter(t))

    # discriminating condition predicates = condition-kind constraint leaves across all tasks
    conds = set()
    for task in tasks:
        leaves = []
        collect_leaf_list(task.get("constraints"), leaves)
        for pred, pm, neg in leaves:
            if preds.get(pred, {}).get("kind") == "condition":
                conds.add(pred)

    rows = []  # (cond, klass, getter, shared)
    for c in sorted(conds):
        if c in tools:
            rows.append((c, "A_callable", c, []))
            continue
        g, sh = best_getter(c, getters)
        if g and sh:
            rows.append((c, "B_getter", g, sh))
        else:
            rows.append((c, "UNGROUNDABLE", "", []))
    return rows, getters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ont_dir", default="./induced")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--domains", default=",".join(DOMAINS))
    args = ap.parse_args()

    tot = defaultdict(int)
    print(f"{'domain':14s} {'#cond':>5s} {'A':>4s} {'B':>4s} {'UNGRND':>7s}   ungroundable_preds")
    print("-" * 100)
    all_ungr = {}
    bank_auto = {}
    for d in args.domains.split(","):
        d = d.strip()
        try:
            rows, getters = census_domain(d, args.ont_dir, args.data_dir)
        except Exception as e:
            print(f"{d:14s}  ERROR: {type(e).__name__}: {e}")
            continue
        nA = sum(1 for r in rows if r[1] == "A_callable")
        nB = sum(1 for r in rows if r[1] == "B_getter")
        nU = sum(1 for r in rows if r[1] == "UNGROUNDABLE")
        ungr = [r[0] for r in rows if r[1] == "UNGROUNDABLE"]
        all_ungr[d] = ungr
        tot["cond"] += len(rows); tot["A"] += nA; tot["B"] += nB; tot["U"] += nU
        print(f"{d:14s} {len(rows):5d} {nA:4d} {nB:4d} {nU:7d}   {', '.join(ungr) if ungr else '-'}")
        if d == "bank":
            bank_auto = {r[0]: r[2] for r in rows if r[1] == "B_getter"}
        # proposed table (verbose)
        for c, k, g, sh in rows:
            if k == "B_getter":
                print(f"      B {c:42s} -> {g:34s} (shared: {','.join(sh)})")
            elif k == "UNGROUNDABLE":
                print(f"      ! {c:42s} -> NONE (pure policy rule)")
    print("-" * 100)
    print(f"{'TOTAL':14s} {tot['cond']:5d} {tot['A']:4d} {tot['B']:4d} {tot['U']:7d}")
    gr = tot["A"] + tot["B"]
    if tot["cond"]:
        print(f"\ngroundable = {gr}/{tot['cond']} = {100*gr/tot['cond']:.1f}%  "
              f"| ungroundable (synth/other scope) = {tot['U']}/{tot['cond']} = {100*tot['U']/tot['cond']:.1f}%")

    # ---- validate auto-derive vs hand map (bank) ----
    hand = GETTER_BY_DOMAIN.get("bank", {})
    if hand and bank_auto:
        print("\n=== auto-derive (token-overlap) vs hand GETTER_BY_DOMAIN['bank'] ===")
        keys = sorted(set(hand) | set(bank_auto))
        agree = 0; both = 0
        for k in keys:
            h = hand.get(k, "-"); a = bank_auto.get(k, "-")
            mark = "OK " if (h == a and h != "-") else ("DIFF" if h != "-" and a != "-" else "    ")
            if h != "-" and a != "-":
                both += 1
                if h == a:
                    agree += 1
            print(f"  {mark} {k:42s} hand={h:30s} auto={a}")
        if both:
            print(f"\n  agreement on co-present keys = {agree}/{both} = {100*agree/both:.0f}%  "
                  f"(high => auto-derive trustworthy => Track A viable)")


if __name__ == "__main__":
    main()
