#!/usr/bin/env python3
"""autoderive_getter_map.py — auto-derive v2 (DEFINITIVE, structural; replaces hand GETTER_BY_DOMAIN).

The Exp-4-precheck-FINAL finding: each condition predicate's evaluation function (in
env/domains/<d>/<d>.py) LITERALLY calls the getter(s) it needs, e.g.
    def above_minimum_age(self, username):
        ... self.domain_system.internal_get_interaction_time() ...
        ... self.domain_system.internal_get_user_birthday(username) ...
So the required getter-set for a condition = the set of `self.domain_system.<getter>(...)` calls in
its predicate body (recursively resolving calls to sibling predicates). This is the auto-derive:
structural, exhaustive, no token guessing, and transfer-clean (a held-out domain is parsed the same
way -> ABox swap, zero retrain). Arg-only predicates (valid_identification, income_proof_enough)
yield an EMPTY set = evaluable from the request, no gather needed.

Output: per-domain {condition -> sorted getter-set}, groundability (every getter is an agent tool?),
arg-only conditions, predicates-not-found, and validation vs hand GETTER_BY_DOMAIN['bank'].
Emits the derived map as JSON (--out) for build_tbox_planner_sft to consume.

  RUN (clone root, PYTHONPATH=clone):
    python scripts/distill/sopbench/autoderive_getter_map.py --ont_dir ./induced --data_dir ./data \
       --src_root env/domains --out induced/getter_map.json
"""
import argparse, ast, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_tbox_planner_sft import DOMAINS, GETTER_BY_DOMAIN
from precheck_getter_groundability import is_getter, build_domain_tools


def is_obs_tool(name):
    """observation/getter tool naming in SOPBench."""
    return is_getter(name) or name.startswith("view_")


def domain_system_calls(funcnode):
    """names X for every `self.domain_system.X(...)` call, and sibling-predicate calls self.Y(...)."""
    getters, sibs = set(), set()
    for n in ast.walk(funcnode):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            f = n.func
            # self.domain_system.X(...)
            if (isinstance(f.value, ast.Attribute) and f.value.attr == "domain_system"
                    and isinstance(f.value.value, ast.Name) and f.value.value.id == "self"):
                getters.add(f.attr)
            # self.Y(...)  -> sibling predicate (resolve recursively)
            elif isinstance(f.value, ast.Name) and f.value.id == "self":
                sibs.add(f.attr)
    return getters, sibs


def resolve(name, funcs, memo, seen=None):
    """getter-set for predicate `name`, recursively resolving sibling-predicate calls."""
    if name in memo:
        return memo[name]
    seen = seen or set()
    if name not in funcs or name in seen:
        return set()
    seen.add(name)
    direct, sibs = domain_system_calls(funcs[name])
    out = {g for g in direct if is_obs_tool(g)}
    for s in sibs:
        out |= resolve(s, funcs, memo, seen)
    memo[name] = out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ont_dir", default="./induced")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--src_root", default="env/domains")
    ap.add_argument("--domains", default=",".join(DOMAINS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    full_map = {}
    tot = defaultdict(int)
    for d in args.domains.split(","):
        d = d.strip()
        src = os.path.join(args.src_root, d, f"{d}.py")
        try:
            tree = ast.parse(open(src).read())
            preds = json.load(open(f"{args.ont_dir}/ontology_{d}.json")).get("predicates", {})
            tools, _ = build_domain_tools(d, args.data_dir)
        except Exception as e:
            print(f"{d}: ERROR {type(e).__name__}: {e}"); continue
        funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        memo = {}
        conds = sorted(p for p, v in preds.items() if v.get("kind") == "condition")
        dmap, argonly, notfound, gap = {}, [], [], []
        for c in conds:
            if c not in funcs:
                notfound.append(c); continue
            gset = sorted(resolve(c, funcs, memo))
            dmap[c] = gset
            if not gset:
                argonly.append(c)
            missing = [g for g in gset if g not in tools]
            if missing:
                gap.append((c, missing))
        full_map[d] = dmap
        ng = sum(1 for c in dmap if dmap[c])
        tot["cond"] += len(conds); tot["mapped"] += ng; tot["argonly"] += len(argonly)
        tot["notfound"] += len(notfound); tot["gap"] += len(gap)
        print(f"\n===== {d}: {len(conds)} cond | getter-mapped {ng} | arg-only {len(argonly)} | "
              f"not-found {len(notfound)} | TOOL-GAP {len(gap)} =====")
        for c in conds:
            if c in dmap:
                tag = " [ARG-ONLY]" if not dmap[c] else ""
                print(f"    {c:46s} -> {', '.join(dmap[c]) if dmap[c] else '(none)'}{tag}")
        if notfound:
            print(f"    NOT-FOUND as predicate method: {', '.join(notfound)}")
        if gap:
            for c, m in gap:
                print(f"    !! TOOL-GAP {c}: getter(s) not in agent tools: {m}")

    print("\n" + "=" * 70)
    print(f"TOTAL: {tot['cond']} cond | getter-mapped {tot['mapped']} | arg-only {tot['argonly']} | "
          f"not-found {tot['notfound']} | tool-gap {tot['gap']}")
    grounded = tot["mapped"] + tot["argonly"]
    if tot["cond"]:
        print(f"grounded (getter-set OR arg-only) = {grounded}/{tot['cond']} = {100*grounded/tot['cond']:.1f}%")

    # validate vs hand bank-map
    hand = GETTER_BY_DOMAIN.get("bank", {})
    bm = full_map.get("bank", {})
    if hand and bm:
        print("\n=== validate vs hand GETTER_BY_DOMAIN['bank'] (hand getter in auto-derived set?) ===")
        ok = 0
        for k, h in sorted(hand.items()):
            auto = bm.get(k)
            hit = auto is not None and h in auto
            ok += hit
            print(f"  {'OK ' if hit else 'MISS'} {k:42s} hand={h:30s} auto={auto}")
        print(f"  hand-map recall = {ok}/{len(hand)}")

    if args.out:
        json.dump(full_map, open(args.out, "w"), indent=2)
        print(f"\nwrote derived map -> {args.out}")


if __name__ == "__main__":
    main()
