#!/usr/bin/env python3
"""triage_ungroundable.py — classify the 44 name-token UNGROUNDABLE conditions against the
ACTUAL domain getter set + tool DESCRIPTIONS, to separate:

  HEURISTIC_MISS    : a getter clearly resolves it (name tokens just didn't overlap) -> groundable,
                      auto-derive v2 (description/co-occurrence) will recover it.
  TRUE_UNGROUNDABLE : no getter returns data from which the condition is computable -> a pure
                      policy rule = the genuine synthetic/other scope (the real ceiling).

For each ungroundable condition prints its args, a CATEGORY guess (temporal/quantity/state/value),
and the top candidate getters by DESCRIPTION-token overlap (with a description snippet) so the
classification can be VERIFIED by hand before building auto-derive v2.

  RUN (clone root, PYTHONPATH=clone):
    python scripts/distill/sopbench/triage_ungroundable.py --ont_dir ./induced --data_dir ./data
"""
import argparse, json, os, re, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_tbox_planner_sft import DOMAINS, collect_leaf_list
from precheck_getter_groundability import toks, is_getter, best_getter, STOP

# CATEGORY keyword buckets (on the condition name).
CAT = [
    ("temporal", ("period", "deadline", "date", "time", "renewal", "enrollment", "appeal",
                  "expiry", "expired", "lead", "before", "after", "schedule")),
    ("quantity", ("limit", "max", "min", "maximum", "minimum", "remaining", "exceeded",
                  "less", "under", "over", "count", "num", "quota", "stays", "slots", "credits")),
    ("state",    ("has", "is", "already", "pending", "conflict", "returned", "active",
                  "inactive", "completed", "cart", "stock", "exists", "in_order", "zero",
                  "probation", "confirmed", "checked")),
    ("value",    ("income", "gpa", "credit", "age", "balance", "amount", "score", "proof",
                  "requirement", "eligible", "residency", "identification", "tuition", "fee")),
]


def category(cond):
    cl = cond.lower()
    for name, kws in CAT:
        if any(k in cl for k in kws):
            return name
    return "other"


def desc_match(cond, getters_desc):
    """rank getters by (description+name)-token overlap with the condition; return top 3."""
    cset = set(toks(cond))
    scored = []
    for g, d in getters_desc.items():
        dtok = set(toks(g)) | set(t for t in re.split(r"[^a-z0-9]+", (d or "").lower())
                                  if t and t not in STOP)
        sh = cset & dtok
        if sh:
            scored.append((len(sh), g, sorted(sh), (d or "")[:70]))
    scored.sort(reverse=True)
    return scored[:3]


def domain_tools_desc(domain, data_dir):
    from env.task import task_default_dep_full, task_initializer
    from swarm.util import function_to_json  # noqa
    di, dfu, dd = task_default_dep_full(domain, "full", "structured", dependency_verb_dep_orig=True)
    raw = json.load(open(f"{data_dir}/{domain}_tasks.json"))
    desc = {}
    tasks = []
    for goal in raw:
        for task in raw[goal]:
            task = dict(task, user_goal=goal)
            _, ui, ai, _ = task_initializer(domain, task, di, dfu, dd, None, "prompt", False, "structured")
            for t in ai["tools"]:
                f = t["function"]
                desc.setdefault(f["name"], f.get("description", ""))
            tasks.append(task)
    return desc, tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ont_dir", default="./induced")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--domains", default=",".join(DOMAINS))
    args = ap.parse_args()

    tally = defaultdict(int)
    cat_tally = defaultdict(int)
    for d in args.domains.split(","):
        d = d.strip()
        try:
            desc, tasks = domain_tools_desc(d, args.data_dir)
            preds = json.load(open(f"{args.ont_dir}/ontology_{d}.json")).get("predicates", {})
        except Exception as e:
            print(f"{d}: ERROR {type(e).__name__}: {e}"); continue
        tools = set(desc)
        getters = sorted(t for t in tools if is_getter(t))
        getters_desc = {g: desc[g] for g in getters}

        # ungroundable = condition leaf with no name-token getter and not directly callable
        cond_args = {}
        for task in tasks:
            leaves = []
            collect_leaf_list(task.get("constraints"), leaves)
            for pred, pm, neg in leaves:
                if preds.get(pred, {}).get("kind") == "condition":
                    cond_args.setdefault(pred, pm)
        ungr = []
        for c in sorted(cond_args):
            if c in tools:
                continue
            g, sh = best_getter(c, getters)
            if not (g and sh):
                ungr.append(c)
        if not ungr:
            continue
        print(f"\n========== {d}  (ungroundable by name-token: {len(ungr)}) ==========")
        print(f"  getters available: {', '.join(getters)}")
        for c in ungr:
            cat = category(c)
            cat_tally[cat] += 1
            cands = desc_match(c, getters_desc)
            args_s = ",".join(f"{k}={v}" for k, v in (cond_args[c] or {}).items()) or "-"
            label = "HEURISTIC_MISS" if cands else "TRUE_UNGROUNDABLE"
            tally[label] += 1
            print(f"\n  [{cat:8s}] {c}   (args: {args_s})  -> PROPOSED: {label}")
            for n, g, sh, dsnip in cands:
                print(f"        ~ {g:40s} shared={','.join(sh):28s} | {dsnip}")
            if not cands:
                print(f"        (no getter description shares any token -> genuine policy rule)")
    print("\n" + "=" * 70)
    print(f"PROPOSED labels: " + " | ".join(f"{k}={v}" for k, v in tally.items()))
    print(f"by category:     " + " | ".join(f"{k}={v}" for k, v in sorted(cat_tally.items())))
    print("VERIFY the HEURISTIC_MISS candidates above; TRUE_UNGROUNDABLE = synthetic scope.")


if __name__ == "__main__":
    main()
