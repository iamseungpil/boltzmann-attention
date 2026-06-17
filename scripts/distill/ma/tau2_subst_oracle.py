#!/usr/bin/env python3
"""Oracle check: does the `substitute` op-IR correctly model τ² retail exchange?

The τ² 2nd-gate negative root cause (HANDOFF_2026_06_17_PM §1) = the 5-op vocabulary
(filter/argmax/argmin/rank/comparative) could NOT represent τ² substitution; the C8-trained
model force-mapped exchange->comparative and invented a `to:` field. This oracle tests the FIX:
for each extracted exchange case, build the GOLD substitute op-IR
  {op: substitute, anchor_id: old_item_id, set: {changed options}}
and check tau2_op_resolver.resolve_op_tau2 reproduces the gold new_item_id.

100% => substitute is the right generator for τ² exchange (a representation-adequacy result,
independent of any learning). <100% => log which cases aren't pure keep-rest substitution
(e.g. the gold changes every option = create-shaped, or catalog/availability quirks) — honest.
"""
import json, argparse
from collections import Counter
from tau2_op_resolver import resolve_op_tau2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="/home/woori/scratch/ma_eval_cases.jsonl")
    args = ap.parse_args()
    cases = [json.loads(l) for l in open(args.cases, encoding="utf-8")]
    n = ok = 0
    nch_dist = Counter()
    fails = []
    for c in cases:
        for ex in c["exchanges"]:
            old = ex["old_options"]; new = ex["gold_new_options"]
            setv = {k: v for k, v in new.items() if str(old.get(k)) != str(v)}
            n_changed = len(setv)
            n_total = len(new)
            nch_dist[(n_changed, n_total)] += 1
            op_ir = {"op": "substitute", "anchor_id": ex["old_item_id"], "set": setv}
            rid = resolve_op_tau2(op_ir, ex["variant_catalog"], anchor_id=ex["old_item_id"])
            hit = int(rid == ex["gold_new_item_id"])
            n += 1; ok += hit
            if not hit:
                fails.append({"task": c["task_id"], "set": setv, "rid": rid,
                              "gold": ex["gold_new_item_id"], "n_changed": n_changed, "n_total": n_total})
    print(f"substitute oracle: {ok}/{n} ({ok/max(n,1):.3f}) reproduce gold new_item_id")
    print(f"  changed/total-options distribution: {dict(sorted(nch_dist.items()))}")
    print(f"  (n_changed==n_total => create-shaped; n_changed<n_total => true keep-rest substitute)")
    for f in fails[:20]:
        print(f"  FAIL task={f['task']} set={f['set']} -> rid={f['rid']} gold={f['gold']} "
              f"({f['n_changed']}/{f['n_total']} changed)")


if __name__ == "__main__":
    main()
