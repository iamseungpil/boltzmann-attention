#!/usr/bin/env python3
"""Exhaustive per-case trajectory trace of the M-A 3-arm eval to FIX the write-wall
root cause (fabrication vs reasoning). Joins ma_eval_results.jsonl with ma_eval_cases.jsonl.

Outputs:
 (1) paired A-vs-B item-level contingency {both ok, A-only, B-only, both wrong}
     -> McNemar view of whether the formal+resolver architecture helps.
 (2) every B failure traced: emitted select_by/fallback vs gold options, classified
     reasoning(wrong_criteria) vs unresolvable(resolver_fail) vs tie.
 (3) the A-only and B-only items (where the architecture flips the outcome) -> isolates
     fabrication (B fixes A's id-copy error) from emit-overhead.
"""
import json, argparse
from collections import Counter


def load(results, cases):
    R = {}
    for l in open(results, encoding="utf-8"):
        d = json.loads(l)
        R[(d["task_id"], d["arm"])] = d
    C = {c["task_id"]: c for c in (json.loads(l) for l in open(cases, encoding="utf-8"))}
    return R, C


def main(results, cases):
    R, C = load(results, cases)
    tids = [t for (t, a) in R if a == "A"]
    # (1) paired item-level A vs B
    pair = Counter()
    item_rows = []
    for t in tids:
        a, b = R.get((t, "A")), R.get((t, "B"))
        if not a or not b:
            continue
        c = C[t]
        for i in range(c["n_items"]):
            ac = a["item_correct"][i]; bc = b["item_correct"][i]
            key = ("A" + ("+" if ac else "-")) + ("B" + ("+" if bc else "-"))
            pair[key] += 1
            item_rows.append((t, i, c["exchanges"][i], a, b, ac, bc))
    print("=== (1) paired A-vs-B item contingency ===")
    bothok = pair.get("A+B+", 0); aonly = pair.get("A+B-", 0)
    bonly = pair.get("A-B+", 0); bothwrong = pair.get("A-B-", 0)
    tot = bothok + aonly + bonly + bothwrong
    print(f"  both correct : {bothok}")
    print(f"  A-only (B broke it / emit-overhead): {aonly}")
    print(f"  B-only (architecture FIXED it = fabrication removed): {bonly}")
    print(f"  both wrong  (reasoning-bound, architecture-invariant): {bothwrong}")
    print(f"  total items : {tot}")
    print(f"  => B-only={bonly} (fabrication share) vs both-wrong={bothwrong} (reasoning share)")

    # (3) B-only items (architecture fixed) and A-only (architecture hurt)
    print("\n=== (3) items where ARCHITECTURE FLIPS outcome ===")
    for (t, i, ex, a, b, ac, bc) in item_rows:
        if ac != bc:
            tag = "B-FIXED(fab)" if bc else "A-WON(overhead)"
            apred = a["pred"][i] if a.get("pred") else None
            bpred = b["pred"][i] if b.get("pred") else None
            sel = b.get("selectors", [{}]*(i+1))[i]
            print(f"  task {t} item{i} [{tag}] gold={ex['gold_new_item_id']}({ex['gold_new_options']})")
            print(f"     A_pred={apred}  B_pred={bpred}  B_select={json.dumps(sel,ensure_ascii=False)}")

    # (2) every B failure traced
    print("\n=== (2) all B failures traced (reasoning vs unresolvable) ===")
    rk = Counter()
    for t in tids:
        b = R.get((t, "B")); c = C[t]
        if not b or b.get("parse_fail"):
            if b and b.get("parse_fail"):
                rk["parse"] += 1
            continue
        for i in range(c["n_items"]):
            if b["item_correct"][i]:
                continue
            ex = c["exchanges"][i]; sel = b.get("selectors", [{}]*(i+1))[i]
            kind = b["fail_kind"][i]
            rk[kind.split(":")[0]] += 1
            print(f"  task {t} item{i} [{kind}]")
            print(f"     NL gold options : {ex['gold_new_options']}  (old: {ex['old_options']})")
            print(f"     B select_by     : {json.dumps(sel.get('select_by',{}),ensure_ascii=False)}  fallback={json.dumps(sel.get('fallback',[]),ensure_ascii=False)}")
            print(f"     B resolved -> {b['pred'][i]}")
    print(f"\n  B failure-kind totals: {dict(rk)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/home/woori/scratch/ma_eval_results.jsonl")
    ap.add_argument("--cases", default="/home/woori/scratch/ma_eval_cases.jsonl")
    args = ap.parse_args()
    main(args.results, args.cases)
