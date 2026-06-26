#!/usr/bin/env python3
"""Argument (`set`) failure autopsy for the multi-domain content-routing eval (tau2_op_eval rows).

Base eval showed op-ROUTING recognition ~0.78-0.94 but new_item_id accuracy ~0.19-0.44: the model
NAMES the right operation (substitute/create) yet resolves the WRONG item. This pins the cause to the
`set` argument (which options to change, to what value). For each item we deterministically compare the
emitted `set` to the gold `set` (= gold_new_options minus old_options) and classify:

  op_mismatch    : emitted op != gold case_op (routing wrong, not an arg problem)
  set_correct    : emitted set == gold set, yet still missed (=> resolver/availability/anchor issue)
  wrong_value    : emitted set keys == gold keys but >=1 value differs (right attr, wrong value)
  missing_key    : emitted set omits >=1 attr the gold changes (kept an attr that should change)
  extra_key      : emitted set changes >=1 attr the gold keeps  (broke keep-rest)
  mixed_keys     : both missing and extra keys
  no_set         : emitted op uses set but none emitted (null/empty)

Splits by domain x case_op. Prints the dominant failure category among recognition-correct misses =
the confirmed root cause. GPU-free; run on a result JSON that has enriched rows (emitted_set/gold_set).
"""
import json, argparse
from collections import Counter, defaultdict


def classify(row):
    op, cop = row.get("op"), row.get("case_op")
    if cop and op != cop:
        return "op_mismatch"
    es = row.get("emitted_set"); gs = row.get("gold_set") or {}
    if not isinstance(es, dict) or not es:
        return "no_set"
    ek, gk = set(es), set(gs)
    missing = gk - ek            # gold changes it, model didn't
    extra = ek - gk              # model changes it, gold keeps it
    # for `extra`, only count as keep-rest break if the emitted value actually differs from old
    old = row.get("old_options") or {}
    extra_real = {k for k in extra if str(es.get(k)) != str(old.get(k))}
    if missing and extra_real:
        return "mixed_keys"
    if missing:
        return "missing_key"
    if extra_real:
        return "extra_key"
    # same effective key set as gold -> check values
    if all(str(es.get(k)) == str(gs.get(k)) for k in gk):
        return "set_correct"     # right set, still missed => non-arg cause
    return "wrong_value"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True, nargs="+", help="tau2_op_eval result JSON(s)")
    ap.add_argument("--show", type=int, default=6, help="example misses per category")
    args = ap.parse_args()
    for path in args.res:
        d = json.load(open(path, encoding="utf-8"))
        rows = d.get("rows", [])
        tag = path.split("/")[-1]
        ok = sum(r["hit"] for r in rows)
        print(f"\n===== {tag} | model={d.get('model')} gloss={d.get('gloss')} | "
              f"acc={ok}/{len(rows)}({ok/max(len(rows),1):.2f}) =====")
        # all rows
        cat_all = Counter(classify(r) for r in rows)
        # misses only (the failures we want to explain)
        misses = [r for r in rows if not r["hit"]]
        cat_miss = Counter(classify(r) for r in misses)
        print(f"  misses: {len(misses)} | category breakdown (misses):")
        for c, n in cat_miss.most_common():
            print(f"    {c:>12}: {n}")
        # recognition-correct misses = the pure argument failures
        rc_miss = [r for r in misses if not (r.get("case_op") and r["op"] != r["case_op"])]
        cat_rc = Counter(classify(r) for r in rc_miss)
        print(f"  recognition-correct misses (op named right, item wrong): {len(rc_miss)}")
        for c, n in cat_rc.most_common():
            print(f"    {c:>12}: {n}")
        # by case_op
        bycop = defaultdict(Counter)
        for r in misses:
            bycop[r.get("case_op")][classify(r)] += 1
        for cop, cc in bycop.items():
            print(f"  [case_op={cop}] miss categories: {dict(cc)}")
        # examples
        shown = defaultdict(int)
        print("  --- example misses ---")
        for r in misses:
            c = classify(r)
            if shown[c] >= args.show:
                continue
            shown[c] += 1
            print(f"    [{c}] task={r['task']} op={r['op']}/{r.get('case_op')} "
                  f"old={r.get('old_options')} gold_set={r.get('gold_set')} emit_set={r.get('emitted_set')} "
                  f"rid={r['rid']} gold={r['gold']}")


if __name__ == "__main__":
    main()
