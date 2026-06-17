#!/usr/bin/env python3
"""C8-2nd τ² selection transfer eval (offline, key-free). For each ma_gold_extract exchange case,
the served model NAMES an op-IR (synth vocabulary) from the τ² NL + the variant options + the
ABox ordinal config; tau2_op_resolver executes it over the catalog -> new_item_id; score vs gold.

Arms (run one per invocation via --model/--gloss):
  S2  C8-trained model, gloss=0  -> transfer: does synth-learned routing fire on real τ² NL?
  S1  base model, gloss=1        -> ceiling (operator vocabulary spoon-fed in-context)
  S0  base model, gloss=0        -> floor

Reports overall new_item_id accuracy + breakdown by case-type (substitution vs superlative) and
emitted-op distribution. Honest: τ² substitution (majority of *items*) is filter-shaped so op-IR
~ static there; op-IR's edge should show on superlative/comparative items.
"""
import json, argparse, re
from collections import Counter
from depth_eval import chat, extract_json, build_spec
from tau2_op_resolver import resolve_op_tau2, is_ordinal, ordkey

SUP = re.compile(r'\b(cheap|expensive|highest|lowest|larg|small|biggest|bigger|smaller|max|min|most|'
                 r'least|better|brighter|faster|bigger|longer|shorter|more|less|prefer|further|farther)\w*', re.I)


def build_user(nl, ex, gloss, synth_format=False):
    catalog = ex["variant_catalog"]
    attrs = sorted({k for it in catalog for k in it["options"]})
    ords = {}
    for a in attrs:
        if is_ordinal(a, catalog):
            vals = sorted({it["options"][a] for it in catalog if a in it["options"]},
                          key=lambda v: ordkey(a, v) if ordkey(a, v) is not None else 0)
            ords[a] = vals
    if synth_format:
        # MATCH synth arm_B structure exactly (format-control diagnostic): "Query / attributes /
        # the ordinal attribute is X / OP_SPEC". Isolates whether op-slot collapse (op=exchange)
        # is NL-verb copying (survives format match) vs OOD-format breakage (heals under it).
        ordattr = next(iter(ords), attrs[0] if attrs else "")
        return (f"Query: {nl}\n\n"
                f"The catalog items have these attributes: {attrs}.\n"
                f"The ordinal/numeric attribute is '{ordattr}'.\n\n"
                f"{build_spec(gloss)}")
    ordline = f"Ordinal options (listed low to high): {json.dumps(ords)}.\n" if ords else ""
    if ex.get("old_options"):  # substitute/exchange: there IS a reference item to keep-rest from
        ref = (f"This changes the item: {ex['old_item_name']} "
               f"(current options: {json.dumps(ex['old_options'])}, id={ex['old_item_id']}).\n")
    else:  # create: no reference item, build fresh
        ref = f"This creates a new {ex.get('product_name','item')} (no existing item to keep options from).\n"
    return (f"Query: {nl}\n\n{ref}"
            f"The available options are: {attrs}.\n{ordline}\n"
            f"{build_spec(gloss)}")


def case_item_type(ex):
    """superlative if a sup keyword targets an ordinal attr; else substitution."""
    n_changed = sum(1 for k, v in ex["gold_new_options"].items() if ex["old_options"].get(k) != v)
    return "superlative" if SUP.search(ex.get("_nl", "")) else "substitution", n_changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="/home/woori/scratch/ma_eval_cases.jsonl")
    ap.add_argument("--base", default="http://localhost:8025/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--gloss", type=int, default=0)
    ap.add_argument("--synth_format", type=int, default=0, help="1=match synth arm_B prompt structure (format-control)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    cases = [json.loads(l) for l in open(args.cases, encoding="utf-8")]
    n = ok = 0
    by_type = {}            # type -> [ok, total]
    op_dist = Counter()
    rows = []
    recog = [0, 0]  # gold op-routing recognition (emitted op == gold case_op), when case_op is known
    for c in cases:
        case_op = c.get("case_op")
        for ex in c["exchanges"]:
            ex["_nl"] = c["nl"]
            user = build_user(c["nl"], ex, bool(args.gloss), synth_format=bool(args.synth_format))
            out = chat(args.base, args.model, [{"role": "system", "content": "Output ONLY JSON."},
                                               {"role": "user", "content": user}], 256)
            ir = extract_json(out)
            rid = resolve_op_tau2(ir, ex["variant_catalog"], anchor_id=ex["old_item_id"]) if ir else None
            hit = int(rid == ex["gold_new_item_id"])
            sup_t, nch = case_item_type(ex)
            t = case_op or sup_t  # prefer gold op-type breakdown (airline); fall back to regex (retail)
            by_type.setdefault(t, [0, 0]); by_type[t][1] += 1; by_type[t][0] += hit
            emitted_op = (ir or {}).get("op")
            op_dist[emitted_op] += 1
            if case_op:
                recog[1] += 1; recog[0] += int(emitted_op == case_op)
            n += 1; ok += hit
            gold_set = {k: v for k, v in ex["gold_new_options"].items()
                        if str(ex["old_options"].get(k)) != str(v)}
            rows.append({"task": c["task_id"], "type": t, "case_op": case_op, "n_changed": nch,
                         "op": emitted_op, "rid": rid, "gold": ex["gold_new_item_id"], "hit": hit, "ir": ir,
                         "old_options": ex["old_options"], "gold_options": ex["gold_new_options"],
                         "gold_set": gold_set, "emitted_set": (ir or {}).get("set")})
    f = lambda x: f"{x[0]}/{x[1]}({x[0]/max(x[1],1):.2f})"
    print(f"model={args.model} gloss={args.gloss} | items={n}")
    print(f"  overall new_item_id acc: {ok}/{n}({ok/max(n,1):.2f})")
    for t in by_type:
        print(f"  {t:>13}: {f(by_type[t])}")
    if recog[1]:
        print(f"  op-ROUTING recognition (emitted op == gold case_op): {f(recog)}")
    print(f"  emitted op dist: {dict(op_dist)}")
    res = {"model": args.model, "gloss": args.gloss, "overall": [ok, n], "recognition": recog,
           "by_type": by_type, "op_dist": {str(k): v for k, v in op_dist.items()}, "rows": rows}
    if args.out:
        json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
