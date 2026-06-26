#!/usr/bin/env python3
"""A/B test fixes for comparative arm-B mislabeling (diag: operands correct, op labeled 'filter').
Tests, comparative cases only, the op-RECOGNITION + engine accuracy under 3 prompt variants:
  B0  baseline OP_SPEC (current depth_eval) — expected ~0.00
  B1  OP_SPEC + per-op GLOSS (defines what 'comparative' means: closest item above/below an anchor)
  B2  B1 + front-loaded NL (lead with the relational clause so 'Among items where' stops priming filter)
Isolates whether the bottleneck is (a) missing operator vocabulary (B1 fixes) or
(b) NL surface priming (needs B2). Engine + gold unchanged; only the LLM-facing prompt varies.
"""
import json, argparse
from collections import Counter
from synth_depth import resolve_operation
from depth_eval import chat, extract_json

BASE_SPEC = ('Output ONLY a JSON OPERATION (do NOT compute the answer): '
             '{"op": "argmax"|"argmin"|"rank"|"filter"|"comparative", "attr": "<ordinal attr name>", '
             '"among": {"<attr>": "<value>", ...}, "k": <int for rank>, "dir": "greater"|"less", '
             '"anchor_id": "<item id for comparative>"}. Name the operation the query asks for.')

GLOSS = ('\nOperation meanings — pick the one the query asks for:\n'
         '- filter: a single item is pinned by attribute equalities alone (no ranking, no reference item).\n'
         '- argmax: the item with the HIGHEST ordinal value (within `among`).\n'
         '- argmin: the item with the LOWEST ordinal value (within `among`).\n'
         '- rank: the k-th highest ordinal value (k from "second/third highest").\n'
         '- comparative: the item whose ordinal value is the CLOSEST one strictly ABOVE (dir=greater) '
         'or BELOW (dir=less) a REFERENCE item named by anchor_id. Use this whenever the query compares '
         'to "that of item <id>" / "just greater/less than item <id>".')

SPEC_B0 = BASE_SPEC
SPEC_B1 = BASE_SPEC + GLOSS


def nl_frontloaded(ex):
    f = ex["filter"]; o = ex["ord_attr"]; aid = ex["op_ir"]["anchor_id"]
    fstr = " and ".join(f"{k} = {v}" for k, v in f.items())
    return (f"Find the item whose {o} is just GREATER than item {aid}'s {o} (the closest value above it), "
            f"considering only items where {fstr}.")


def run(base, model, ex, spec, nl):
    attrs = ex["cat_attrs"] + [ex["ord_attr"]]
    user = (f"Query: {nl}\n\nThe catalog items have these attributes: {attrs}.\n"
            f"The ordinal/numeric attribute is '{ex['ord_attr']}'.\n\n{spec}")
    out = chat(base, model, [{"role": "system", "content": "Output ONLY JSON."},
                             {"role": "user", "content": user}], 200)
    ir = extract_json(out)
    rid = resolve_operation(ir, ex["items"]) if ir else None
    g = ex["op_ir"]
    recog = bool(ir) and ir.get("op") == g.get("op")
    return rid == ex["gold_id"], recog, (ir or {}).get("op")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="http://localhost:8025/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    exs = [json.loads(l) for l in open(args.data, encoding="utf-8") if '"op": "comparative"' in l][:args.limit]
    n = len(exs)
    print(f"comparative cases: {n} | model={args.model}")
    variants = [("B0", SPEC_B0, lambda e: e["nl"]),
                ("B1", SPEC_B1, lambda e: e["nl"]),
                ("B2", SPEC_B1, nl_frontloaded)]
    res = {}
    for name, spec, nlf in variants:
        acc = recog = 0; ops = Counter()
        for ex in exs:
            ok, rc, op = run(args.base, args.model, ex, spec, nlf(ex))
            acc += int(ok); recog += int(rc); ops[op] += 1
        res[name] = {"acc": [acc, n], "recog": [recog, n], "labeled_ops": dict(ops)}
        print(f"  {name}: engine-acc={acc}/{n}({acc/n:.2f})  op-recog={recog}/{n}({recog/n:.2f})  labels={dict(ops)}")
    if args.out:
        json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
