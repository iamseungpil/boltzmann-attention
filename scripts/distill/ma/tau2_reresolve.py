#!/usr/bin/env python3
"""Offline counterfactual: re-resolve a saved tau2_op_eval result with the GROUNDED-anchor resolver,
to quantify how much of the argument-failure is the anchor_id hallucination (our design flaw) vs the
genuine `set` extraction error. The model output (emitted op_ir) is UNCHANGED; only the deterministic
engine changes (grounds the anchor to the context old item instead of trusting the model's anchor_id).

Replays the eval's (case -> exchange) iteration order to re-pair each saved row with its case (so the
catalog + grounded old_item_id are available). Reports old acc vs new acc + how many flipped. GPU-free.
"""
import json, argparse
from tau2_op_resolver import resolve_op_tau2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True, help="tau2_op_eval result json (has rows with emitted ir)")
    ap.add_argument("--cases", required=True, help="the cases jsonl that produced it")
    args = ap.parse_args()
    res = json.load(open(args.res, encoding="utf-8"))
    rows = res["rows"]
    cases = [json.loads(l) for l in open(args.cases, encoding="utf-8")]
    # replay iteration order: case -> exchange (exactly how tau2_op_eval built rows)
    pairs = [(c, ex) for c in cases for ex in c["exchanges"]]
    assert len(pairs) == len(rows), f"row/case mismatch {len(rows)} vs {len(pairs)}"
    old_ok = new_ok = flipped = 0
    still_miss = {}
    for (c, ex), r in zip(pairs, rows):
        ir = r.get("ir")
        rid2 = resolve_op_tau2(ir, ex["variant_catalog"], anchor_id=ex["old_item_id"]) if ir else None
        h_old = r["hit"]; h_new = int(rid2 == ex["gold_new_item_id"])
        old_ok += h_old; new_ok += h_new
        if h_new and not h_old:
            flipped += 1
        if not h_new:
            # classify residual (genuine, anchor-independent)
            es = (ir or {}).get("set") or {}
            gs = r.get("gold_set") or {}
            ek, gk = set(es), set(gs)
            if r.get("case_op") and (ir or {}).get("op") != r["case_op"]:
                cat = "op_mismatch"
            elif not es:
                cat = "no_set"
            elif gk - ek:
                cat = "missing_key"
            elif {k for k in ek - gk if str(es.get(k)) != str(ex["old_options"].get(k))}:
                cat = "extra_key"
            elif all(str(es.get(k)) == str(gs.get(k)) for k in gk):
                cat = "set_correct_other"
            else:
                cat = "wrong_value"
            still_miss[cat] = still_miss.get(cat, 0) + 1
    n = len(rows)
    print(f"{args.res.split('/')[-1]}: acc {old_ok}/{n}({old_ok/n:.2f}) -> GROUNDED {new_ok}/{n}({new_ok/n:.2f}) "
          f"| anchor-fix recovered {flipped}")
    print(f"  residual miss categories (genuine, anchor-independent): {dict(sorted(still_miss.items(), key=lambda x:-x[1]))}")


if __name__ == "__main__":
    main()
