#!/usr/bin/env python3
"""Diagnose WHY comparative arm-B = 0.00 (B_BUDGET_SCALE §0-bis follow-up).
Runs arm-B (op-IR emit) on comparative cases ONLY, captures the raw emitted IR, and
categorises the failure against the gold IR field-by-field. The engine needs ALL of
{op, attr, dir, anchor_id, among} correct — so we localise the bottleneck:
  - bad_json        : no parseable JSON
  - op_misnamed     : op != comparative (named argmax/argmin/rank/filter instead) = RECOGNITION fail
  - attr_wrong      : attr mismatch
  - anchor_missing  : no anchor_id field
  - anchor_wrong    : anchor_id present but != gold (copy failure)
  - dir_wrong       : dir != gold
  - among_wrong     : among (filter) mismatch
  - engine_none     : all fields look right but engine returned None (logic edge)
  - OK              : engine reproduced gold
One example can have multiple flags; we report the FIRST blocking cause + full counts.
"""
import json, argparse, re
from synth_depth import resolve_operation
from depth_eval import chat, extract_json, OP_SPEC, arm_B  # reuse exact arm-B prompt


def diag_one(base, model, ex):
    attrs = ex["cat_attrs"] + [ex["ord_attr"]]
    user = (f"Query: {ex['nl']}\n\nThe catalog items have these attributes: {attrs}.\n"
            f"The ordinal/numeric attribute is '{ex['ord_attr']}'.\n\n{OP_SPEC}")
    raw = chat(base, model, [{"role": "system", "content": "Output ONLY JSON."},
                             {"role": "user", "content": user}], 200)
    ir = extract_json(raw)
    g = ex["op_ir"]
    rid = resolve_operation(ir, ex["items"]) if ir else None
    flags = []
    if not ir:
        flags.append("bad_json")
        return {"flags": flags, "first": "bad_json", "ok": False, "raw": raw[:200], "ir": None}
    if ir.get("op") != g.get("op"):
        flags.append(f"op_misnamed:{ir.get('op')}")
    if str(ir.get("attr")) != str(g.get("attr")):
        flags.append("attr_wrong")
    if "anchor_id" not in ir or ir.get("anchor_id") in (None, ""):
        flags.append("anchor_missing")
    elif str(ir.get("anchor_id")) != str(g.get("anchor_id")):
        flags.append("anchor_wrong")
    if str(ir.get("dir", "greater")) != str(g.get("dir")):
        flags.append("dir_wrong")
    if (ir.get("among") or {}) != (g.get("among") or {}):
        flags.append("among_wrong")
    ok = (rid == ex["gold_id"])
    if not flags and not ok:
        flags.append("engine_none")
    first = "OK" if ok else (flags[0] if flags else "engine_none")
    return {"flags": flags, "first": first, "ok": ok, "ir": ir, "gold_ir": g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="http://localhost:8025/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    exs = [json.loads(l) for l in open(args.data, encoding="utf-8") if '"op": "comparative"' in l][:args.limit]
    print(f"comparative cases: {len(exs)} | model={args.model}")
    from collections import Counter
    first_c = Counter(); flag_c = Counter(); rows = []
    for ex in exs:
        d = diag_one(args.base, args.model, ex)
        first_c[d["first"]] += 1
        for fl in d["flags"]:
            flag_c[fl.split(":")[0]] += 1
        rows.append(d)
    n = len(exs)
    print("=== FIRST blocking cause (mutually exclusive) ===")
    for k, v in first_c.most_common():
        print(f"  {k:>16}: {v}/{n} ({v/n:.2f})")
    print("=== ALL flags (overlapping) ===")
    for k, v in flag_c.most_common():
        print(f"  {k:>16}: {v}/{n} ({v/n:.2f})")
    print("=== 5 sample emitted IRs ===")
    for d in rows[:5]:
        print(f"  first={d['first']:>14} ir={json.dumps(d.get('ir'), ensure_ascii=False)}")
        print(f"                 gold={json.dumps(d.get('gold_ir'), ensure_ascii=False)}")
    if args.out:
        json.dump({"first": dict(first_c), "flags": dict(flag_c),
                   "rows": [{k: r.get(k) for k in ("first", "flags", "ir", "gold_ir")} for r in rows]},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
