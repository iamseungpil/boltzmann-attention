#!/usr/bin/env python3
"""THESIS eval — does a model overcome superlative/comparative WITH deterministic structure?
(B_BUDGET_SCALE_DESIGN §0-bis). 4 arms on synth_depth examples:

  A  in-head ALONE : LLM sees the catalog + NL query, emits the answer id directly (scan in-head).
  Acot in-head+CoT : same, free CoT then id (external serial extension).
  B  op-IR + ENGINE: LLM sees ONLY the NL query + attr names, NAMES the operation (op-IR JSON);
                     a deterministic engine (resolve_operation) executes it over the catalog.
                     = the thesis structure. LLM does shallow recognition; engine does the deep scan.
  D  oracle        : engine on the GOLD op-IR (sanity, must be 1.0).

THESIS prediction: B >> A (small model overcomes via structure) and B ~ D; B >= big-model-A.
Also reports op-RECOGNITION accuracy for B (did the LLM name the right operation?) — shallow, should
be high even for small models if the theory (recognition is low-depth) holds.
"""
import json, argparse, re
from synth_depth import resolve_operation


def chat(base, model, messages, max_tokens=400):
    import requests
    r = requests.post(f"{base}/chat/completions",
                      json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0},
                      timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"].get("content") or ""


def extract_json(text):
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    s = m.group(0)
    for end in range(len(s), 0, -1):
        try:
            return json.loads(s[:end])
        except Exception:
            continue
    return None


def catalog_str(items):
    # compact: id + attrs (drop nothing; the model needs to scan for arm A)
    return json.dumps(items, ensure_ascii=False)


def arm_A(base, model, ex, cot=False):
    items = ex["items"]
    sys = "You are a precise selection assistant. Output ONLY JSON."
    user = (f"Catalog (list of items):\n{catalog_str(items)}\n\nQuery: {ex['nl']}\n\n")
    if cot:
        msgs = [{"role": "system", "content": sys},
                {"role": "user", "content": user + "Think step by step about which item the query selects. Reason only; do not answer yet."}]
        r1 = chat(base, model, msgs)
        msgs += [{"role": "assistant", "content": r1},
                 {"role": "user", "content": 'Now output ONLY {"id": "<item_id>"}.'}]
        out = chat(base, model, msgs, 80)
    else:
        out = chat(base, model, [{"role": "system", "content": sys},
                                 {"role": "user", "content": user + 'Output ONLY {"id": "<item_id>"}.'}], 80)
    j = extract_json(out)
    return (j or {}).get("id")


# v2 (2026-06-17): per-op GLOSS added. Diagnosis showed the 7B extracts every operand
# (attr/among/dir/anchor_id) correctly for comparative but mislabels op="filter" (the leading
# "Among items where {filter}" primes it). The gloss DEFINES the operator vocabulary (a formal
# IR always specifies its ops) and recovers comparative 0.00->1.00 (N-invariant) with no regression
# on the other ops. Original spec (no gloss) kept in git history; v1 numbers = depth_7B_N*.json.
OP_SPEC = ('Output ONLY a JSON OPERATION (do NOT compute the answer): '
           '{"op": "argmax"|"argmin"|"rank"|"filter"|"comparative", "attr": "<ordinal attr name>", '
           '"among": {"<attr>": "<value>", ...}, "k": <int for rank>, "dir": "greater"|"less", '
           '"anchor_id": "<item id for comparative>"}. Name the operation the query asks for.\n'
           'Operation meanings — pick the one the query asks for:\n'
           '- filter: a single item is pinned by attribute equalities alone (no ranking, no reference item).\n'
           '- argmax: the item with the HIGHEST ordinal value (within `among`).\n'
           '- argmin: the item with the LOWEST ordinal value (within `among`).\n'
           '- rank: the k-th highest ordinal value (k from "second/third highest").\n'
           '- comparative: the item whose ordinal value is the CLOSEST one strictly ABOVE (dir=greater) '
           'or BELOW (dir=less) a REFERENCE item named by anchor_id. Use this whenever the query compares '
           'to "that of item <id>" / "just greater/less than item <id>".')


def arm_B(base, model, ex):
    attrs = ex["cat_attrs"] + [ex["ord_attr"]]
    user = (f"Query: {ex['nl']}\n\nThe catalog items have these attributes: {attrs}.\n"
            f"The ordinal/numeric attribute is '{ex['ord_attr']}'.\n\n{OP_SPEC}")
    out = chat(base, model, [{"role": "system", "content": "Output ONLY JSON."},
                             {"role": "user", "content": user}], 200)
    op_ir = extract_json(out)
    rid = resolve_operation(op_ir, ex["items"]) if op_ir else None
    # recognition = emitted op matches gold op (structural)
    g = ex["op_ir"]
    # recognition = named the right OPERATOR (+ attr when the op uses one). filter has no attr in gold,
    # so don't penalise an emitted attr there (v1 mismeasured filter as recog-fail = artifact).
    recog = (bool(op_ir) and op_ir.get("op") == g.get("op")
             and (g.get("attr") is None or str(op_ir.get("attr")) == str(g.get("attr"))))
    return rid, recog


def score(base, model, exs, arms):
    agg = {a: [0, 0] for a in arms}
    recog = [0, 0]
    by_op = {}
    by_op_recog = {}
    for ex in exs:
        gold = ex["gold_id"]; op = ex["op"]
        by_op.setdefault(op, {a: [0, 0] for a in arms})
        by_op_recog.setdefault(op, [0, 0])
        for a in arms:
            if a == "A": rid = arm_A(base, model, ex, cot=False)
            elif a == "Acot": rid = arm_A(base, model, ex, cot=True)
            elif a == "B":
                rid, rc = arm_B(base, model, ex)
                recog[1] += 1; recog[0] += int(rc)
                by_op_recog[op][1] += 1; by_op_recog[op][0] += int(rc)
            elif a == "D": rid = resolve_operation(ex["op_ir"], ex["items"])
            else: rid = None
            ok = int(rid == gold)
            agg[a][1] += 1; agg[a][0] += ok
            by_op[op][a][1] += 1; by_op[op][a][0] += ok
    f = lambda x: f"{x[0]}/{x[1]}({x[0]/max(x[1],1):.2f})"
    print(f"=== overall (n={len(exs)}) ===")
    for a in arms:
        print(f"  arm {a:>4}: {f(agg[a])}")
    if "B" in arms:
        print(f"  op-RECOGNITION (B): {f(recog)}")
    print("=== by op ===")
    for op in by_op:
        extra = f"  recog={f(by_op_recog[op])}" if "B" in arms else ""
        print(f"  {op:>11}: " + "  ".join(f"{a}={f(by_op[op][a])}" for a in arms) + extra)
    return {"overall": {a: agg[a] for a in arms}, "recognition": recog,
            "by_op": by_op, "by_op_recog": by_op_recog}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="http://localhost:8021/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", default="A,B,D")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    exs = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    print(f"loaded {len(exs)} | model={args.model} | arms={arms}")
    res = score(args.base, args.model, exs, arms)
    if args.out:
        json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("wrote", args.out)
