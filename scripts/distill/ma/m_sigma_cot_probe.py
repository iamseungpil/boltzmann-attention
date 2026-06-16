#!/usr/bin/env python3
"""§4 CoT probe — capability vs artifact for P4 fallback formalization (P4_FALLBACK_IR_DESIGN §4).

2-STAGE (review boost 4): free CoT FIRST -> SEPARATE extraction. Forcing CoT->schema in one
shot lets forced-JSON distortion (CRANE) contaminate the probe; staging isolates capability.

Cells (each scored on new_item_ids accuracy vs gold, n=29 held-out tau2):
  P-lit     : free CoT -> extract final item_id(s) as a literal list   (in-head selection)
  P-old-CoT : free CoT -> extract {by, fallback} spec                  (current schema)
  P-new-CoT : free CoT -> extract ordered [{set:{..}}] ops             (proposed IR, set-only)

REGIME GATE (review boost 1): each cell also runs with --withhold get_product_details, which
DROPS the variant catalog from what the MODEL sees (the resolver still has it). Catalog-given
favors in-head literal (the single-turn artifact); catalog-withheld FORCES structure -> the
regime where decomposition pays. Readout of "no training needed" is scoped to catalog-GIVEN.

Reuses build_obs / TOOLS from m_sigma_transfer_eval_v4 (read-only import). The set-only IR
resolver (resolve_setops) lives here until folded into ma_resolver (§8-3).
"""
import json, argparse, re
from m_sigma_transfer_eval_v4 import build_obs, resolve_select, _norm


def chat(base, model, messages, tools=None, max_tokens=512):
    import requests
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
    if tools:
        body["tools"] = tools; body["tool_choice"] = "auto"
    r = requests.post(f"{base}/chat/completions", json=body, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def conv_prefix(case, obs):
    """system + user request + the (visible) tool observations."""
    msgs = [{"role": "system", "content": "You are a tool-using assistant helping with an order exchange."},
            {"role": "user", "content": case["nl"]}]
    for i, (tname, out) in enumerate(obs):
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": f"c{i}", "type": "function",
                                     "function": {"name": tname, "arguments": "{}"}}]})
        msgs.append({"role": "tool", "content": json.dumps(out, ensure_ascii=False), "tool_call_id": f"c{i}"})
    return msgs


STAGE1 = ("Think step by step about the exchange. Identify: (a) which attribute(s) the user "
          "wants to CHANGE and to what value; (b) any CONDITIONAL FALLBACK ('if not available, "
          "then ...') — note this describes a backup, it is NOT a new constraint; (c) which "
          "attributes stay the SAME. Do NOT output the final answer yet — reason only.")

STAGE2 = {
    "P-lit":     ('Now output ONLY a JSON object {"new_item_ids": ["<item_id>", ...]} with the '
                  'final chosen variant item_id(s).'),
    "P-old-CoT": ('Now output ONLY a JSON object {"new_item_ids": [{"$select": {"by": {"<attr>": '
                  '"<val>"}, "fallback": [{"<attr>": "<val>"}]}}]} — criteria, not ids. One $select '
                  'per item being exchanged.'),
    "P-new-CoT": ('Now output ONLY a JSON object {"new_item_ids": [{"$select": [{"set": {"<attr>": '
                  '"<val>"}}, {"set": {"<attr>": "<val>"}}]}]} — an ORDERED list of set-operations '
                  '(1st = primary change, later = fallback deltas applied on top), criteria not ids. '
                  'One $select per item being exchanged.'),
}


def extract_json(text):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        # tolerate trailing prose / code fences
        s = m.group(0)
        for end in range(len(s), 0, -1):
            try:
                return json.loads(s[:end])
            except Exception:
                continue
    return None


def resolve_setops(ops, old_options, variant_catalog):
    """ordered set-ops folded cumulatively; first AVAILABLE unique match wins (proposed IR §3.2)."""
    cur = dict(old_options)
    # value-space for loose key reverse-match (mirror resolve_select.normalize tolerance)
    space = {}
    for v in variant_catalog:
        for k, val in v["options"].items():
            space.setdefault(k, set()).add(val)

    def apply_set(state, d):
        out = dict(state)
        for k, val in d.items():
            if k in space:
                out[k] = val
            else:                                   # reverse-match value -> unique owner key
                owners = [ok for ok, vals in space.items() if any(_norm(val) == _norm(x) for x in vals)]
                if len(owners) == 1:
                    out[owners[0]] = val
                else:
                    out[k] = val
        return out

    for op in ops:
        d = op.get("set", op) if isinstance(op, dict) else {}
        cur = apply_set(cur, d)
        hits = [v["item_id"] for v in variant_catalog if v["available"] and v["options"] == cur]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            return None                              # tie (set-only should be rare)
    return None


def resolve_new_item(cell, val, case):
    """resolve the emitted new_item_ids `val` (a list) to concrete ids per the cell's IR."""
    out = []
    items = val if isinstance(val, list) else [val]
    for i, x in enumerate(items):
        e = case["exchanges"][i] if i < len(case["exchanges"]) else case["exchanges"][-1]
        if cell == "P-lit":
            out.append(x if isinstance(x, str) else None)
        elif isinstance(x, dict) and "$select" in x:
            spec = x["$select"]
            if cell == "P-new-CoT" and isinstance(spec, list):
                out.append(resolve_setops(spec, e["old_options"], e["variant_catalog"]))
            elif isinstance(spec, dict):             # P-old-CoT by/fallback
                rid, _ = resolve_select(spec, e["old_options"], e["variant_catalog"])
                out.append(rid)
            else:
                out.append(None)
        else:
            out.append(None)
    return out


def run_cell(base, model, cases, cell, withhold):
    wh = {"get_product_details"} if withhold else None
    ok = 0
    n = 0
    for case in cases:
        obs = build_obs(case, withhold=wh)
        gold = [e["gold_new_item_id"] for e in case["exchanges"]]
        msgs = conv_prefix(case, obs)
        msgs.append({"role": "user", "content": STAGE1})
        r1 = chat(base, model, msgs, max_tokens=512)
        reasoning = r1.get("content") or ""
        msgs.append({"role": "assistant", "content": reasoning})
        msgs.append({"role": "user", "content": STAGE2[cell]})
        r2 = chat(base, model, msgs, max_tokens=256)
        parsed = extract_json(r2.get("content") or "")
        val = (parsed or {}).get("new_item_ids")
        resolved = resolve_new_item(cell, val, case) if val is not None else [None] * len(gold)
        n += 1
        ok += int(resolved == gold)
    return ok, n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="/home/woori/scratch/ma_eval_cases.jsonl")
    ap.add_argument("--base", default="http://localhost:8015/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cells", default="P-lit,P-old-CoT,P-new-CoT")
    ap.add_argument("--regimes", default="given,withheld", help="given,withheld (catalog visibility)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    cases = [json.loads(l) for l in open(args.cases, encoding="utf-8")]
    print(f"loaded {len(cases)} tau2 exchange cases | model={args.model}")
    res = {}
    for cell in [c.strip() for c in args.cells.split(",") if c.strip()]:
        for regime in [r.strip() for r in args.regimes.split(",") if r.strip()]:
            ok, n = run_cell(args.base, args.model, cases, cell, withhold=(regime == "withheld"))
            res[f"{cell}/{regime}"] = [ok, n]
            print(f"  {cell:12s} [{regime:8s}] new_item_ids = {ok}/{n} ({ok/max(n,1):.2f})")
    print("baseline refs: base-no-CoT-lit/given=0.48  M0-forced/given=0.41")
    if args.out:
        json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("wrote", args.out)
