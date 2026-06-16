#!/usr/bin/env python3
"""M-sigma v4 TRANSFER eval (held-out tau2 retail exchange) with PER-PROVENANCE split.

Fixes over m_sigma_transfer_eval.py (M_SIGMA_V4_UNION_CORPUS_DESIGN.md §2, SUBTRACT_MAP §2):
  (1) payment_method_id stored as a WALKABLE VALUE (was a dict-key -> $ref-unwalkable = the
      0.07 artifact in M_A_RESULTS §11).
  (2) $select support: model may emit new_item_ids as {"$select":{"by":{opt:val},"fallback":[...]}}
      resolved DETERMINISTICALLY against the exchange's old options + variant catalog
      (reuses ma_resolver select_variant semantics; concrete ids are NEVER a learn target).
  (3) PER-PROVENANCE 3-way split readout (binary forbidden, SUBTRACT_MAP §2):
        passive-$ref bucket = {order_id, item_ids, payment_method_id}  (re-extractable from real benches)
        $select      bucket = {new_item_ids}                            (P4/P-select; synth-exclusive)
      Single-shot harness measures these TWO; proactive-gather (control-flow) only via --withhold.
  (4) over-$ref rate: an NL-literal arg emitted as $ref (M-D negative cause (a)).
  (5) $select autopsy: fail_unresolved_criteria=LEXICAL vs fail_no_available/fail_tie=STRUCTURAL.

Scores any served model the same way; the {concrete-target vs typed-target} contrast (experiment 0)
is two TRAINED models evaluated by this one harness. Honest: this proves NEGATIVE+SIZING only
(does target-level bind the re-extractable part, and how big is the synth-exclusive residual);
"synth FIXES it" is the factorial's job (v4 §7).
"""
import json, argparse, copy


# ----------------------------- obs construction -----------------------------
def build_obs(case, withhold=None):
    """observations the model can $ref into (index order = $ref idx). withhold: a set of
    tool names to DROP (proactive-gather probe; those args become unbindable by copy)."""
    order = {"order_id": case["order_id"], "status": "delivered",
             "items": [{"item_id": e["old_item_id"], "name": e["old_item_name"],
                        "product_id": e["product_id"], "options": e["old_options"]} for e in case["exchanges"]]}
    obs = [("get_order_details", order)]
    for e in case["exchanges"]:
        obs.append(("get_product_details",
                    {"product_id": e["product_id"],
                     "variants": {v["item_id"]: {"options": v["options"], "available": v["available"]}
                                  for v in e["variant_catalog"]}}))
    # payment as a walkable VALUE (fix (1)): list of records, pm_id is a scalar field.
    obs.append(("get_user_details",
                {"payment_methods": [{"payment_method_id": case["gold_payment_method_id"], "source": "credit_card"}],
                 "orders": [case["order_id"]]}))
    if withhold:
        obs = [(t, o) for (t, o) in obs if t not in withhold]
    return obs


TOOLS = [{"type": "function", "function": {
    "name": "exchange_delivered_order_items",
    "description": "Exchange items in a delivered order for new variants of the same product.",
    "parameters": {"type": "object", "properties": {
        "order_id": {"type": "string"}, "item_ids": {"type": "array", "items": {"type": "string"}},
        "new_item_ids": {"type": "array", "items": {"type": "string"}}, "payment_method_id": {"type": "string"}},
        "required": ["order_id", "item_ids", "new_item_ids", "payment_method_id"]}}}]


def build_conv(case, obs):
    msgs = [{"role": "system", "content": "You are a tool-using assistant. Call the appropriate functions to fulfill the user request."},
            {"role": "user", "content": case["nl"]}]
    for i, (tname, out) in enumerate(obs):
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": tname, "arguments": "{}"}}]})
        msgs.append({"role": "tool", "content": json.dumps(out, ensure_ascii=False), "tool_call_id": f"c{i}"})
    msgs.append({"role": "user", "content": "Now call exchange_delivered_order_items to perform the requested exchange."})
    return msgs


# ----------------------------- resolution -----------------------------
def _walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, obj


def _norm(s):
    return str(s).strip().lower()


def resolve_ref(ref, obs):
    """{$ref:'idx#path'} -> scalar at that prior-output path."""
    try:
        idx_s, path = ref.split("#", 1); idx = int(idx_s)
        for p, v in _walk(obs[idx][1]):
            if p == path:
                return v
    except Exception:
        return None
    return None


def resolve_select(spec, old_options, variant_catalog):
    """$select.by (then fallback) applied as OVERRIDES on old_options; pick the AVAILABLE variant
    whose options == target. Deterministic (ma_resolver.select_variant semantics on a catalog list).
    Returns (item_id|None, diag.kind in {ok,ok_fallback,fail_unresolved_criteria,fail_no_available,fail_tie})."""
    # value-space across the catalog -> reverse-match loose keys
    space = {}
    for v in variant_catalog:
        for k, val in v["options"].items():
            space.setdefault(k, set()).add(val)

    def normalize(crit):
        # known key + value present in that key's value-space -> accept; else reverse-match the
        # VALUE to its unique owner key; else unresolved = LEXICAL residual (synonym not grounded).
        norm, unresolved = {}, []
        for k, val in crit.items():
            if k in space and any(_norm(val) == _norm(x) for x in space[k]):
                norm[k] = val; continue
            owners = [ok for ok, vals in space.items() if any(_norm(val) == _norm(x) for x in vals)]
            if len(owners) == 1:
                norm[owners[0]] = val
            else:
                unresolved.append((k, val))
        return norm, unresolved

    steps = [spec.get("by", {})] + list(spec.get("fallback", []) or [])
    last_unresolved = None
    for i, override in enumerate(steps):
        norm, unresolved = normalize(override)
        if unresolved:
            last_unresolved = unresolved
            continue
        target = {**old_options, **norm}
        hits = [v["item_id"] for v in variant_catalog if v["available"] and v["options"] == target]
        if len(hits) == 1:
            return hits[0], ("ok" if i == 0 else "ok_fallback")
        if len(hits) > 1:
            return None, "fail_tie"
    if last_unresolved is not None:
        return None, "fail_unresolved_criteria"
    return None, "fail_no_available"


def classify(val):
    """provenance of an emitted arg value: 'literal' | 'ref' | 'select' (lists -> by element)."""
    if isinstance(val, dict):
        if "$ref" in val:
            return "ref"
        if "$select" in val:
            return "select"
        return "literal"
    if isinstance(val, list):
        kinds = {classify(x) for x in val}
        if "select" in kinds:
            return "select"
        if "ref" in kinds:
            return "ref"
        return "literal"
    return "literal"


def resolve_arg(key, val, case, obs):
    """Resolve one emitted arg to concrete value(s)."""
    if isinstance(val, dict) and "$ref" in val:
        return resolve_ref(val["$ref"], obs)
    if isinstance(val, list):
        out = []
        for i, x in enumerate(val):
            if isinstance(x, dict) and "$select" in x and i < len(case["exchanges"]):
                e = case["exchanges"][i]
                rid, _ = resolve_select(x["$select"], e["old_options"], e["variant_catalog"])
                out.append(rid)
            elif isinstance(x, dict) and "$ref" in x:
                out.append(resolve_ref(x["$ref"], obs))
            else:
                out.append(x)
        return out
    if isinstance(val, dict) and "$select" in val and case["exchanges"]:
        e = case["exchanges"][0]
        rid, _ = resolve_select(val["$select"], e["old_options"], e["variant_catalog"])
        return [rid]
    return val


# ----------------------------- scoring -----------------------------
PASSIVE = ("order_id", "item_ids", "payment_method_id")  # re-extractable buckets
SELECT = ("new_item_ids",)                                # P4/synth-exclusive bucket


def chat(base, model, messages):
    import requests
    r = requests.post(f"{base}/chat/completions",
                      json={"model": model, "messages": messages, "tools": TOOLS, "tool_choice": "auto",
                            "max_tokens": 512, "temperature": 0.0}, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def score(base, model, cases, tag):
    agg = {k: [0, 0] for k in ("order_id", "item_ids", "new_item_ids", "payment_method_id")}
    bucket = {"passive_ref": [0, 0], "select": [0, 0], "all": [0, 0]}
    emit = {"ref": 0, "select": 0, "literal": 0}
    over_ref = 0            # NL-literal arg emitted as $ref
    sel_fail = {"fail_unresolved_criteria": 0, "fail_no_available": 0, "fail_tie": 0, "ok": 0, "ok_fallback": 0}
    for case, obs in cases:
        gold = {"order_id": case["order_id"], "item_ids": [e["old_item_id"] for e in case["exchanges"]],
                "new_item_ids": [e["gold_new_item_id"] for e in case["exchanges"]],
                "payment_method_id": case["gold_payment_method_id"]}
        msg = chat(base, model, build_conv(case, obs))
        args = {}
        if msg.get("tool_calls"):
            try: args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
            except Exception: args = {}
        nl = _norm(case["nl"])
        # provenance + over-$ref + select autopsy
        for k, v in args.items():
            kind = classify(v)
            emit[kind] = emit.get(kind, 0) + 1
            if kind == "ref":
                gv = gold.get(k)
                if isinstance(gv, str) and _norm(gv) in nl:
                    over_ref += 1   # was literally in the user NL -> should be literal, not $ref
            if k == "new_item_ids":
                for i, x in enumerate(v if isinstance(v, list) else [v]):
                    if isinstance(x, dict) and "$select" in x and i < len(case["exchanges"]):
                        e = case["exchanges"][i]
                        _, kd = resolve_select(x["$select"], e["old_options"], e["variant_catalog"])
                        sel_fail[kd] = sel_fail.get(kd, 0) + 1
        # resolve + per-arg score
        resolved = {k: resolve_arg(k, v, case, obs) for k, v in args.items()}
        allok = True
        for k in agg:
            ok = resolved.get(k) == gold[k]
            agg[k][1] += 1; agg[k][0] += int(ok); allok = allok and ok
            b = "passive_ref" if k in PASSIVE else "select"
            bucket[b][1] += 1; bucket[b][0] += int(ok)
        bucket["all"][1] += 1; bucket["all"][0] += int(allok)
    n = len(cases)
    f = lambda a: f"{a[0]}/{a[1]}({a[0]/max(a[1],1):.2f})"
    print(f"=== V4 SPLIT [{tag}] n={n} ===")
    print("  per-arg : " + "  ".join(f"{k}={f(a)}" for k, a in agg.items()))
    print(f"  BUCKETS : passive_ref={f(bucket['passive_ref'])}  $select={f(bucket['select'])}  all={f(bucket['all'])}")
    print(f"  emit    : ref={emit['ref']} select={emit['select']} literal={emit['literal']}  over_$ref={over_ref}")
    print(f"  select_autopsy: {sel_fail}")
    return {"tag": tag, "n": n, "per_arg": agg, "buckets": bucket, "emit": emit,
            "over_ref": over_ref, "select_autopsy": sel_fail}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="/home/woori/scratch/ma_eval_cases.jsonl")
    ap.add_argument("--base", default="http://localhost:8015/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--withhold", default="", help="comma tool names to drop (proactive-gather probe)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    wh = set(t.strip() for t in args.withhold.split(",") if t.strip())
    cases = [(c, build_obs(c, withhold=wh)) for c in (json.loads(l) for l in open(args.cases, encoding="utf-8"))]
    print(f"loaded {len(cases)} tau2 exchange cases" + (f" (withhold={sorted(wh)})" if wh else ""))
    res = score(args.base, args.model, cases, args.tag)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as w:
            json.dump(res, w, ensure_ascii=False, indent=2)
        print("wrote", args.out)
