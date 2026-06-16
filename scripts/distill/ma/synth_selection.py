#!/usr/bin/env python3
"""M-sigma v4 — PURE-ABSTRACT selection-by-criteria synthesis (P-select / matrix P4).

Generates training triples for the pure-synth 2^3 factorial (M_SIGMA_V4_UNION_CORPUS_DESIGN §4,
M_SIGMA_V3_TRANSFER_FACTORIAL_DESIGN). The model learns the DOMAIN-INVARIANT structural
operation "exchange this item for the SAME item with these option changes, keep the rest, fall
back if unavailable" — emitted as a provenance-typed spec ($select/$ref/literal); a DETERMINISTIC
resolver fills concrete ids (concrete is NEVER a learn target; feedback-nl-formalize-llm-...).

The conversation structure MIRRORS the held-out tau2 harness (m_sigma_transfer_eval_v4.build_conv):
  system, user(NL request), [get_order_details, get_product_details, get_user_details] obs as
  assistant-call+tool-output pairs, user("now call exchange..."), assistant target = typed spec.
Only the SCHEMA CONTENT is abstract/random, so the learned structure transfers to tau2's real
options. Round-trip validated against the SAME resolver the eval uses (resolve_select) — only
clean round-trips are emitted (honest denominator).

KNOBS (the 3 factorial axes; arm->flags via ma_factorial_batch.sh):
  --iso  1/0  isotropization: random schema/attr-names/value-vocab PER EXAMPLE (surface-group
              coverage) vs fixed names. (Relation change/keep/fallback ALWAYS preserved.)
  --nl   1/0  NL paraphrase of the change/keep/fallback request vs literal attr=val.
  --prov 1/0  provenance MIX (order_id/item_ids/payment as $ref-from-obs or literal; new_item_ids
              $select) vs $select-only degenerate (threading args literal, no $ref discrimination).
"""
import json, argparse, random, string
from m_sigma_transfer_eval_v4 import resolve_select, resolve_ref, _walk  # single-source resolver


def _tok(rng, n=6):
    return "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(n))


def gen_schema(rng, iso, K):
    """K option keys, each with a small value-vocab. iso -> random names/values per example."""
    schema = {}
    for i in range(K):
        key = f"attr_{_tok(rng)}" if iso else f"opt_{i}"
        nv = rng.randint(2, 4)
        vals = [f"v_{_tok(rng,4)}" if iso else f"opt{i}_val{j}" for j in range(nv)]
        schema[key] = vals
    return schema


def gen_catalog(rng, schema, M):
    """M variants = option-value combos (unique) + random item_id + available flag."""
    keys = list(schema)
    seen, variants = set(), []
    tries = 0
    while len(variants) < M and tries < M * 20:
        tries += 1
        opts = {k: rng.choice(schema[k]) for k in keys}
        sig = tuple(sorted(opts.items()))
        if sig in seen:
            continue
        seen.add(sig)
        variants.append({"item_id": f"I{_tok(rng,8)}", "options": opts, "available": True})
    return variants


def _nl_request(changes, fallback, item_label, nl):
    """NL paraphrase (nl=1) or literal attr=val (nl=0) of change/keep/fallback."""
    ch = ", ".join(f"{k} to {v}" for k, v in changes.items())
    if nl:
        s = f"I'd like to exchange my {item_label} for the same item but change {ch}, keeping everything else the same."
        if fallback:
            fb = ", ".join(f"{k} to {v}" for k, v in fallback[0].items())
            s += f" If that exact variant isn't available, change {fb} instead."
        return s
    s = f"exchange {item_label}: set {'; '.join(f'{k}={v}' for k,v in changes.items())} (keep rest)"
    if fallback:
        s += f" | fallback: {'; '.join(f'{k}={v}' for k,v in fallback[0].items())}"
    return s


def build_example(rng, iso, nl, prov, K=3, M=8):
    """Return (triple|None, ok_roundtrip). triple mirrors the tau2 harness conversation with an
    abstract schema; assistant target = provenance-typed spec; gold resolved deterministically."""
    schema = gen_schema(rng, iso, K)
    catalog = gen_catalog(rng, schema, M)
    if len(catalog) < 3:
        return None, False
    keys = list(schema)
    old = rng.choice(catalog)
    old_options = dict(old["options"])

    # pick 1-2 option keys to CHANGE to a DIFFERENT value (keep the rest) -> defines the target.
    n_ch = rng.randint(1, min(2, K))
    ch_keys = rng.sample(keys, n_ch)
    changes = {}
    for k in ch_keys:
        alt = [v for v in schema[k] if v != old_options[k]]
        if not alt:
            return None, False
        changes[k] = rng.choice(alt)
    target_primary = {**old_options, **changes}

    # ensure a UNIQUE available variant for the (primary OR fallback) target; build fallback by
    # relaxing one change. Control difficulty: primary may be made unavailable to exercise fallback.
    def find_avail(opts):
        hits = [v["item_id"] for v in catalog if v["available"] and v["options"] == opts]
        return hits[0] if len(hits) == 1 else None

    fallback = []
    # fallback = relax (drop) one changed key back toward old value, change a different sense
    if n_ch >= 1 and rng.random() < 0.5:
        relax_k = rng.choice(ch_keys)
        fb = dict(changes)
        alt2 = [v for v in schema[relax_k] if v != changes[relax_k] and v != old_options[relax_k]]
        if alt2:
            fb[relax_k] = rng.choice(alt2)
            fallback = [fb]

    # decide gold: try primary; if we deliberately drop primary availability, use fallback.
    gold_id = find_avail(target_primary)
    use_fallback = False
    if gold_id is None and fallback:
        gold_id = find_avail({**old_options, **fallback[0]})
        use_fallback = True
    if gold_id is None:
        # make primary unique+available
        for v in catalog:
            if v["options"] == target_primary:
                v["available"] = True
                gold_id = v["item_id"]
                break
        if gold_id is None:
            old2 = dict(catalog[0]["options"]); old2.update(target_primary)
            catalog[0]["options"] = target_primary; catalog[0]["available"] = True
            gold_id = catalog[0]["item_id"]
        # disambiguate: drop any other variant equal to target
        for v in catalog:
            if v["item_id"] != gold_id and v["options"] == target_primary:
                v["available"] = False

    order_id = f"#O{_tok(rng,7)}"
    pm_id = f"pm_{_tok(rng,6)}"
    item_label = "item" if not iso else f"item_{_tok(rng,4)}"

    # ---- observations (idx order = $ref idx), MIRRORS harness v4 build_obs ----
    order = {"order_id": order_id, "status": "delivered",
             "items": [{"item_id": old["item_id"], "name": item_label,
                        "product_id": "P0", "options": old_options}]}
    product = {"product_id": "P0",
               "variants": {v["item_id"]: {"options": v["options"], "available": v["available"]} for v in catalog}}
    user = {"payment_methods": [{"payment_method_id": pm_id, "source": "credit_card"}], "orders": [order_id]}
    obs = [("get_order_details", order), ("get_product_details", product), ("get_user_details", user)]

    # ---- provenance-typed target spec ----
    select_spec = {"$select": {"from": "1", "by": changes}}
    if fallback:
        select_spec["$select"]["fallback"] = fallback
    if prov:  # MIX: threading args as $ref into obs; selection as $select
        args = {"order_id": {"$ref": "0#.order_id"},
                "item_ids": [{"$ref": "0#.items[0].item_id"}],
                "new_item_ids": [select_spec],
                "payment_method_id": {"$ref": "2#.payment_methods[0].payment_method_id"}}
    else:     # $select-only degenerate: threading args literal (no $ref discrimination taught)
        args = {"order_id": order_id, "item_ids": [old["item_id"]],
                "new_item_ids": [select_spec], "payment_method_id": pm_id}

    # ---- round-trip: resolve spec against obs/catalog -> must equal gold ----
    rid, _ = resolve_select(select_spec["$select"], old_options,
                            [{"item_id": v["item_id"], "options": v["options"], "available": v["available"]} for v in catalog])
    oid = resolve_ref("0#.order_id", obs) if prov else order_id
    pmid = resolve_ref("2#.payment_methods[0].payment_method_id", obs) if prov else pm_id
    if rid != gold_id or oid != order_id or pmid != pm_id:
        return None, False

    # ---- conversation (mirror harness v4 build_conv) ----
    nlreq = _nl_request(changes, fallback, item_label, nl)
    msgs = [{"role": "system", "content": "You are a tool-using assistant. Call the appropriate functions to fulfill the user request."},
            {"role": "user", "content": nlreq}]
    for i, (tname, out) in enumerate(obs):
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": tname, "arguments": "{}"}}]})
        msgs.append({"role": "tool", "content": json.dumps(out, ensure_ascii=False), "tool_call_id": f"c{i}"})
    msgs.append({"role": "user", "content": "Now call exchange_delivered_order_items to perform the requested exchange."})
    msgs.append({"role": "assistant", "content": None,
                 "tool_calls": [{"id": "call0", "type": "function",
                                 "function": {"name": "exchange_delivered_order_items",
                                              "arguments": json.dumps(args, ensure_ascii=False)}}]})
    tools = [{"type": "function", "function": {
        "name": "exchange_delivered_order_items",
        "description": "Exchange items in a delivered order for new variants of the same product.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"}, "item_ids": {"type": "array", "items": {"type": "string"}},
            "new_item_ids": {"type": "array", "items": {"type": "string"}}, "payment_method_id": {"type": "string"}},
            "required": ["order_id", "item_ids", "new_item_ids", "payment_method_id"]}}}]
    triple = {"tools": tools, "messages": msgs, "supervise": "assistant",
              "_meta": {"synth": "v4_pselect", "iso": iso, "nl": nl, "prov": prov, "fallback": use_fallback}}
    return triple, True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/woori/scratch/synth/pselect.jsonl")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--iso", type=int, default=1)
    ap.add_argument("--nl", type=int, default=1)
    ap.add_argument("--prov", type=int, default=1)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validate_only", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    n_emit = n_try = n_fb = 0
    import os
    if not args.validate_only:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    w = None if args.validate_only else open(args.out, "w", encoding="utf-8")
    while n_emit < args.n and n_try < args.n * 30:
        n_try += 1
        triple, ok = build_example(rng, args.iso, args.nl, args.prov, args.K, args.M)
        if not ok:
            continue
        n_fb += int(triple["_meta"]["fallback"])
        if w:
            w.write(json.dumps(triple, ensure_ascii=False) + "\n")
        n_emit += 1
    if w:
        w.close()
    print(f"emitted={n_emit}/{args.n} (tries={n_try}) fallback={n_fb} "
          f"iso={args.iso} nl={args.nl} prov={args.prov} K={args.K} M={args.M} -> "
          f"{'(validate_only)' if args.validate_only else args.out}")
