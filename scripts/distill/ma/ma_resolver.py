#!/usr/bin/env python3
"""M-A deterministic resolver: formal selector (abstract criteria) -> concrete
exchange_delivered_order_items(order_id, item_ids, new_item_ids, payment_method_id).

Thesis split (feedback-nl-formalize-llm-selection-deterministic):
  LLM emits provenance-typed SELECTOR (which options / preference order) — NO concrete ids.
  This module deterministically RESOLVES options->item_id, name->item_id, source->pm_id
  against the tau2 retail db.json. It faithfully resolves whatever criteria it is given;
  wrong criteria (e.g. NL synonym "Google Home" not mapped to option value "Google
  Assistant") surface as resolver FAIL, NOT silent wrong answers — that is the M-A
  diagnostic (fabrication vs reasoning, see M_A_PROTOTYPE_DESIGN.md §0/§7).

Formal selector schema (what the LLM emits per M_A_PROTOTYPE_DESIGN.md §4):
  {"order_ref": {"order_id_hint": "#W..."},
   "exchanges": [{"old_item": {"item_name": "mechanical keyboard"},
                  "desired_variant": {"select_by": {<opt>: <val>, ...},
                                      "fallback": [{<opt>: <val>}, ...]}}],
   "payment_ref": {"source": "original" | "credit_card" | "gift_card" | "paypal"}}

select_by / fallback semantics = "exchange for the SAME item but with these changes":
  target = {**old_item.options, **overrides}; unspecified option keys inherit the old
  item's value. Keys may be loose ("switch" -> "switch type"); values are reverse-matched
  to the owning option key. A criterion value not present in ANY option value-space is
  unresolvable -> FAIL (the reasoning residual).
"""
import json, argparse, sys
from typing import Optional


def load_db(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _norm(s):
    return str(s).strip().lower()


def match_order_item(order, item_name):
    """Find the OrderItem whose product name matches item_name (substring, ci)."""
    q = _norm(item_name)
    hits = [it for it in order["items"] if q in _norm(it["name"]) or _norm(it["name"]) in q]
    if not hits:
        return None, f"no order item matching name '{item_name}'"
    if len({it["item_id"] for it in hits}) > 1:
        # ambiguous only if they are genuinely different items
        names = {it["name"] for it in hits}
        if len(names) > 1:
            return None, f"ambiguous order item for '{item_name}': {sorted(names)}"
    return hits[0], None


def _option_value_space(product):
    """key -> set(values) across all variants of the product."""
    space = {}
    for v in product["variants"].values():
        for k, val in v["options"].items():
            space.setdefault(k, set()).add(val)
    return space


def normalize_criteria(criteria, product):
    """Map a loose {key|loose: value} dict onto the product's real option keys via
    value-reverse-match. Returns (normalized {real_key: value}, unresolved [values])."""
    space = _option_value_space(product)
    keys = set(space.keys())
    norm, unresolved = {}, []
    for k, val in criteria.items():
        if k in keys:
            norm[k] = val
            continue
        # reverse-match the value to the option key whose value-space contains it
        owners = [ok for ok, vals in space.items() if any(_norm(val) == _norm(x) for x in vals)]
        if len(owners) == 1:
            norm[owners[0]] = val
        elif len(owners) > 1:
            unresolved.append((k, val, f"value '{val}' ambiguous across keys {owners}"))
        else:
            unresolved.append((k, val, f"value '{val}' not in any option value-space"))
    return norm, unresolved


def _find_variant(product, target_options, require_available=True):
    """Return item_id of the variant whose options == target_options (and available)."""
    matches = []
    for iid, v in product["variants"].items():
        if require_available and not v["available"]:
            continue
        if v["options"] == target_options:
            matches.append(iid)
    return matches


def select_variant(product, old_options, desired_variant):
    """Apply select_by (then fallback overrides) on top of old_options; return
    (item_id|None, diagnostic). diagnostic.kind in
    {ok, ok_fallback, fail_unresolved_criteria, fail_no_available, fail_tie}."""
    select_by = desired_variant.get("select_by", {})
    fallback = desired_variant.get("fallback", []) or []

    # Build preference list of override dicts: primary first, then each fallback.
    override_steps = [select_by] + list(fallback)
    last_unresolved = None
    for i, override in enumerate(override_steps):
        norm, unresolved = normalize_criteria(override, product)
        if unresolved:
            last_unresolved = unresolved
            continue  # this criteria step is un-formalizable; try next preference
        target = {**old_options, **norm}
        hits = _find_variant(product, target, require_available=True)
        if len(hits) == 1:
            return hits[0], {"kind": "ok" if i == 0 else "ok_fallback", "step": i, "target": target}
        if len(hits) > 1:
            return None, {"kind": "fail_tie", "step": i, "target": target, "candidates": hits}
        # zero available matches at this step -> fall through to next preference
    if last_unresolved is not None:
        return None, {"kind": "fail_unresolved_criteria", "detail": last_unresolved}
    return None, {"kind": "fail_no_available", "tried": override_steps, "old_options": old_options}


def resolve_payment(order, db, payment_ref):
    if not payment_ref:
        return None, "no payment_ref"
    src = _norm(payment_ref.get("source", ""))
    if src in ("original", "same", ""):
        ph = order.get("payment_history") or []
        if ph:
            return ph[0]["payment_method_id"], None
        return None, "no payment_history for 'original'"
    user = db["users"][order["user_id"]]
    hits = [pid for pid, pm in user["payment_methods"].items() if _norm(pm.get("source")) == src]
    if len(hits) == 1:
        return hits[0], None
    if len(hits) > 1:
        return None, f"ambiguous payment source '{src}': {hits}"
    return None, f"no payment method with source '{src}'"


def resolve_exchange(formal, db):
    """Resolve a formal selector into a concrete exchange call. Returns (call|None, report)."""
    report = {"steps": []}
    order_id = formal["order_ref"]["order_id_hint"]
    order = db["orders"].get(order_id)
    if order is None:
        return None, {"error": f"order {order_id} not found"}
    if _norm(order["status"]) != "delivered":
        return None, {"error": f"order {order_id} not delivered (status={order['status']})"}
    item_ids, new_item_ids = [], []
    for ex in formal["exchanges"]:
        old, err = match_order_item(order, ex["old_item"]["item_name"])
        if old is None:
            return None, {"error": err, "steps": report["steps"]}
        product = db["products"][old["product_id"]]
        new_id, diag = select_variant(product, old["options"], ex["desired_variant"])
        report["steps"].append({"old_item_id": old["item_id"], "product": product["name"], "diag": diag})
        if new_id is None:
            return None, {"error": "variant selection failed", "steps": report["steps"]}
        item_ids.append(old["item_id"])
        new_item_ids.append(new_id)
    pm_id, perr = resolve_payment(order, db, formal.get("payment_ref"))
    if pm_id is None:
        return None, {"error": f"payment resolve failed: {perr}", "steps": report["steps"]}
    call = {"order_id": order_id, "item_ids": item_ids, "new_item_ids": new_item_ids,
            "payment_method_id": pm_id}
    return call, report


# ----------------------------- unit tests -----------------------------
def _task0_formal():
    """The selector an accurate LLM would emit for tasks.json task 0 (Yusuf Rossi).
    Note: 'Google Home' is correctly mapped to option value 'Google Assistant' (the
    reasoning step); color is left unspecified so it inherits the old item's value."""
    return {
        "order_ref": {"order_id_hint": "#W2378156"},
        "exchanges": [
            {"old_item": {"item_name": "mechanical keyboard"},
             "desired_variant": {"select_by": {"switch type": "clicky", "backlight": "RGB", "size": "full size"},
                                 "fallback": [{"switch type": "clicky", "backlight": "none", "size": "full size"}]}},
            {"old_item": {"item_name": "smart thermostat"},
             "desired_variant": {"select_by": {"compatibility": "Google Assistant"}}},
        ],
        "payment_ref": {"source": "original"},
    }


def run_tests(db):
    GOLD = {"order_id": "#W2378156", "item_ids": ["1151293680", "4983901480"],
            "new_item_ids": ["7706410293", "7747408585"], "payment_method_id": "credit_card_9513926"}
    fails = []

    # T1: task 0 gold reproduction
    call, rep = resolve_exchange(_task0_formal(), db)
    ok = call is not None and call["new_item_ids"] == GOLD["new_item_ids"] and \
        call["item_ids"] == GOLD["item_ids"] and call["payment_method_id"] == GOLD["payment_method_id"]
    print(f"[T1 task0 gold] {'PASS' if ok else 'FAIL'} call={call}")
    if not ok:
        fails.append(("T1", call, rep))

    # T2: loose key ('switch' instead of 'switch type') must still resolve via value-reverse-match
    f2 = _task0_formal()
    f2["exchanges"][0]["desired_variant"] = {"select_by": {"switch": "clicky", "backlight": "RGB", "size": "full size"},
                                             "fallback": [{"switch": "clicky", "backlight": "none", "size": "full size"}]}
    call2, _ = resolve_exchange(f2, db)
    ok2 = call2 is not None and call2["new_item_ids"][0] == "7706410293"
    print(f"[T2 loose-key reverse-match] {'PASS' if ok2 else 'FAIL'} kbd={call2['new_item_ids'][0] if call2 else None}")
    if not ok2:
        fails.append(("T2", call2))

    # T3 (ADVERSARIAL, review): perturb db so the size-preserving fallback is actually exercised.
    # Make clicky+none+full (7706410293) UNAVAILABLE and clicky+none+80% (9665000388) AVAILABLE.
    # Correct (size-preserving) fallback => no clicky+none+full available => FAIL (honest), NOT 9665000388.
    # A size-DROPPING fallback bug would wrongly pick 9665000388.
    import copy
    db2 = copy.deepcopy(db)
    kb = db2["products"]["1656367028"]["variants"]
    kb["7706410293"]["available"] = False
    kb["9665000388"]["available"] = True  # clicky+none+80%
    call3, rep3 = resolve_exchange(_task0_formal(), db2)
    # size-preserving fallback => keyboard step fails => whole resolve fails (returns None)
    diag3 = rep3.get("steps", [{}])[0].get("diag", {}) if call3 is None else None
    ok3 = call3 is None  # must NOT silently pick the wrong-size 9665000388
    print(f"[T3 adversarial size-preserve] {'PASS' if ok3 else 'FAIL'} call={call3} diag={diag3}")
    if not ok3:
        fails.append(("T3", call3, rep3))

    # T4 (reasoning residual): if the LLM fails to map 'Google Home'->'Google Assistant',
    # the resolver must FAIL (unresolved criteria), not silently mis-resolve.
    f4 = _task0_formal()
    f4["exchanges"][1]["desired_variant"] = {"select_by": {"compatibility": "Google Home"}}
    call4, rep4 = resolve_exchange(f4, db)
    ok4 = call4 is None  # 'Google Home' not in option value-space -> FAIL
    print(f"[T4 unmapped-synonym FAIL] {'PASS' if ok4 else 'FAIL'} call={call4}")
    if not ok4:
        fails.append(("T4", call4, rep4))

    print(f"\n=== {4 - len(fails)}/4 unit tests passed ===")
    return len(fails) == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/home/woori/scratch/tau2-bench/data/tau2/domains/retail/db.json")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    db = load_db(args.db)
    if args.test:
        sys.exit(0 if run_tests(db) else 1)
    print("loaded db:", {k: len(v) for k, v in db.items()})
