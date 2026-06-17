#!/usr/bin/env python3
"""M-A eval-set extractor. From tau2 retail tasks.json, pull every task whose gold
evaluation_criteria contains an exchange_delivered_order_items action, and materialize
the offline value-accuracy case: the NL request, plus the post-gather context (old item
options + the product's full variant catalog) and the gold new_item_id per exchanged item.

This isolates the WRITE decision (variant selection) from multi-turn gathering: the model
is given exactly what it would have after get_order_details + get_product_details, and must
produce the new_item_ids (arm A: concrete) or a formal selector (arm B/C). See
M_A_PROTOTYPE_DESIGN.md §7.

Emits JSONL, one eval case per task. Tasks whose gold is non-extractable (old item or gold
variant not found in db) are skipped with a logged reason (honest denominator).
"""
import json, argparse, sys


def build_cases(tasks, db):
    cases, skipped = [], []
    for t in tasks:
        actions = (t.get("evaluation_criteria") or {}).get("actions") or []
        ex_actions = [a for a in actions if a.get("name") == "exchange_delivered_order_items"]
        if not ex_actions:
            continue
        a = ex_actions[0]["arguments"]
        order_id = a.get("order_id")
        order = db["orders"].get(order_id)
        if order is None:
            skipped.append((t["id"], f"order {order_id} missing"))
            continue
        old_ids = a.get("item_ids") or []
        new_ids = a.get("new_item_ids") or []
        if not new_ids or len(old_ids) != len(new_ids):
            skipped.append((t["id"], "empty/mismatched item ids"))
            continue
        # map each old item_id -> OrderItem, and gold new item_id -> its variant options
        exchanges, bad = [], None
        for old_id, gold_new in zip(old_ids, new_ids):
            oi = next((it for it in order["items"] if it["item_id"] == old_id), None)
            if oi is None:
                bad = f"old item {old_id} not in order"; break
            product = db["products"].get(oi["product_id"])
            if product is None or gold_new not in product["variants"]:
                bad = f"gold variant {gold_new} not in product {oi['product_id']}"; break
            catalog = [{"item_id": iid, "options": v["options"], "available": v["available"]}
                       for iid, v in product["variants"].items()]
            exchanges.append({
                "old_item_id": old_id,
                "old_item_name": oi["name"],
                "old_options": oi["options"],
                "product_id": oi["product_id"],
                "product_name": product["name"],
                "variant_catalog": catalog,
                "gold_new_item_id": gold_new,
                "gold_new_options": product["variants"][gold_new]["options"],
            })
        if bad:
            skipped.append((t["id"], bad)); continue
        nl = ((t.get("user_scenario") or {}).get("instructions") or {}).get("reason_for_call") or ""
        cases.append({
            "task_id": t["id"],
            "nl": nl,
            "order_id": order_id,
            "exchanges": exchanges,
            "gold_payment_method_id": a.get("payment_method_id"),
            "n_items": len(exchanges),
        })
    return cases, skipped


CABINS = ["basic_economy", "economy", "business"]  # ordinal class hierarchy


def _cabin_catalog():
    return [{"item_id": c, "options": {"cabin": c}, "available": True} for c in CABINS]


def build_cases_airline(tasks, db):
    """Airline content cases on the CABIN dimension (the cross-domain twin of retail variant-exchange:
    cabin is an ordinal class hierarchy basic_economy<economy<business). update_reservation_flights that
    keep the flights and change cabin = pure keep-rest SUBSTITUTE; book_reservation = CREATE. We score
    the cabin selection (honest: book is multi-field; cabin is the comparable content slot). Schema is
    ISOMORPHIC to retail cases (variant_catalog + gold_new_item_id) so tau2_op_resolver/eval reuse."""
    res = db["reservations"]
    cases, skipped = [], []
    for t in tasks:
        actions = (t.get("evaluation_criteria") or {}).get("actions") or []
        nl = ((t.get("user_scenario") or {}).get("instructions") or {}).get("reason_for_call") or ""
        for a in actions:
            name = a.get("name"); arg = a.get("arguments") or {}
            if name == "update_reservation_flights":
                rid = arg.get("reservation_id"); r = res.get(rid)
                if r is None:
                    skipped.append((t["id"], f"reservation {rid} missing")); continue
                old_cab = r.get("cabin"); new_cab = arg.get("cabin")
                if old_cab not in CABINS or new_cab not in CABINS:
                    skipped.append((t["id"], f"cabin not in enum ({old_cab}->{new_cab})")); continue
                if old_cab == new_cab:
                    skipped.append((t["id"], "cabin unchanged (no content selection)")); continue
                of = [(f.get("flight_number"), f.get("date")) for f in r.get("flights", [])]
                nf = [(f.get("flight_number"), f.get("date")) for f in arg.get("flights", [])]
                cases.append({
                    "task_id": t["id"], "nl": nl, "order_id": rid, "n_items": 1,
                    "case_op": "substitute", "flights_kept": of == nf,
                    "exchanges": [{
                        "old_item_id": old_cab, "old_item_name": f"reservation {rid}",
                        "old_options": {"cabin": old_cab}, "product_id": rid, "product_name": "reservation",
                        "variant_catalog": _cabin_catalog(),
                        "gold_new_item_id": new_cab, "gold_new_options": {"cabin": new_cab},
                    }]})
            elif name == "book_reservation":
                cab = arg.get("cabin")
                if cab not in CABINS:
                    skipped.append((t["id"], f"book cabin not in enum ({cab})")); continue
                cases.append({
                    "task_id": t["id"], "nl": nl, "order_id": None, "n_items": 1,
                    "case_op": "create", "flights_kept": False,
                    "exchanges": [{
                        "old_item_id": "", "old_item_name": "(new reservation)",
                        "old_options": {}, "product_id": None, "product_name": "reservation",
                        "variant_catalog": _cabin_catalog(),
                        "gold_new_item_id": cab, "gold_new_options": {"cabin": cab},
                    }]})
    return cases, skipped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="retail", choices=["retail", "airline"])
    ap.add_argument("--tasks", default="")
    ap.add_argument("--db", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    base = f"/home/woori/scratch/tau2-bench/data/tau2/domains/{args.domain}"
    tasks = json.load(open(args.tasks or f"{base}/tasks.json", encoding="utf-8"))
    db = json.load(open(args.db or f"{base}/db.json", encoding="utf-8"))
    out = args.out or ("/home/woori/scratch/ma_eval_cases.jsonl" if args.domain == "retail"
                       else f"/home/woori/scratch/ma_eval_cases_{args.domain}.jsonl")
    if args.domain == "airline":
        cases, skipped = build_cases_airline(tasks, db)
    else:
        cases, skipped = build_cases(tasks, db)
    with open(out, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    from collections import Counter
    opd = Counter(c.get("case_op", "exchange") for c in cases)
    print(f"[{args.domain}] extracted {len(cases)} cases -> {out} | op-types: {dict(opd)}")
    print(f"skipped {len(skipped)} non-extractable:")
    for tid, why in skipped[:20]:
        print(f"  task {tid}: {why}")
