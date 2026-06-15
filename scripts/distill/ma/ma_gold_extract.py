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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="/home/woori/scratch/tau2-bench/data/tau2/domains/retail/tasks.json")
    ap.add_argument("--db", default="/home/woori/scratch/tau2-bench/data/tau2/domains/retail/db.json")
    ap.add_argument("--out", default="/home/woori/scratch/ma_eval_cases.jsonl")
    args = ap.parse_args()
    tasks = json.load(open(args.tasks, encoding="utf-8"))
    db = json.load(open(args.db, encoding="utf-8"))
    cases, skipped = build_cases(tasks, db)
    with open(args.out, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    n_multi = sum(1 for c in cases if c["n_items"] > 1)
    print(f"extracted {len(cases)} exchange cases ({n_multi} multi-item) -> {args.out}")
    print(f"skipped {len(skipped)} non-extractable:")
    for tid, why in skipped[:20]:
        print(f"  task {tid}: {why}")
