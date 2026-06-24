#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage-1 수동 카탈로그용 원자료 덤프 (15 gap task)."""
import json, os
from escape_scope_diag import load_json, compute_gap, first_sim, DOM, resolve_user_orders

db = load_json(os.path.join(DOM, "db.json"))
tasks = {str(t["id"]): t for t in load_json(os.path.join(DOM, "tasks.json"))}
orders, products = db["orders"], db["products"]

def gold_acts(t):
    return (t.get("evaluation_criteria", {}) or {}).get("actions", [])

def order_view(oid):
    o = orders.get(oid, {}); ad = o.get("address", {})
    its = [{"item_id": it.get("item_id"), "pid": it.get("product_id"),
            "name": it.get("name"), "opt": it.get("options")} for it in o.get("items", [])]
    return {"status": o.get("status"), "city": ad.get("city"), "state": ad.get("state"), "items": its}

def variants_of(pid):
    p = products.get(pid, {})
    return {vid: {"opt": v.get("options"), "avail": v.get("available"), "price": v.get("price")}
            for vid, v in (p.get("variants", {}) or {}).items()}

for tid in compute_gap():
    t = tasks[tid]; sim = first_sim("on_n32int8_floor_retail", tid)
    ins = t["user_scenario"]["instructions"]
    uid, _ = resolve_user_orders(db, t, sim)
    print("="*90)
    print(f"TASK {tid}  user={uid}")
    print("REASON:", (ins.get("reason_for_call") or "")[:600])
    ga = gold_acts(t)
    print("GOLD:")
    for a in ga:
        ar = a.get("arguments", {})
        key = {k: ar[k] for k in ("order_id","item_ids","new_item_ids","payment_method_id","city","state","address1") if k in ar}
        print(f"   {a.get('name')}  {json.dumps(key, ensure_ascii=False)}")
    # user orders
    print("ORDERS of user:")
    for oid in (db["users"].get(uid, {}).get("orders", []) if uid else []):
        v = order_view(oid)
        items_s = "; ".join(f"{it['name']}{it['opt']}" for it in v["items"])
        print(f"   {oid} [{v['status']}] {v['city']}/{v['state']} :: {items_s[:160]}")
    # variants for any gold new_item_ids (L3) — find the product
    for a in ga:
        nii = a.get("arguments", {}).get("new_item_ids") or []
        iis = a.get("arguments", {}).get("item_ids") or []
        for old_iid in iis:
            # find pid of old item
            pid = None
            for oid in (db["users"].get(uid, {}).get("orders", []) if uid else []):
                for it in orders.get(oid, {}).get("items", []):
                    if it.get("item_id") == old_iid: pid = it.get("product_id")
            if pid:
                vs = variants_of(pid)
                print(f"   VARIANTS of pid={pid} (old_item {old_iid}):")
                for vid, vv in vs.items():
                    mark = " <-GOLD_new" if vid in nii else ""
                    print(f"      {vid} {vv['opt']} avail={vv['avail']}{mark}")
