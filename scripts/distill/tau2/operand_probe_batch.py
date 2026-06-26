#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""operand_probe_batch.py — isolated operand-pick 안정성 (n≥10·gpt-4.1 0원).
각 task의 gold operand 결정(변형-pick: old→new item / ⋈-pick: 어느 order)을 *격리 단일턴*으로
로컬 Qwen에 제시(기준=reason_for_call·choice-set=DB) → gold 일치율.
make-or-break "operand 스킬 present?"의 안정 측정(full-flow 스크립팅 충실도 위험 회피).
"""
import json, urllib.request, argparse
DOM = "/home/woori/scratch/tau2-bench/data/tau2/domains/retail"
db = json.load(open(DOM + "/db.json"))
tasks = {str(t["id"]): t for t in json.load(open(DOM + "/tasks.json"))}
prods, orders, users = db["products"], db["orders"], db["users"]
WMOD = {"modify_pending_order_items", "exchange_delivered_order_items"}


def ask(prompt, model, base):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "max_tokens": 80}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["choices"][0]["message"]["content"]


def product_of(item_id):
    for p in prods.values():
        if item_id in (p.get("variants") or {}):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent_base", default="http://localhost:8360/v1")
    ap.add_argument("--agent_model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    a = ap.parse_args()
    TIDS = ["17", "20", "36", "37", "71", "92", "99", "105", "109", "111"]
    correct = total = 0
    for tid in TIDS:
        t = tasks[tid]
        reason = str(t.get("user_scenario", {}).get("instructions", {}).get("reason_for_call", ""))[:700]
        # ── variant-pick probes (gold modify/exchange의 old→new 각 쌍) ──
        for act in (t.get("evaluation_criteria", {}) or {}).get("actions", []):
            if act.get("name") not in WMOD:
                continue
            ar = act.get("arguments") or {}
            olds = ar.get("item_ids") or []
            news = ar.get("new_item_ids") or []
            for old_id, gold_new in zip(olds, news):
                p = product_of(old_id) or product_of(gold_new)
                if not p:
                    continue
                cur = (p.get("variants") or {}).get(old_id, {})
                vs = p["variants"]
                lines = [f"  item_id={vid}: options={v['options']} price={v['price']} available={v.get('available')}"
                         for vid, v in vs.items()]
                prompt = (f"Customer's overall request:\n{reason}\n\n"
                          f"Right now we are choosing the replacement for ONE item: a {p['name']} the customer "
                          f"currently owns with options {cur.get('options')} (price {cur.get('price')}).\n"
                          f"Available variants of this {p['name']}:\n" + "\n".join(lines) +
                          f"\n\nPer the customer's request, which item_id should this {p['name']} be changed to? "
                          f"Output ONLY the item_id.")
                ans = ask(prompt, a.agent_model, a.agent_base)
                ok = str(gold_new) in ans
                correct += ok
                total += 1
                print(f"t{tid} variant {old_id}->gold {gold_new}: {'✓' if ok else '✗ '+ans.strip()[:40]}")
        # ── ⋈ order-pick probes (gold address: 어느 order) ──
        for act in (t.get("evaluation_criteria", {}) or {}).get("actions", []):
            if act.get("name") != "modify_pending_order_address":
                continue
            gold_oid = (act.get("arguments") or {}).get("order_id")
            uid = orders.get(gold_oid, {}).get("user_id")
            if not uid:
                continue
            uords = {oid: o for oid, o in orders.items() if o.get("user_id") == uid}
            if len(uords) <= 1:
                continue
            lines = [f"  {oid}: status={o['status']} ship_to={o['address'].get('city')},{o['address'].get('state')} ({o['address'].get('address1')})"
                     for oid, o in uords.items()]
            prompt = (f"Customer's overall request:\n{reason}\n\n"
                      f"The customer's orders:\n" + "\n".join(lines) +
                      f"\n\nWhich order_id is the one whose shipping address the customer wants changed? Output ONLY the order_id.")
            ans = ask(prompt, a.agent_model, a.agent_base)
            ok = str(gold_oid) in ans
            correct += ok
            total += 1
            print(f"t{tid} ⋈order ->gold {gold_oid}: {'✓' if ok else '✗ '+ans.strip()[:40]}")
    print(f"\n=== isolated operand-pick 안정성: {correct}/{total} correct (gpt-4.1 0·single-turn) ===")


if __name__ == "__main__":
    main()
