#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""operand_controlled.py — user-sim 입력을 통제변인으로 분리해 순수 에이전트 operand 능력 측정 (gpt-4.1 0).
변형-pick: GIVEN-SPEC(gold 옵션 명시) vs GOAL(reason만) → 갭=criterion-해석 부하 vs 실행능력.
⋈ order-pick: ID-GIVEN(유저가 id 줌·probe 불공정→제외) vs DESCRIBED(묘사·genuine ⋈만 fair).
"""
import json, urllib.request, argparse, re
from collections import Counter
DOM = "/home/woori/scratch/tau2-bench/data/tau2/domains/retail"
db = json.load(open(DOM + "/db.json")); TASKS = json.load(open(DOM + "/tasks.json"))
prods, orders = db["products"], db["orders"]
WMOD = {"modify_pending_order_items", "exchange_delivered_order_items"}
DESC_KEYS = ["look it up", "look up", "don't want to mention", "do not want to mention", "don't want to reveal",
             "do not want to reveal", "don't reveal", "son's", "old address", "default address", "new address",
             "new home", "should be able to look", "in your orders", "in orders profile", "wrong order"]


def ask(prompt, model, base):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 50}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["choices"][0]["message"]["content"]


def product_of(iid):
    for p in prods.values():
        if iid in (p.get("variants") or {}):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent_base", default="http://localhost:8360/v1")
    ap.add_argument("--agent_model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    a = ap.parse_args()
    M, B = a.agent_model, a.agent_base
    gs_ok = gs_n = goal_ok = goal_n = 0          # 변형: GIVEN-SPEC / GOAL
    xj_desc_ok = xj_desc_n = xj_idgiven = 0       # ⋈: DESCRIBED(fair) / ID-GIVEN(제외카운트)
    for t in TASKS:
        tid = str(t["id"])
        reason = str(t.get("user_scenario", {}).get("instructions", {}).get("reason_for_call", ""))
        rlow = reason.lower()
        described = any(k in rlow for k in DESC_KEYS)
        for act in (t.get("evaluation_criteria", {}) or {}).get("actions", []):
            ar = act.get("arguments") or {}
            if act.get("name") in WMOD:
                for old_id, gold_new in zip(ar.get("item_ids") or [], ar.get("new_item_ids") or []):
                    p = product_of(old_id) or product_of(gold_new)
                    if not p or gold_new not in (p.get("variants") or {}):
                        continue
                    vs = p["variants"]
                    lines = "\n".join(f"  item_id={vid}: options={v['options']} price={v['price']} available={v.get('available')}" for vid, v in vs.items())
                    gold_opts = vs[gold_new]["options"]
                    # C1 GIVEN-SPEC: gold 옵션 명시
                    p1 = (f"Customer wants a {p['name']} with EXACTLY these options: {gold_opts}.\nVariants:\n{lines}\n\nWhich item_id matches? Output ONLY the item_id.")
                    a1 = ask(p1, M, B); gs_ok += (gold_new in a1); gs_n += 1
                    # C2 GOAL: reason만
                    cur = vs.get(old_id, {})
                    p2 = (f"Customer's request:\n{reason[:700]}\n\nChoosing replacement for a {p['name']} the customer owns with options {cur.get('options')}.\nVariants:\n{lines}\n\nWhich item_id? Output ONLY the item_id.")
                    a2 = ask(p2, M, B); goal_ok += (gold_new in a2); goal_n += 1
            if act.get("name") == "modify_pending_order_address":
                goid = ar.get("order_id")
                if not goid or goid not in orders:
                    continue
                uid = orders[goid].get("user_id"); uords = {o: orders[o] for o in orders if orders[o].get("user_id") == uid}
                if len(uords) <= 1:
                    continue
                if not described:
                    xj_idgiven += 1   # 유저가 id 줄 task → inference-probe 불공정 → 제외
                    continue
                lines = "\n".join(f"  {oid}: status={o['status']} ship_to={o['address'].get('city')},{o['address'].get('state')} ({o['address'].get('address1')})" for oid, o in uords.items())
                pj = (f"Customer's request (they will NOT give the order number, you must infer it):\n{reason[:700]}\n\nOrders:\n{lines}\n\nWhich order_id? Output ONLY the order_id.")
                aj = ask(pj, M, B); xj_desc_ok += (goid in aj); xj_desc_n += 1
    print("=== 통제 실험: 순수 operand 능력 (gpt-4.1 0) ===\n")
    print(f"변형-pick C1 GIVEN-SPEC(gold옵션 명시→매칭): {gs_ok}/{gs_n} ({gs_ok/max(gs_n,1):.0%})  ← 순수 실행능력")
    print(f"변형-pick C2 GOAL(reason 목표만):          {goal_ok}/{goal_n} ({goal_ok/max(goal_n,1):.0%})  ← +criterion해석")
    print(f"  → 갭(criterion-해석 부하) = {gs_ok/max(gs_n,1)-goal_ok/max(goal_n,1):+.0%}")
    print(f"\n⋈ DESCRIBED(genuine·유저 id 안줌·fair): {xj_desc_ok}/{xj_desc_n} ({xj_desc_ok/max(xj_desc_n,1):.0%})")
    print(f"⋈ ID-GIVEN(유저가 id 줌→probe 불공정·제외): {xj_idgiven}건 (이전 49% 오염원)")


if __name__ == "__main__":
    main()
