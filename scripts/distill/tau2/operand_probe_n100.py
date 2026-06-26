#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""operand_probe_n100.py — 전 retail task isolated operand-pick (n~100·gpt-4.1 0) + 철저한 원인분석.
각 gold operand 결정(변형 old→new·⋈ order)을 격리 단일턴으로 로컬 Qwen에 → 정답률 + 실패 분류:
  gold가 argmax/argmin(price)인가 · model이 argmax/argmin 골랐나 · gold 변형이 available인가 ·
  복합기준(price+옵션) · cross-item/budget 의존(probe under-spec) 후보.
→ 실패가 {compute-offloadable(argmax/argmin) / explicit-spec-formalize / 복합·예산(under-spec) / 모호} 중 무엇인지.
"""
import json, urllib.request, argparse
from collections import Counter
DOM = "/home/woori/scratch/tau2-bench/data/tau2/domains/retail"
db = json.load(open(DOM + "/db.json"))
TASKS = json.load(open(DOM + "/tasks.json"))
prods, orders = db["products"], db["orders"]
WMOD = {"modify_pending_order_items", "exchange_delivered_order_items"}


def ask(prompt, model, base):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "max_tokens": 60}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
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
    ap.add_argument("--max", type=int, default=100000)
    ap.add_argument("--quiet", action="store_true", help="실패만 출력(컨텍스트 절약)")
    a = ap.parse_args()
    rows = []
    orow = []  # ⋈ order-pick rows
    for t in TASKS:
        tid = str(t["id"])
        reason = str(t.get("user_scenario", {}).get("instructions", {}).get("reason_for_call", ""))[:700]
        rlow = reason.lower()
        compound = ("most expensive" in rlow and "size" in rlow) or ("cheapest" in rlow and ("budget" in rlow or "$" in reason or "credit" in rlow))
        budget_dep = any(w in rlow for w in ["budget", "bring the cost", "credit left", "total is"])
        for act in (t.get("evaluation_criteria", {}) or {}).get("actions", []):
            if act.get("name") not in WMOD:
                continue
            ar = act.get("arguments") or {}
            for old_id, gold_new in zip(ar.get("item_ids") or [], ar.get("new_item_ids") or []):
                p = product_of(old_id) or product_of(gold_new)
                if not p or gold_new not in (p.get("variants") or {}):
                    continue
                vs = p["variants"]
                prices = {vid: v.get("price", 0) for vid, v in vs.items()}
                avail = {vid: vs[vid].get("available") for vid in vs}
                maxp = max(prices, key=prices.get)
                minp = min(prices, key=prices.get)
                cur = vs.get(old_id, {})
                lines = [f"  item_id={vid}: options={v['options']} price={v['price']} available={v.get('available')}" for vid, v in vs.items()]
                prompt = (f"Customer's request:\n{reason}\n\nChoosing the replacement for ONE {p['name']} the customer "
                          f"owns with options {cur.get('options')} (price {cur.get('price')}).\nVariants:\n" + "\n".join(lines) +
                          f"\n\nWhich item_id should this {p['name']} be changed to? Output ONLY the item_id.")
                ans = ask(prompt, a.agent_model, a.agent_base)
                pick = next((vid for vid in vs if vid in ans), None)
                r = dict(tid=tid, kind="variant", gold=gold_new, pick=pick, ok=(gold_new in ans),
                         gold_argmax=(gold_new == maxp), gold_argmin=(gold_new == minp),
                         pick_argmax=(pick == maxp), pick_argmin=(pick == minp),
                         gold_avail=avail.get(gold_new), compound=compound, budget=budget_dep, nvar=len(vs))
                rows.append(r)
                if not a.quiet or not r["ok"]:
                    print(f"t{tid} variant {old_id}->{gold_new}: {'OK' if r['ok'] else 'X '+str(pick)}")
        # ── ⋈ order-pick (전 task·gold write의 order_id·user>1 order·decision당 1회) ──
        seen_dec = set()
        for act in (t.get("evaluation_criteria", {}) or {}).get("actions", []):
            goid = (act.get("arguments") or {}).get("order_id")
            if not goid or goid not in orders or goid in seen_dec:
                continue
            uid = orders[goid].get("user_id")
            uords = {oid: o for oid, o in orders.items() if o.get("user_id") == uid}
            if len(uords) <= 1:
                continue
            seen_dec.add(goid)
            lines = [f"  {oid}: status={o['status']} ship_to={o['address'].get('city')},{o['address'].get('state')} items={[i['name'] for i in o['items']][:4]}" for oid, o in uords.items()]
            prompt = (f"Customer's request:\n{reason}\n\nThe customer's orders:\n" + "\n".join(lines) +
                      f"\n\nWhich order_id is the one the customer is referring to for this change/return? Output ONLY the order_id.")
            ans = ask(prompt, a.agent_model, a.agent_base)
            ok = str(goid) in ans
            orow.append(dict(tid=tid, gold=goid, ok=ok, nord=len(uords)))
            if not a.quiet or not ok:
                print(f"t{tid} ⋈order ->{goid}: {'OK' if ok else 'X'}")
    # ── aggregate ──
    n = len(rows); ok = sum(r["ok"] for r in rows)
    print(f"=== isolated variant-pick n={n} · 정답 {ok}/{n} ({ok/n:.0%}) ===\n")
    F = [r for r in rows if not r["ok"]]
    print(f"실패 {len(F)}건 분류:")
    cat = Counter()
    for r in F:
        if r["compound"] or r["budget"]:
            c = "복합/예산기준(probe under-spec·cross-item)"
        elif r["gold_argmax"] or r["gold_argmin"]:
            c = "단순 argmax/argmin인데 틀림(compute-offloadable·model 규칙적용 실패)"
        else:
            c = "explicit-spec/기타(criterion-formalize)"
        cat[c] += 1
    for c, v in cat.most_common():
        print(f"  {v:3d}  {c}")
    # model이 argmax/argmin 규칙을 (잘못) 적용한 비율
    print(f"\nmodel-pick이 argmax-price: {sum(r['pick_argmax'] for r in rows)}/{n} · argmin-price: {sum(r['pick_argmin'] for r in rows)}/{n}")
    print(f"gold가 argmax: {sum(r['gold_argmax'] for r in rows)} · argmin: {sum(r['gold_argmin'] for r in rows)} · 둘다아님: {sum(not(r['gold_argmax'] or r['gold_argmin']) for r in rows)}")
    print(f"gold 변형 available=False(불가피 실패?): {sum(r['gold_avail'] is False for r in rows)}")
    # 정답률 by 단순(non-compound) vs 복합
    simple = [r for r in rows if not (r["compound"] or r["budget"])]
    print(f"\n단순기준(non-compound/budget) 정답률: {sum(r['ok'] for r in simple)}/{len(simple)} ({(sum(r['ok'] for r in simple)/max(len(simple),1)):.0%})")
    comp = [r for r in rows if r["compound"] or r["budget"]]
    print(f"복합/예산기준 정답률: {sum(r['ok'] for r in comp)}/{len(comp)} ({(sum(r['ok'] for r in comp)/max(len(comp),1)):.0%})")
    # ── ⋈ order-pick 집계 ──
    if orow:
        ook = sum(r["ok"] for r in orow)
        print(f"\n=== isolated ⋈ order-pick n={len(orow)} · 정답 {ook}/{len(orow)} ({ook/len(orow):.0%}) ===")
        multi = [r for r in orow if r["nord"] >= 3]
        print(f"  (orders>=3인 어려운 케이스: {sum(r['ok'] for r in multi)}/{len(multi)})")
    # ── 전체 operand 종합 ──
    alln = n + len(orow); allok = ok + (sum(r["ok"] for r in orow) if orow else 0)
    print(f"\n=== ★전체 operand-pick(변형+⋈) n={alln} · 정답 {allok}/{alln} ({allok/alln:.0%}) ===")


if __name__ == "__main__":
    main()
