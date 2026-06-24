#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Arm-II Probe-B (RAW) — v1(escape_arm2_probe.py) 무효화 교정 (다른세션 리뷰 2026-06-25).
v1 결함: 판별필드(state·count)를 미리 추출·라벨해 표로 줌 → lookup이지 formalize 아님(15/15 자명).
Probe-B: 후보=raw 주문 dict(중첩 address·미카운트 items)·criterion 필드 라벨 금지 → 모델이
  ① criterion formalize ② raw서 추출(address.state·item count·⋈ cross-order) ③ 매칭. = 진짜 ⓑ 스킬.
+ 102/101 ⋈(NY주소=다른주문 출처) = 실제 실패 sub-part probe(watch-id 아님).
판정: raw서 맞힘→진짜 grounded(트리거 OK) / raw 틀림(특히 ⋈)→formalize-from-raw=진짜 잔여(learn중심).
"""
import json, os, re, argparse, urllib.request
from escape_scope_diag import load_json, DOM

db = load_json(os.path.join(DOM, "db.json"))
USERS, ORDERS = db["users"], db["orders"]

def raw_orders(uid):
    out = []
    for oid in USERS.get(uid, {}).get("orders", []):
        o = ORDERS.get(oid, {})
        out.append({"order_id": oid, "status": o.get("status"),
                    "address": o.get("address"),
                    "items": [{"name": it.get("name"), "options": it.get("options")} for it in o.get("items", [])]})
    return out

def ny_address1(uid):  # ⋈ gold = NY 주문의 address1
    for oid in USERS.get(uid, {}).get("orders", []):
        ad = ORDERS.get(oid, {}).get("address", {})
        if ad.get("state") == "NY":
            return ad.get("address1")
    return None

# Probe-B 스펙: raw 후보 + 유저말투 질문(criterion 미라벨). kind=id(order_id) | addr(address1 ⋈)
PROBE = {
    "71":  ("ivan_khan_7475", "id", "The customer says: 'I ordered an order that was sent to my son's address in Washington DC, and I want to modify that order.' Which order_id is the one the customer means?", "#W5270061"),
    "72":  ("ivan_khan_7475", "id", "The customer says: 'I sent an order to my son's address in Washington DC and want to modify it.' Which order_id is the one the customer means?", "#W5270061"),
    "74":  ("lei_li_6575", "id", "The customer says: 'I want to cancel my pending order — the one with five items.' Which order_id is the one the customer means?", "#W3189752"),
    "101": ("noah_ito_3850", "id", "The customer says: 'I placed an order with two watches; I want to change its address.' Which order_id is the order with two watches?", "#W4219264"),
    "102w":("noah_ito_3850", "id", "The customer says: 'I placed an order with two watches; I want to change its address.' Which order_id is the order with two watches?", "#W4219264"),
    # ⋈ cross-entity (실제 101/102 실패 sub-part): NY 주소를 *다른 주문*서 캐기
    "101x":("noah_ito_3850", "addr", "The customer says: 'Change my two-watch order to my New York address — I won't tell you, but it's the shipping address of one of my OTHER orders.' What is the customer's New York street address (address1)?", None),
    "102x":("noah_ito_3850", "addr", "The customer says: 'Change my order to my New York address — it's the shipping address of one of my OTHER orders, I won't reveal it.' What is the customer's New York street address (address1)?", None),
}

def call(port, model, text, temp):
    body = json.dumps({"model": model, "temperature": temp, "max_tokens": 60,
        "messages": [{"role": "user", "content": text}]}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8360)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()
    print(f"# Arm-II Probe-B (RAW·formalize-from-raw) — 32B trials={args.trials}\n")
    summ = {}
    for tid, (uid, kind, q, gold) in PROBE.items():
        if kind == "addr" and gold is None:
            gold = ny_address1(uid)
        cand = json.dumps(raw_orders(uid), ensure_ascii=False, indent=1)
        text = f"You are a retail agent. Here are the customer's orders (raw):\n{cand}\n\n{q}\n" + \
               ("Respond with ONLY the order_id (#W...)." if kind == "id" else "Respond with ONLY the street address (address1).")
        got = []
        for i in range(args.trials):
            temp = 0.0 if i == 0 else 0.7
            try: out = call(args.port, args.model, text, temp)
            except Exception as e: out = f"ERR:{e}"
            if kind == "id":
                m = re.search(r"#W\d+", out); pick = m.group(0) if m else f"?({out[:25]})"
                ok = (pick == gold)
            else:
                pick = out.strip().strip('"')[:40]
                ok = bool(gold) and gold.lower() in out.lower()
            got.append((pick, ok))
        c = sum(1 for _, ok in got if ok)
        verdict = "GROUNDED" if c == args.trials else ("PARTIAL" if c else "FAIL(formalize-from-raw)")
        summ[tid] = (c, args.trials, verdict)
        print(f"[{tid:4s}] gold={gold}  picks={[p for p,_ in got]}  → {c}/{args.trials} {verdict}")
    print("\n# 요약:", {k: f"{v[0]}/{v[1]} {v[2]}" for k, v in summ.items()})
    print("# id-probe(71·72·74·101·102w)=formalize+extract+match / addr-probe(101x·102x)=⋈ cross-entity(가장 어려운 formalize).")

if __name__ == "__main__":
    main()
