#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Arm-II select-probe (ESCAPE_SCOPE..LAYERS_AUG §4.7) — capability adjudicator.
5개 ⓑ mis-ground 케이스(71·72·74·101·102)에 후보집합(유저 주문들)을 *떠먹이고*
disambiguation 질문 → 32B가 정답 order를 고르나?
  맞힘 → grounding 됨·free-formalize/retrieval만 실패 = 학습여지(GO)
  틀림 → 후보 줘도 못 함 = capability-bound(NO-GO escalate)
로컬 32B만(gpt-4.1 불요·throttle 없음). 공정성: 궤적서 실제 retrieve 가능했던 정보만 제시.
"""
import json, os, re, argparse, urllib.request
from escape_scope_diag import load_json, DOM

PROBE = {
    "71": {"req": "I made a mistake and ordered an order that was sent to my son's address in Washington DC. I want to modify that order's shipping address.",
           "q": "Which order_id is the one shipped to Washington, DC?", "gold": "#W5270061", "user": "ivan_khan_7475"},
    "72": {"req": "I made a mistake and sent an order to my son's address in Washington DC. I want to modify that order.",
           "q": "Which order_id is the one shipped to Washington, DC?", "gold": "#W5270061", "user": "ivan_khan_7475"},
    "74": {"req": "I have a pending order with five items that I no longer need and want to cancel.",
           "q": "Which order_id is the pending order that contains exactly five items?", "gold": "#W3189752", "user": "lei_li_6575"},
    "101": {"req": "I just placed an order with two watches and want to change its shipping address.",
            "q": "Which order_id is the order that contains two watches (wristwatches)?", "gold": "#W4219264", "user": "noah_ito_3850"},
    "102": {"req": "I just placed an order with two watches and want to change its shipping address.",
            "q": "Which order_id is the order that contains two watches (wristwatches)?", "gold": "#W4219264", "user": "noah_ito_3850"},
}

def candidates(db, uid):
    users, orders = db["users"], db["orders"]
    rows = []
    for oid in users.get(uid, {}).get("orders", []):
        o = orders.get(oid, {}); ad = o.get("address", {})
        items = [it.get("name") for it in o.get("items", [])]
        rows.append((oid, o.get("status"), ad.get("city"), ad.get("state"), items))
    return rows

def prompt(req, rows, q):
    lines = ["Customer says: \"%s\"" % req, "", "The customer's orders:"]
    for oid, st, city, state, items in rows:
        lines.append(f"- {oid} [status: {st}] shipped to {city}, {state} | {len(items)} item(s): {', '.join(items)}")
    lines += ["", q, "Respond with ONLY the order_id (format #W followed by digits). No explanation."]
    return "\n".join(lines)

def call(port, model, text, temp):
    body = json.dumps({"model": model, "temperature": temp, "max_tokens": 40,
        "messages": [{"role": "user", "content": text}]}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=120).read())
    return r["choices"][0]["message"]["content"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8360)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()
    db = load_json(os.path.join(DOM, "db.json"))
    print(f"# Arm-II select-probe — 32B={args.model} trials={args.trials}\n")
    summ = {}
    for tid, p in PROBE.items():
        rows = candidates(db, p["user"]); text = prompt(p["req"], rows, p["q"])
        got = []
        for i in range(args.trials):
            temp = 0.0 if i == 0 else 0.7
            try:
                out = call(args.port, args.model, text, temp)
            except Exception as e:
                out = f"ERR:{e}"
            m = re.search(r"#W\d+", out)
            pick = m.group(0) if m else f"?({out[:30]})"
            got.append(pick)
        correct = sum(1 for g in got if g == p["gold"])
        verdict = "GROUNDED(학습여지)" if correct == args.trials else ("PARTIAL" if correct else "CAP-BOUND(escalate)")
        summ[tid] = (correct, args.trials, verdict)
        print(f"[task {tid}] gold={p['gold']}  picks={got}  → {correct}/{args.trials} {verdict}")
    print("\n# 요약:", {k: f"{v[0]}/{v[1]} {v[2]}" for k, v in summ.items()})
    allc = sum(v[0] for v in summ.values()); alln = sum(v[1] for v in summ.values())
    print(f"# 전체 {allc}/{alln} 정답. GROUNDED 다수=학습여지(formalize-from-scratch만 실패) / CAP-BOUND 다수=번역 capability 한계(escalate).")

if __name__ == "__main__":
    main()
