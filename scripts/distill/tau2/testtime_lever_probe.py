#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""testtime_lever_probe.py — Phase A of TESTTIME_COMPUTE_LEVER_DESIGN (rev3·2026-07-07).

A0 (base, direct) vs A1 (prompted-CoT, FIXED max_tokens=900) on the *isolated-valid* buckets
(variant-criterion + cross-order ⋈). Per-bucket accuracy + Δ + completion-tokens + truncation + flips.
noise-free by construction (isolated single-turn·no user-sim in loop). gpt-4.1=0 (local Qwen only).

Scope (rev3 §4): variant-criterion=isolated-valid · ⋈=isolated-valid (UNDER-SPEC=lower bound).
coverage=load-only=OUT OF SCOPE (not probed here). calc=deterministic scaffold (not probed).
A1 is the ONLY pure same-weights test-time-compute arm. A2(QwQ native)=Phase B. o4-mini=frontier ceiling (elsewhere).
"""
import json, urllib.request, argparse, sys
from collections import Counter

DOM = "/home/woori/scratch/tau2-bench/data/tau2/domains/retail"
db = json.load(open(DOM + "/db.json"))
TASKS = json.load(open(DOM + "/tasks.json"))
prods, orders = db["products"], db["orders"]
WMOD = {"modify_pending_order_items", "exchange_delivered_order_items"}

COT_MAX = 900     # rev3 §3: FIXED reproducible CoT budget
DIRECT_MAX = 60
COT_SUFFIX = "\n\nReason step by step. Then on the LAST line write exactly:\nFinal answer: <id>"

# Phase B (QwQ native thinking): reasoning models degenerate at temp 0 -> set --temperature 0.6
# (QwQ documented setting) and raise --req_timeout for the large thinking budget.
TEMPERATURE = 0.0   # default preserves Phase A reproducibility (same-weights A0/A1)
REQ_TIMEOUT = 240


def ask(prompt, model, base, max_tokens, cot=False):
    """returns (content, completion_tokens, truncated). robust to errors -> (None,0,False)."""
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt + (COT_SUFFIX if cot else "")}],
                       "temperature": TEMPERATURE, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=REQ_TIMEOUT).read())
        ch = r["choices"][0]
        content = ch["message"].get("content") or ""
        ntok = (r.get("usage") or {}).get("completion_tokens", 0)
        trunc = (ch.get("finish_reason") == "length")
        return content, ntok, trunc
    except Exception as e:
        print(f"  [ask error] {e}", flush=True)
        return None, 0, False


def extract(ans, cands, cot):
    """pick chosen id. cot: prefer 'Final answer:' line, else LAST candidate mentioned. base: first match."""
    if ans is None:
        return None
    if not cot:
        return next((c for c in cands if c in ans), None)
    for line in reversed(ans.splitlines()):
        if "final answer" in line.lower():
            hit = [c for c in cands if c in line]
            if hit:
                return hit[0]
    # fallback: last candidate mentioned anywhere in the reasoning
    best = None
    for c in cands:
        idx = ans.rfind(c)
        if idx >= 0 and (best is None or idx > best[0]):
            best = (idx, c)
    return best[1] if best else None


def product_of(iid):
    for p in prods.values():
        if iid in (p.get("variants") or {}):
            return p
    return None


def run_decision(prompt, cands, gold, model, base, extra, cot_max=COT_MAX):
    a0, n0, _ = ask(prompt + " Output ONLY the id.", model, base, DIRECT_MAX, cot=False)
    a1, n1, tr1 = ask(prompt, model, base, cot_max, cot=True)
    p0, p1 = extract(a0, cands, False), extract(a1, cands, True)
    r = dict(gold=gold, ok0=(p0 == gold), ok1=(p1 == gold), n0=n0, n1=n1, tr1=tr1)
    r.update(extra)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent_base", default="http://localhost:8140/v1")
    ap.add_argument("--agent_model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--max", type=int, default=100000)
    ap.add_argument("--save_json", default=None, help="dump raw per-decision rows for [[08]] forensic")
    ap.add_argument("--cot_max", type=int, default=COT_MAX,
                    help="CoT/thinking token budget. 900=A1 prompted-CoT (Phase A); 8000=A2 QwQ native (Phase B)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0=Phase A reproducible (same-weights A0/A1); 0.6=QwQ native (Phase B, avoids temp-0 loops)")
    ap.add_argument("--req_timeout", type=int, default=240,
                    help="per-request wall timeout (s). raise for large --cot_max (QwQ 8000 tok can exceed 240s)")
    a = ap.parse_args()
    global TEMPERATURE, REQ_TIMEOUT
    TEMPERATURE, REQ_TIMEOUT = a.temperature, a.req_timeout
    vrows, orows = [], []
    for ti, t in enumerate(TASKS):
        if ti >= a.max:
            break
        tid = str(t["id"])
        reason = str(t.get("user_scenario", {}).get("instructions", {}).get("reason_for_call", ""))[:700]
        rlow = reason.lower()
        compound = ("most expensive" in rlow and "size" in rlow) or \
                   ("cheapest" in rlow and ("budget" in rlow or "$" in reason or "credit" in rlow))
        budget_dep = any(w in rlow for w in ["budget", "bring the cost", "credit left", "total is"])
        # --- variant-criterion ---
        for act in (t.get("evaluation_criteria", {}) or {}).get("actions", []):
            if act.get("name") not in WMOD:
                continue
            ar = act.get("arguments") or {}
            for old_id, gold_new in zip(ar.get("item_ids") or [], ar.get("new_item_ids") or []):
                p = product_of(old_id) or product_of(gold_new)
                if not p or gold_new not in (p.get("variants") or {}):
                    continue
                vs = p["variants"]
                cur = vs.get(old_id, {})
                lines = [f"  item_id={vid}: options={v['options']} price={v['price']} available={v.get('available')}"
                         for vid, v in vs.items()]
                prompt = (f"Customer's request:\n{reason}\n\nChoosing the replacement for ONE {p['name']} the customer "
                          f"owns with options {cur.get('options')} (price {cur.get('price')}).\nVariants:\n"
                          + "\n".join(lines) + f"\n\nWhich item_id should this {p['name']} be changed to?")
                vrows.append(run_decision(prompt, list(vs.keys()), gold_new, a.agent_model, a.agent_base,
                                          dict(tid=tid, compound=(compound or budget_dep)), cot_max=a.cot_max))
        # --- cross-order ⋈ ---
        seen = set()
        for act in (t.get("evaluation_criteria", {}) or {}).get("actions", []):
            goid = (act.get("arguments") or {}).get("order_id")
            if not goid or goid not in orders or goid in seen:
                continue
            uid = orders[goid].get("user_id")
            uords = {oid: o for oid, o in orders.items() if o.get("user_id") == uid}
            if len(uords) <= 1:
                continue
            seen.add(goid)
            lines = [f"  {oid}: status={o['status']} ship_to={o['address'].get('city')},{o['address'].get('state')} "
                     f"items={[i['name'] for i in o['items']][:4]}" for oid, o in uords.items()]
            prompt = (f"Customer's request:\n{reason}\n\nThe customer's orders:\n" + "\n".join(lines) +
                      f"\n\nWhich order_id is the one the customer is referring to for this change/return?")
            orows.append(run_decision(prompt, list(uords.keys()), goid, a.agent_model, a.agent_base,
                                       dict(tid=tid, nord=len(uords)), cot_max=a.cot_max))
        if ti % 10 == 0:
            print(f"  ...task {ti}/{len(TASKS)} (v={len(vrows)} o={len(orows)})", flush=True)

    def agg(rows, label):
        n = len(rows)
        if not n:
            print(f"\n=== {label} === n=0")
            return
        ok0 = sum(r["ok0"] for r in rows); ok1 = sum(r["ok1"] for r in rows)
        mt0 = sum(r["n0"] for r in rows) / n; mt1 = sum(r["n1"] for r in rows) / n
        trunc = sum(r["tr1"] for r in rows)
        gain = sum(1 for r in rows if r["ok1"] and not r["ok0"])
        loss = sum(1 for r in rows if r["ok0"] and not r["ok1"])
        print(f"\n=== {label} (n={n}) ===")
        print(f"  A0 base   : {ok0}/{n} = {ok0/n:.3f}   (mean {mt0:.0f} tok)")
        print(f"  A1 CoT    : {ok1}/{n} = {ok1/n:.3f}   (mean {mt1:.0f} tok · trunc {trunc}/{n})")
        print(f"  Δ(CoT-base): {(ok1-ok0)/n:+.3f}   ·  Δtok {mt1-mt0:+.0f}   ·  cost/point "
              f"{((mt1-mt0)/max((ok1-ok0),1e-9)):.0f} tok/pt" if ok1 > ok0 else
              f"  Δ(CoT-base): {(ok1-ok0)/n:+.3f}   ·  Δtok {mt1-mt0:+.0f}")
        print(f"  flips: base✗→CoT✓ {gain} · base✓→CoT✗ {loss}")

    print("\n" + "=" * 62)
    agg(vrows, "BUCKET variant-criterion  [isolated-valid]")
    agg([r for r in vrows if not r["compound"]], "  variant simple (non-compound/budget)")
    agg([r for r in vrows if r["compound"]], "  variant compound/budget")
    agg(orows, "BUCKET cross-order ⋈  [isolated-valid; UNDER-SPEC=lower bound]")
    print("\n[scope] coverage=load-only=OUT OF SCOPE (not probed). calc=deterministic scaffold (not probed).")
    print(f"[note] model={a.agent_model} · A0=direct(60tok) · A1(CoT)=cot_max {a.cot_max}tok.")
    print("[note] Phase A: model=Qwen2.5-32B, cot_max=900 (A1=pure test-time-compute·same weights).")
    print("[note] Phase B: model=QwQ-32B-AWQ, cot_max~8000 (A1-column = A2 native thinking·upper bound thinking+RL).")
    print("[guard] Δ>0=promise only (not deployment). ⋈ Δ=0=boundary-SUSPECT (under-spec); full-run confirms (§2/§8).")
    if a.save_json:
        json.dump({"variant": vrows, "cross_order": orows}, open(a.save_json, "w"), indent=1)
        print(f"[saved] {a.save_json}  (variant={len(vrows)} cross_order={len(orows)})")


if __name__ == "__main__":
    main()
