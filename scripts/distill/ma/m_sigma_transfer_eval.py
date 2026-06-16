#!/usr/bin/env python3
"""M-D TRANSFER eval (C8 first measurement): does M-sigma (cfb-derivation-trained) transfer
its typed-$ref derivation skill to HELD-OUT tau2 retail exchange? We reconstruct each tau2
exchange case as a cfb-style conversation (observations = get_order_details/get_product_details/
get_user_details outputs as tool messages), serve M-sigma, let it emit exchange_delivered_order_items
with $ref args, resolve $refs against the observations, and score PER-ARG-TYPE vs gold:
  - THREADING args (order_id, item_ids[old], payment_method_id): copyable from an observation
    field -> M-sigma's cfb threading skill SHOULD transfer.
  - SELECTION arg (new_item_ids): requires selecting a variant by criteria -> ORPHAN to cfb
    (threading != selection) -> expected to FAIL (quantifies the granularity gap).
Compares M-sigma vs base. Honest: confirms how far threading transfers and that selection needs data v2.
"""
import json, argparse, requests, sys
sys.path.insert(0, "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/ma")
from m_sigma_data import _walk  # noqa


def build_obs(case):
    """observations the model can $ref into (index order matters for $ref idx)."""
    e0 = case["exchanges"][0]
    order = {"order_id": case["order_id"], "status": "delivered",
             "items": [{"item_id": e["old_item_id"], "name": e["old_item_name"],
                        "product_id": e["product_id"], "options": e["old_options"]} for e in case["exchanges"]]}
    obs = [("get_order_details", order)]
    for e in case["exchanges"]:
        obs.append(("get_product_details",
                    {"product_id": e["product_id"],
                     "variants": {v["item_id"]: {"options": v["options"], "available": v["available"]}
                                  for v in e["variant_catalog"]}}))
    obs.append(("get_user_details", {"payment_methods": {case["gold_payment_method_id"]: {"source": "credit_card"}},
                                      "orders": [case["order_id"]]}))
    return obs


TOOLS = [{"type": "function", "function": {
    "name": "exchange_delivered_order_items",
    "description": "Exchange items in a delivered order for new variants of the same product.",
    "parameters": {"type": "object", "properties": {
        "order_id": {"type": "string"}, "item_ids": {"type": "array", "items": {"type": "string"}},
        "new_item_ids": {"type": "array", "items": {"type": "string"}}, "payment_method_id": {"type": "string"}},
        "required": ["order_id", "item_ids", "new_item_ids", "payment_method_id"]}}}]


def build_conv(case, obs):
    msgs = [{"role": "system", "content": "You are a tool-using assistant. Call the appropriate functions to fulfill the user request."},
            {"role": "user", "content": case["nl"]}]
    # present observations as prior assistant-call + tool-output pairs (so $ref idx = obs index)
    for i, (tname, out) in enumerate(obs):
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": tname, "arguments": "{}"}}]})
        msgs.append({"role": "tool", "content": json.dumps(out, ensure_ascii=False), "tool_call_id": f"c{i}"})
    msgs.append({"role": "user", "content": "Now call exchange_delivered_order_items to perform the requested exchange."})
    return msgs


def resolve(val, obs):
    """resolve {$ref:'idx#path'} against obs; lists/scalars passed through (recursively)."""
    if isinstance(val, dict) and "$ref" in val:
        try:
            idx_s, path = val["$ref"].split("#", 1); idx = int(idx_s)
            for p, v in _walk(obs[idx][1]):
                if p == path:
                    return v
        except Exception:
            return None
        return None
    if isinstance(val, list):
        return [resolve(x, obs) for x in val]
    return val


def chat(base, model, messages):
    r = requests.post(f"{base}/chat/completions",
                      json={"model": model, "messages": messages, "tools": TOOLS, "tool_choice": "auto",
                            "max_tokens": 512, "temperature": 0.0}, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def score(base, model, cases, tag):
    agg = {"order_id": [0, 0], "item_ids": [0, 0], "new_item_ids": [0, 0], "payment_method_id": [0, 0], "all": [0, 0]}
    used_ref = 0
    for case, obs in cases:
        gold = {"order_id": case["order_id"], "item_ids": [e["old_item_id"] for e in case["exchanges"]],
                "new_item_ids": [e["gold_new_item_id"] for e in case["exchanges"]],
                "payment_method_id": case["gold_payment_method_id"]}
        msg = chat(base, model, build_conv(case, obs))
        args = {}
        if msg.get("tool_calls"):
            try: args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
            except Exception: args = {}
        if any(isinstance(v, dict) and "$ref" in v or (isinstance(v, list) and any(isinstance(x, dict) and "$ref" in x for x in v)) for v in args.values()):
            used_ref += 1
        resolved = {k: resolve(v, obs) for k, v in args.items()}
        allok = True
        for k in ("order_id", "item_ids", "new_item_ids", "payment_method_id"):
            ok = resolved.get(k) == gold[k]
            agg[k][1] += 1; agg[k][0] += int(ok); allok = allok and ok
        agg["all"][1] += 1; agg["all"][0] += int(allok)
    print(f"[{tag}] used_$ref={used_ref}/{len(cases)} | " +
          " ".join(f"{k}={a[0]}/{a[1]}({a[0]/max(a[1],1):.2f})" for k, a in agg.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="/home/woori/scratch/ma_eval_cases.jsonl")
    ap.add_argument("--base", default="http://localhost:8015/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="model")
    args = ap.parse_args()
    cases = [(c, build_obs(c)) for c in (json.loads(l) for l in open(args.cases, encoding="utf-8"))]
    print(f"loaded {len(cases)} tau2 exchange cases")
    score(args.base, args.model, cases, args.tag)
