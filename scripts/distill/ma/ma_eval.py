#!/usr/bin/env python3
"""M-A 3-arm offline value-accuracy eval (M_A_PROTOTYPE_DESIGN.md §6/§7).

Per exchange case (from ma_gold_extract.py), query a served base model under 3 arms:
  A  concrete-emit  : model picks new_item_ids directly from the catalog (current paradigm)
  B  formal+resolver: model emits an abstract selector (option-value criteria, NO item_id),
                      guided_json (xgrammar) enforces TYPE, ma_resolver maps -> item_id
  C  formal, no grammar: same prompt as B but WITHOUT guided_json (separates grammar
                      enforcement from the learned/emergent content — review guard)

Score: per-item new_item_id correctness vs gold + case-level all-correct. For arm B/C,
decompose failures (§7⑥): wrong_criteria (resolved != gold but mapped), resolver_fail
(None / unresolved), tie. Also parse/type-violation rate. Pre-registered diagnostic: if B
errors are dominated by wrong_criteria, the write-wall root cause is reasoning (sigma /
NL->formalize), not fabrication -> first-class result.
"""
import json, argparse, sys, re
import requests
sys.path.insert(0, "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/ma")
from ma_resolver import select_variant, _option_value_space  # noqa: E402

SELECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "selectors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "select_by": {"type": "object", "additionalProperties": {"type": "string"}},
                    "fallback": {"type": "array",
                                 "items": {"type": "object", "additionalProperties": {"type": "string"}}},
                },
                "required": ["select_by"],
            },
        }
    },
    "required": ["selectors"],
}
CONCRETE_SCHEMA = {
    "type": "object",
    "properties": {"new_item_ids": {"type": "array", "items": {"type": "string"}}},
    "required": ["new_item_ids"],
}


def chat(base, model, messages, guided_json=None, max_tokens=512, temperature=0.0):
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": temperature}
    if guided_json is not None:
        payload["guided_json"] = guided_json  # vLLM extension (xgrammar)
    r = requests.post(f"{base}/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def extract_json(text, key):
    """Return the LAST balanced {...} object in text that parses and contains `key`.
    Robust to CoT prose preceding the JSON (and to a 'FINAL' marker)."""
    if not text:
        return None
    # prefer content after a FINAL marker if present
    mk = re.split(r"FINAL[^\n:{]*[:\n]", text, flags=re.IGNORECASE)
    scan = mk[-1] if len(mk) > 1 else text
    cands = []
    for region in (scan, text):
        starts = [i for i, c in enumerate(region) if c == "{"]
        for s in starts:
            depth = 0
            for j in range(s, len(region)):
                if region[j] == "{":
                    depth += 1
                elif region[j] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            o = json.loads(region[s:j + 1])
                            if isinstance(o, dict) and key in o:
                                cands.append(o)
                        except Exception:
                            pass
                        break
        if cands:
            return cands[-1]
    return None


_COT = ("\nFirst REASON step by step (a) which option(s) the user wants CHANGED and which "
        "stay the SAME as the current item, (b) map any synonym/brand the user says to the "
        "exact catalog option value, (c) handle any fallback/preference. THEN on the final "
        "line write 'FINAL: ' followed by the JSON.")


def prompt_concrete(case, cot=False):
    lines = ["You are a retail agent processing an exchange. The user wants to exchange items "
             "for new variants of the SAME product. For each item, pick the new variant's item_id "
             "from the catalog that best satisfies the request (respect any fallback preference). "
             "Only available variants can be chosen.",
             f"\nUser request: {case['nl']}\n"]
    for i, e in enumerate(case["exchanges"]):
        lines.append(f"Item {i+1}: {e['old_item_name']} (current options: {json.dumps(e['old_options'])})")
        lines.append("Available variants:")
        for v in e["variant_catalog"]:
            lines.append(f"  item_id={v['item_id']} options={json.dumps(v['options'])} available={v['available']}")
    tail = '{"new_item_ids": ["<id for item 1>", ...]} in item order.'
    lines.append(("\nReason, then output " + tail + _COT) if cot else ("\nOutput JSON only: " + tail))
    return [{"role": "user", "content": "\n".join(lines)}]


def prompt_formal(case, cot=False):
    lines = ["You are a retail agent processing an exchange. For each item, describe the DESIRED "
             "new variant by its OPTION VALUES — NOT an item_id. The new item is the SAME product "
             "as the current one, changed ONLY as the user requests; options the user does not "
             "mention stay the SAME as the current item. If the user gives a fallback preference "
             "(e.g. 'if X is unavailable, then Y'), list fallback override(s) in order. Use the "
             "exact option values from the product's vocabulary.",
             f"\nUser request: {case['nl']}\n"]
    for i, e in enumerate(case["exchanges"]):
        prod = {"variants": {v["item_id"]: {"options": v["options"], "available": v["available"]}
                             for v in e["variant_catalog"]}}
        vspace = {k: sorted(vals) for k, vals in _option_value_space(prod).items()}
        lines.append(f"Item {i+1}: {e['old_item_name']} (current options: {json.dumps(e['old_options'])})")
        lines.append(f"  valid option values: {json.dumps(vspace)}")
    tail = ('{"selectors": [{"select_by": {"<opt>": "<value>", ...}, '
            '"fallback": [{"<opt>": "<value>"}]}, ...]} in item order.')
    lines.append(("\nReason, then output " + tail + _COT) if cot else ("\nOutput JSON only: " + tail))
    return [{"role": "user", "content": "\n".join(lines)}]


def resolve_one(case_exchange, selector):
    """Map one formal selector to an item_id via the deterministic resolver."""
    prod = {"name": case_exchange["product_name"],
            "variants": {v["item_id"]: {"options": v["options"], "available": v["available"]}
                         for v in case_exchange["variant_catalog"]}}
    return select_variant(prod, case_exchange["old_options"], selector)


# arm -> (mode, cot, grammar)
ARM_SPEC = {
    "A": ("concrete", False, True), "Acot": ("concrete", True, False),
    "B": ("formal", False, True), "Bcot": ("formal", True, False),
    "C": ("formal", False, False),
}


def eval_case(case, base, model, arm):
    n = case["n_items"]
    golds = [e["gold_new_item_id"] for e in case["exchanges"]]
    out = {"task_id": case["task_id"], "arm": arm, "n": n, "gold": golds}
    mode, cot, grammar = ARM_SPEC[arm]
    if mode == "concrete":
        gj = CONCRETE_SCHEMA if grammar else None
        txt = chat(base, model, prompt_concrete(case, cot), guided_json=gj, max_tokens=1024 if cot else 512)
        obj = extract_json(txt, "new_item_ids")
        if not obj:
            out.update(parse_fail=True, pred=None, item_correct=[False]*n); return out
        pred = (list(obj["new_item_ids"]) + [None]*n)[:n]
        out.update(parse_fail=False, pred=pred,
                   item_correct=[pred[i] == golds[i] for i in range(n)])
        return out
    # formal (B / C / Bcot)
    gj = SELECTOR_SCHEMA if grammar else None
    txt = chat(base, model, prompt_formal(case, cot), guided_json=gj, max_tokens=1024 if cot else 512)
    obj = extract_json(txt, "selectors")
    if not obj:
        out.update(parse_fail=True, pred=None, item_correct=[False]*n, fail_kind=["parse"]*n); return out
    sels = (obj["selectors"] + [{}]*n)[:n]
    pred, kinds = [], []
    for i, e in enumerate(case["exchanges"]):
        new_id, diag = resolve_one(e, sels[i])
        pred.append(new_id)
        if new_id == golds[i]:
            kinds.append("ok")
        elif new_id is None:
            kinds.append("resolver_fail:" + diag.get("kind", "?"))
        else:
            kinds.append("wrong_criteria")  # resolved to a real variant, but not gold
    out.update(parse_fail=False, pred=pred, selectors=sels,
               item_correct=[pred[i] == golds[i] for i in range(n)], fail_kind=kinds)
    return out


def summarize(results):
    by_arm = {}
    for r in results:
        a = by_arm.setdefault(r["arm"], {"cases": 0, "case_correct": 0, "items": 0,
                                         "item_correct": 0, "parse_fail": 0, "kinds": {}})
        a["cases"] += 1
        a["items"] += r["n"]
        a["item_correct"] += sum(r["item_correct"])
        a["case_correct"] += int(all(r["item_correct"]))
        a["parse_fail"] += int(r.get("parse_fail", False))
        for k in r.get("fail_kind", []):
            if k != "ok":
                a["kinds"][k.split(":")[0]] = a["kinds"].get(k.split(":")[0], 0) + 1
    return by_arm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="/home/woori/scratch/ma_eval_cases.jsonl")
    ap.add_argument("--base", default="http://localhost:8013/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--arms", default="A,Acot,B,Bcot,C")
    ap.add_argument("--out", default="/home/woori/scratch/ma_eval_results.jsonl")
    args = ap.parse_args()
    cases = [json.loads(l) for l in open(args.cases, encoding="utf-8")]
    results = []
    with open(args.out, "w", encoding="utf-8") as f:
        for arm in args.arms.split(","):
            for c in cases:
                try:
                    r = eval_case(c, args.base, args.model, arm)
                except Exception as e:
                    r = {"task_id": c["task_id"], "arm": arm, "n": c["n_items"],
                         "error": str(e), "item_correct": [False]*c["n_items"]}
                results.append(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"arm {arm} done ({len(cases)} cases)")
    print("\n=== SUMMARY ===")
    for arm, a in sorted(summarize(results).items()):
        ic = a["item_correct"] / max(a["items"], 1)
        cc = a["case_correct"] / max(a["cases"], 1)
        print(f"arm {arm}: item-acc={ic:.3f} ({a['item_correct']}/{a['items']}) "
              f"case-acc={cc:.3f} ({a['case_correct']}/{a['cases']}) "
              f"parse_fail={a['parse_fail']} fail_kinds={a['kinds']}")
