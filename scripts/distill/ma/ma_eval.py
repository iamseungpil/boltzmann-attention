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
import json, argparse, sys, re, collections
import requests
sys.path.insert(0, "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/ma")
from ma_resolver import select_variant, _option_value_space, _norm  # noqa: E402

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


# cost instrumentation: chat() appends each call's token usage here; the driver clears
# it before each case and attaches the totals to the result (accuracy-PER-COST analysis).
_USAGE = []


def chat(base, model, messages, guided_json=None, max_tokens=512, temperature=0.0):
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": temperature}
    if guided_json is not None:
        payload["guided_json"] = guided_json  # vLLM extension (xgrammar)
    r = requests.post(f"{base}/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    j = r.json()
    _USAGE.append(j.get("usage", {}) or {})
    return j["choices"][0]["message"]["content"]


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


def prompt_concrete_level(case, level, cot=False):
    """Information-FLOOR ladder (MSC §6, 3-way). Output mode held CONSTANT (concrete
    item_id pick) so deltas isolate INPUT effects. Levels:
      L0  catalog WITHOUT availability  -> info-limited for fallback cases (floor below)
      L1  catalog + availability (== arm A baseline)
      L2a available-filtered, raw list  -> decision-space shrink (filter)
      L2b available, formalized TABLE + current-item row -> formatting (on top of filter)
      L3  available, each annotated with its DIFF from the current item -> pre-computed
          comparison (extreme deterministic offload; diff is deterministic, no NL needed)
    (L1->L2a)=filter, (L2a->L2b)=format, (L2b->L3)=pre-resolution, (L0->L1)=info floor."""
    lines = ["You are a retail agent processing an exchange. For each item, pick the new "
             "variant's item_id that best satisfies the request (respect any fallback "
             "preference). Only available variants can be chosen.",
             f"\nUser request: {case['nl']}\n"]
    for i, e in enumerate(case["exchanges"]):
        old = e["old_options"]; cat = e["variant_catalog"]; avail = [v for v in cat if v["available"]]
        lines.append(f"Item {i+1}: {e['old_item_name']} (current options: {json.dumps(old)})")
        if level == "L0":
            lines.append("Variants:")
            for v in cat:
                lines.append(f"  item_id={v['item_id']} options={json.dumps(v['options'])}")
        elif level == "L1":
            lines.append("Available variants:")
            for v in cat:
                lines.append(f"  item_id={v['item_id']} options={json.dumps(v['options'])} available={v['available']}")
        elif level == "L2a":
            lines.append("Available variants:")
            for v in avail:
                lines.append(f"  item_id={v['item_id']} options={json.dumps(v['options'])}")
        elif level == "L2b":
            keys = sorted({k for v in avail for k in v["options"]})
            lines.append("Available variants (table):")
            lines.append("  | item_id | " + " | ".join(keys) + " |")
            for v in avail:
                lines.append("  | " + v["item_id"] + " | " + " | ".join(str(v["options"].get(k, "-")) for k in keys) + " |")
            lines.append("  current item: | " + " | ".join(str(old.get(k, "-")) for k in keys) + " |")
        elif level == "L3":
            lines.append("Available variants (each annotated with how it differs from your current item):")
            for v in avail:
                diff = {k: f"{old.get(k)}->{v['options'][k]}" for k in v["options"] if old.get(k) != v["options"][k]}
                lines.append(f"  item_id={v['item_id']} differs_from_current={json.dumps(diff) if diff else 'identical'}")
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


def prompt_formal_fair(case, cot=False):
    """FAIR formal arm: shows the SAME catalog (options + availability) as the concrete
    arm, but asks for an abstract selector (criteria) instead of an item_id. Removes the
    info asymmetry of prompt_formal (which hid availability) — isolates 'emit criteria vs
    emit id GIVEN IDENTICAL INFO'."""
    lines = ["You are a retail agent processing an exchange. For each item, describe the DESIRED "
             "new variant by its OPTION VALUES — NOT an item_id. The new item is the SAME product "
             "changed ONLY as the user requests; unmentioned options stay the SAME as the current "
             "item. Choose values that correspond to an AVAILABLE variant (respect any fallback "
             "preference). Use the exact option values shown.",
             f"\nUser request: {case['nl']}\n"]
    for i, e in enumerate(case["exchanges"]):
        lines.append(f"Item {i+1}: {e['old_item_name']} (current options: {json.dumps(e['old_options'])})")
        lines.append("Variants (choose options matching an available one):")
        for v in e["variant_catalog"]:
            lines.append(f"  options={json.dumps(v['options'])} available={v['available']}")
    tail = ('{"selectors": [{"select_by": {"<opt>": "<value>", ...}, '
            '"fallback": [{"<opt>": "<value>"}]}, ...]} in item order.')
    lines.append(("\nReason, then output " + tail + _COT) if cot else ("\nOutput JSON only: " + tail))
    return [{"role": "user", "content": "\n".join(lines)}]


_CHANGES_SCHEMA = {"type": "object", "properties": {"changes": {"type": "object", "additionalProperties": {"type": "string"}}}, "required": ["changes"]}
_RELAX_SCHEMA = {"type": "object", "properties": {"relax": {"type": "object", "additionalProperties": {"type": "string"}}}, "required": ["relax"]}


def _snap(vocab, k, v):
    """If (k,v) or value v maps to a real (key,canonical-value), return (key,canon) else None."""
    if k in vocab:
        for x in vocab[k]:
            if _norm(v) == _norm(x):
                return k, x
        return None  # key valid, value invalid
    owners = [(ok, x) for ok in vocab for x in vocab[ok] if _norm(v) == _norm(x)]
    return owners[0] if len(owners) == 1 else None


def step_resolve(nl, ex, base, model, verify=True):
    """STRONG FORM: deterministic scaffold drives typed incremental steps with deterministic
    per-step verification (THESIS_STATEMENT §2A). Step1: LLM emits {key:value} CHANGES ->
    scaffold verifies against vocab, feeds back errors (caps compounding/mis-decomposition,
    catches synonyms like 'Google Home'). Step2: scaffold deterministically builds target =
    old (+) changes (keep-rest free) and checks availability. Step3 (if unavailable): LLM emits
    ONE relaxation -> verified -> retried. Returns (item_id|None, trace).
    verify=False (ABLATION/Snover): same decomposition but NO per-step verify/feedback retry
    (single attempt per step) — isolates whether external verification (not mere decomposition)
    is the load-bearing ingredient."""
    n_try = 3 if verify else 1
    n_fb = 2 if verify else 1
    prod = {"variants": {v["item_id"]: {"options": v["options"], "available": v["available"]} for v in ex["variant_catalog"]}}
    vocab = _option_value_space(prod)
    old = ex["old_options"]; keys = sorted(vocab)
    vdisp = {k: sorted(vocab[k]) for k in keys}
    trace = []

    # STEP 1 — changes, with verify+feedback loop
    base_msg = (f"User request: {nl}\nFocus item: {ex['old_item_name']} (current options: {json.dumps(old)})\n"
                f"Valid option values: {json.dumps(vdisp)}\n"
                "Which option(s) does the user want CHANGED for THIS item, and to what value? Use EXACT keys/values "
                'above; unmentioned options stay the same.\nOutput JSON only: {"changes": {"<key>": "<value>"}}')
    msgs = [{"role": "user", "content": base_msg}]; changes = {}
    for _ in range(n_try):
        txt = chat(base, model, msgs, guided_json=_CHANGES_SCHEMA)
        cand = (extract_json(txt, "changes") or {}).get("changes", {}) or {}
        errs, norm = [], {}
        for k, v in cand.items():
            snapped = _snap(vocab, k, v)
            if snapped:
                norm[snapped[0]] = snapped[1]
            elif k in vocab:
                errs.append(f"value '{v}' invalid for '{k}' (valid: {sorted(vocab[k])})")
            else:
                errs.append(f"'{v}'/'{k}' not a valid option (keys: {keys})")
        if not errs or not verify:  # verify=False: best-effort (use snapped subset, no feedback)
            changes = norm; break
        msgs = [{"role": "user", "content": base_msg}, {"role": "assistant", "content": txt},
                {"role": "user", "content": "Invalid: " + "; ".join(errs) + ". Re-output corrected JSON only."}]
    trace.append(("changes", changes))

    # STEP 2 — deterministic construct + availability
    target = {**old, **changes}
    hit = [v["item_id"] for v in ex["variant_catalog"] if v["available"] and v["options"] == target]
    if len(hit) == 1:
        return hit[0], trace + [("ok", target)]

    # STEP 3 — fallback relaxation (<=2), verified
    avail = [v["options"] for v in ex["variant_catalog"] if v["available"]]
    fb = (f"User request: {nl}\nFor item {ex['old_item_name']}, desired options {json.dumps(target)} are NOT available.\n"
          f"Available variants: {json.dumps(avail)}\nPer the user's fallback preference, change EXACTLY the option(s) "
          'needed to reach an available variant.\nOutput JSON only: {"relax": {"<key>": "<value>"}}')
    msgs = [{"role": "user", "content": fb}]
    for _ in range(n_fb):
        txt = chat(base, model, msgs, guided_json=_RELAX_SCHEMA)
        rel = (extract_json(txt, "relax") or {}).get("relax", {}) or {}
        norm = {}
        for k, v in rel.items():
            snapped = _snap(vocab, k, v)
            if snapped:
                norm[snapped[0]] = snapped[1]
        target2 = {**target, **norm}
        hit = [v["item_id"] for v in ex["variant_catalog"] if v["available"] and v["options"] == target2]
        if len(hit) == 1:
            return hit[0], trace + [("relax", norm), ("ok", target2)]
        msgs = [{"role": "user", "content": fb}, {"role": "assistant", "content": txt},
                {"role": "user", "content": "That did not reach an available variant. Try a different single relaxation. JSON only."}]
    return None, trace + [("fail", target)]


def resolve_one(case_exchange, selector):
    """Map one formal selector to an item_id via the deterministic resolver."""
    prod = {"name": case_exchange["product_name"],
            "variants": {v["item_id"]: {"options": v["options"], "available": v["available"]}
                         for v in case_exchange["variant_catalog"]}}
    return select_variant(prod, case_exchange["old_options"], selector)


# arm -> (mode, cot, grammar, twocall)
#  twocall: free-reasoning call (no grammar) THEN a strict guided_json transcription call.
#  => preserves reasoning AND guarantees schema validity, with no per-token gating
#     (the 'constrain only the final segment' idea; portable, works on any backend).
ARM_SPEC = {
    "A": ("concrete", False, True, False), "Acot": ("concrete", True, False, False),
    "Atwo": ("concrete", True, True, True),
    "B": ("formal", False, True, False), "Bcot": ("formal", True, False, False),
    "Btwo": ("formal", True, True, True),
    "C": ("formal", False, False, False),
    # FAIR formal arms: same catalog+availability info as concrete (no asymmetry).
    "Bfair": ("formalfair", False, True, False), "Bfaircot": ("formalfair", True, False, False),
    "Bfairtwo": ("formalfair", True, True, True),
}
PROMPT_FN = {"formal": prompt_formal, "formalfair": prompt_formal_fair}


def _twocall(base, model, prompt_msgs, schema, key):
    """Call 1: free CoT reasoning (unconstrained). Call 2: strict guided_json transcription
    of the decided answer, with the reasoning in context."""
    r1 = chat(base, model, prompt_msgs, guided_json=None, max_tokens=1024)
    body = prompt_msgs[0]["content"]
    msgs2 = [{"role": "user", "content": body},
             {"role": "assistant", "content": r1},
             {"role": "user", "content": "Now output ONLY the final JSON object — no other text, no markdown."}]
    txt = chat(base, model, msgs2, guided_json=schema, max_tokens=512)
    return extract_json(txt, key), r1


LEVEL_ARMS = ("L0", "L1", "L2a", "L2b", "L3")


def eval_case(case, base, model, arm):
    n = case["n_items"]
    golds = [e["gold_new_item_id"] for e in case["exchanges"]]
    out = {"task_id": case["task_id"], "arm": arm, "n": n, "gold": golds}
    # STRONG FORM + ablation: scaffolded incremental typed-step (Sstep=verify ON / Snover=OFF)
    if arm in ("Sstep", "Snover"):
        preds = [step_resolve(case["nl"], e, base, model, verify=(arm == "Sstep"))[0] for e in case["exchanges"]]
        out.update(parse_fail=False, pred=preds, item_correct=[preds[i] == golds[i] for i in range(n)])
        return out
    # SELF-CONSISTENCY baseline: A-style concrete sampled N times (temp>0), majority vote per item
    if arm == "SCv":
        N = 5
        votes = [collections.Counter() for _ in range(n)]
        for _ in range(N):
            txt = chat(base, model, prompt_concrete_level(case, "L1"), guided_json=CONCRETE_SCHEMA, temperature=0.7)
            obj = extract_json(txt, "new_item_ids")
            ids = (list(obj["new_item_ids"]) + [None] * n)[:n] if obj else [None] * n
            for i in range(n):
                if ids[i] is not None:
                    votes[i][ids[i]] += 1
        preds = [v.most_common(1)[0][0] if v else None for v in votes]
        out.update(parse_fail=False, pred=preds, item_correct=[preds[i] == golds[i] for i in range(n)])
        return out
    # info-FLOOR ladder: concrete output held constant, input info level varies
    if arm in LEVEL_ARMS:
        txt = chat(base, model, prompt_concrete_level(case, arm), guided_json=CONCRETE_SCHEMA)
        obj = extract_json(txt, "new_item_ids")
        if not obj:
            out.update(parse_fail=True, pred=None, item_correct=[False]*n); return out
        pred = (list(obj["new_item_ids"]) + [None]*n)[:n]
        out.update(parse_fail=False, pred=pred, item_correct=[pred[i] == golds[i] for i in range(n)])
        return out
    mode, cot, grammar, twocall = ARM_SPEC[arm]
    if mode == "concrete":
        if twocall:
            obj, _ = _twocall(base, model, prompt_concrete(case, cot=True), CONCRETE_SCHEMA, "new_item_ids")
        else:
            gj = CONCRETE_SCHEMA if grammar else None
            txt = chat(base, model, prompt_concrete(case, cot), guided_json=gj, max_tokens=1024 if cot else 512)
            obj = extract_json(txt, "new_item_ids")
        if not obj:
            out.update(parse_fail=True, pred=None, item_correct=[False]*n); return out
        pred = (list(obj["new_item_ids"]) + [None]*n)[:n]
        out.update(parse_fail=False, pred=pred,
                   item_correct=[pred[i] == golds[i] for i in range(n)])
        return out
    # formal family (B/C/Bcot/Btwo + Bfair*): mode picks the prompt builder; resolve path shared
    pf = PROMPT_FN[mode]
    if twocall:
        obj, _ = _twocall(base, model, pf(case, cot=True), SELECTOR_SCHEMA, "selectors")
    else:
        gj = SELECTOR_SCHEMA if grammar else None
        txt = chat(base, model, pf(case, cot), guided_json=gj, max_tokens=1024 if cot else 512)
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
        a["calls"] = a.get("calls", 0) + r.get("n_calls", 0)
        a["ptok"] = a.get("ptok", 0) + r.get("prompt_tokens", 0)
        a["ctok"] = a.get("ctok", 0) + r.get("completion_tokens", 0)
        for k in r.get("fail_kind", []):
            if k != "ok":
                a["kinds"][k.split(":")[0]] = a["kinds"].get(k.split(":")[0], 0) + 1
    return by_arm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="/home/woori/scratch/ma_eval_cases.jsonl")
    ap.add_argument("--base", default="http://localhost:8013/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--arms", default="A,Acot,Atwo,B,Bcot,Btwo,C")
    ap.add_argument("--out", default="/home/woori/scratch/ma_eval_results.jsonl")
    args = ap.parse_args()
    cases = [json.loads(l) for l in open(args.cases, encoding="utf-8")]
    results = []
    with open(args.out, "w", encoding="utf-8") as f:
        for arm in args.arms.split(","):
            for c in cases:
                _USAGE.clear()
                try:
                    r = eval_case(c, args.base, args.model, arm)
                except Exception as e:
                    r = {"task_id": c["task_id"], "arm": arm, "n": c["n_items"],
                         "error": str(e), "item_correct": [False]*c["n_items"]}
                # cost proxies (attached regardless of eval_case's internal early returns)
                r["n_calls"] = len(_USAGE)
                r["prompt_tokens"] = sum(u.get("prompt_tokens", 0) for u in _USAGE)
                r["completion_tokens"] = sum(u.get("completion_tokens", 0) for u in _USAGE)
                results.append(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"arm {arm} done ({len(cases)} cases)")
    print("\n=== SUMMARY ===")
    for arm, a in sorted(summarize(results).items()):
        ic = a["item_correct"] / max(a["items"], 1)
        cc = a["case_correct"] / max(a["cases"], 1)
        tot_tok = a.get("ptok", 0) + a.get("ctok", 0)
        print(f"arm {arm}: item-acc={ic:.3f} ({a['item_correct']}/{a['items']}) "
              f"case-acc={cc:.3f} ({a['case_correct']}/{a['cases']}) "
              f"parse_fail={a['parse_fail']} fail_kinds={a['kinds']} | "
              f"calls={a.get('calls',0)} tok={tot_tok} (p{a.get('ptok',0)}/c{a.get('ctok',0)})")
