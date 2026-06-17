#!/usr/bin/env python3
"""B(L,width) sweep — is multi-attribute delta extraction a budget WALL that PERSISTS at scale, or
does it dissolve with model size? This decides whether width-offload (decomposition: emit one attr-
change at a time, engine assembles) is NECESSARY or just a small-model crutch. (M_A_RESULTS §20: base
7B set-extraction accuracy falls with width 0.64->0.40->0.29->0.25.)

Controlled synthetic substitute (synth_depth, width = #attrs changed, rest kept). Two arms, the model
given EVERY advantage (operator gloss on, explicit prompt) so failure = genuine width-binding limit:
  A  in-head : catalog + old item + NL -> output the new item id directly (extract delta + apply + match).
  B  op-IR   : emit {op:substitute, set:{...}}; we score (i) SET-EXACT (emitted set == gold set = the
               pure FORMALIZE metric) and (ii) item accuracy via the deterministic engine.

Endpoint-agnostic: --base + --model hits any OpenAI-compatible server (vLLM-served open weights) or
OpenRouter frontier (--base https://openrouter.ai/api/v1 --model openai/gpt-4.1 --api_key_file ...).
Per width: arm_A item acc, arm_B item acc, SET-EXACT acc, mean set-recall (captured/width), set-size bias.

THESIS read: if SET-EXACT(width) stays high at frontier -> scale matches width, offload NOT necessary.
If it decays at every scale up to frontier -> width is a budget wall -> decomposition-offload necessary.
"""
import json, argparse, random, re
from collections import defaultdict
from synth_depth import gen_example, render_nl_diverse, resolve_operation
from depth_eval import build_arm_B_user, extract_json


def chat(base, model, messages, max_tokens=256, api_key=None):
    import requests
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = requests.post(f"{base}/chat/completions", headers=headers,
                      json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0},
                      timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"].get("content") or ""


def arm_A_user(ex):
    items = [{k: v for k, v in it.items() if v is not None} for it in ex["items"]]  # drop None ord attr
    aid = ex["op_ir"]["anchor_id"]
    old = next((it for it in ex["items"] if it["id"] == aid), None)
    oldopt = {k: v for k, v in (old or {}).items() if k != "id" and v is not None}
    return ("Catalog (each item = id + options):\n" + json.dumps(items, ensure_ascii=False) +
            f"\n\nThe customer currently has item {aid} with options {json.dumps(oldopt, ensure_ascii=False)}.\n"
            f"Request: {ex['nl']}\n\n"
            'Apply the requested change(s), KEEPING every unmentioned option the same, and find the catalog '
            'item with the resulting options. Output ONLY {"id": "<that item id>"}.')


def set_match(emit, gold):
    if not isinstance(emit, dict):
        return 0.0, False
    hit = sum(1 for k, v in gold.items() if str(emit.get(k)) == str(v))
    recall = hit / max(len(gold), 1)
    exact = (set(emit) == set(gold)) and all(str(emit.get(k)) == str(v) for k, v in gold.items())
    return recall, exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--widths", default="1,2,3,4,5")
    ap.add_argument("--n", type=int, default=100, help="examples per width")
    ap.add_argument("--N", type=int, default=12, help="catalog size")
    ap.add_argument("--seed", type=int, default=20260618)
    ap.add_argument("--arms", default="A,B")
    ap.add_argument("--gloss", type=int, default=1)
    ap.add_argument("--api_key_file", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    api_key = None
    if args.api_key_file:
        raw = open(args.api_key_file).read().strip()
        # tolerate `export OPENROUTER_API_KEY='sk-...'` shell-export format, not just a bare key
        if "=" in raw:
            raw = raw.split("=", 1)[1]
        api_key = raw.strip().strip("'").strip('"').strip()
    widths = [int(x) for x in args.widths.split(",")]
    arms = [a.strip() for a in args.arms.split(",")]
    res = {"model": args.model, "tag": args.tag, "by_width": {}}
    for W in widths:
        rng = random.Random(args.seed + W)
        exs = []
        while len(exs) < args.n:
            ex = gen_example(rng, 1, "substitute", args.N, width=W)
            if ex is None:
                continue
            ex["nl"] = render_nl_diverse(ex, rng)
            exs.append(ex)
        a_ok = b_ok = sx = 0; rec = 0.0; szbias = 0
        for ex in exs:
            if "A" in arms:
                outA = chat(args.base, args.model, [{"role": "system", "content": "Output ONLY JSON."},
                            {"role": "user", "content": arm_A_user(ex)}], 80, api_key)
                jA = extract_json(outA)
                a_ok += int((jA or {}).get("id") == ex["gold_id"])
            if "B" in arms:
                outB = chat(args.base, args.model, [{"role": "system", "content": "Output ONLY JSON."},
                            {"role": "user", "content": build_arm_B_user(ex, gloss=bool(args.gloss))}], 256, api_key)
                ir = extract_json(outB)
                # ground the anchor (offload concrete id) so we isolate the SET extraction
                if isinstance(ir, dict):
                    ir["anchor_id"] = ex["op_ir"]["anchor_id"]
                emit = (ir or {}).get("set") or {}
                r, ex_exact = set_match(emit, ex["op_ir"]["set"])
                rec += r; sx += int(ex_exact); szbias += (len(emit) - W)
                b_ok += int(resolve_operation(ir, ex["items"]) == ex["gold_id"] if ir else 0)
        n = len(exs)
        row = {"n": n, "arm_A_item": a_ok / n if "A" in arms else None,
               "arm_B_item": b_ok / n if "B" in arms else None,
               "set_exact": sx / n if "B" in arms else None,
               "set_recall": rec / n if "B" in arms else None,
               "set_size_bias": szbias / n if "B" in arms else None}
        res["by_width"][W] = row
        print(f"[{args.tag}] width={W} n={n} | "
              f"A_item={row['arm_A_item']} B_item={row['arm_B_item']} "
              f"SET_EXACT={row['set_exact']} set_recall={row['set_recall'] and round(row['set_recall'],3)} "
              f"size_bias={row['set_size_bias'] and round(row['set_size_bias'],2)}")
    if args.out:
        json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
