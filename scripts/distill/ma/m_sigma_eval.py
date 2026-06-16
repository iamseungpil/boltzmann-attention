#!/usr/bin/env python3
"""M-sigma IN-DIST eval (§5 gate, before transfer): did the model learn to EMIT the typed
$ref derivation for copy-threaded args? Reconstructs cfb context up to a threaded tool-call,
asks the served model for the next call, and scores whether it emits the CORRECT $ref path
for args that gold derives from a prior output. Compares M-sigma adapter vs base.
NOTE: cfb was used in training, so this is a TRAIN-SET learning check (did training take),
NOT generalization — labeled as such. Transfer (tau2) is the separate M-D eval.
"""
import json, argparse, requests, sys
sys.path.insert(0, "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/ma")
from m_sigma_data import build_example  # noqa


def chat(base, model, messages, max_tokens=512):
    r = requests.post(f"{base}/chat/completions",
                      json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0},
                      timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def first_threaded_call(rec):
    """Return (context_messages_upto_call, gold_tool_name, gold_args_with_refs) for the first
    assistant tool-call that contains a $ref (threaded) arg, using build_example's derivation."""
    triple, na, nr, nrt = build_example(rec)
    if nr == 0 or nr != nrt:
        return None
    msgs = triple["messages"]
    ctx = []
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tc = m["tool_calls"][0]
            args = json.loads(tc["function"]["arguments"])
            if any(isinstance(v, dict) and "$ref" in v for v in args.values()):
                return ctx + [{"role": m["role"], "content": m.get("content")}], tc["function"]["name"], args
            ctx.append({k: m[k] for k in ("role", "content", "tool_calls") if k in m})
        else:
            ctx.append({k: m[k] for k in ("role", "content") if k in m})
    return None


def parse_call(msg):
    if msg.get("tool_calls"):
        tc = msg["tool_calls"][0]
        try: return tc["function"]["name"], json.loads(tc["function"]["arguments"])
        except Exception: return tc["function"].get("name"), {}
    # fallback: parse JSON in content
    import re
    m = re.search(r"\{.*\}", str(msg.get("content") or ""), re.DOTALL)
    if m:
        try:
            o = json.loads(m.group(0)); return o.get("name"), o.get("arguments", o)
        except Exception: pass
    return None, {}


def score(base, model, cases, tag):
    n = ref_total = ref_correct = name_ok = emitted_ref = 0
    for ctx, gname, gargs in cases:
        msg = chat(base, model, ctx)
        pn, pargs = parse_call(msg)
        n += 1
        if pn == gname: name_ok += 1
        for k, gv in gargs.items():
            if isinstance(gv, dict) and "$ref" in gv:
                ref_total += 1
                pv = pargs.get(k) if isinstance(pargs, dict) else None
                if isinstance(pv, dict) and "$ref" in pv:
                    emitted_ref += 1
                    if pv["$ref"] == gv["$ref"]:
                        ref_correct += 1
    print(f"[{tag}] n={n} name_ok={name_ok}/{n}  $ref-emitted={emitted_ref}/{ref_total}  "
          f"$ref-CORRECT-path={ref_correct}/{ref_total} ({ref_correct/max(ref_total,1):.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/woori/scratch/fc_build/cfb.jsonl")
    ap.add_argument("--base", default="http://localhost:8014/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    cases = []
    for l in open(args.src, encoding="utf-8"):
        r = first_threaded_call(json.loads(l))
        if r: cases.append(r)
        if len(cases) >= args.n: break
    print(f"loaded {len(cases)} threaded-call cases")
    score(args.base, args.model, cases, args.tag)
