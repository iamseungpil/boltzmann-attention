#!/usr/bin/env python3
"""Precise autopsy of agent_collapse (too_many_errors/max_steps) in a τ² rollout: is the collapse
driven by PARAMETER mis-prediction (right tool, wrong argument value -> 'not found' -> retry loop)
rather than generic FLOW/sequencing failure? Pairs each assistant tool_call with its tool result,
collects errored calls (tool + arguments + error), and classifies the offending parameter.

τ² msg shapes: assistant.tool_calls = [{id,name,arguments(dict)}]; tool msg = {role:tool, name?, error,
content}. order_id in db carries a '#' prefix (user says 'W2378156', db key '#W2378156') = a common
format mis-prediction; item_id/product_id are long numeric ids the agent often fabricates.
"""
import json, argparse, re
from collections import Counter


def term(s):
    return str(s.get("termination_reason") or (s.get("trajectory") or {}).get("termination_reason"))


def msgs_of(s):
    return s.get("messages") or (s.get("trajectory") or {}).get("messages") or []


def is_err(m):
    if m.get("error") is True:
        return True
    c = str(m.get("content") or "")
    return c.strip().lower().startswith("error") or "not found" in c.lower() or "invalid" in c.lower()


def classify(tool, args, err):
    e = err.lower()
    a = json.dumps(args, ensure_ascii=False)
    if "order" in e and "not found" in e:
        oid = str((args or {}).get("order_id", ""))
        return "order_id_format" if (oid and not oid.startswith("#")) else "order_id_wrong"
    if "item" in e and "not found" in e:
        return "item_id_wrong"
    if "product" in e and "not found" in e:
        return "product_id_wrong"
    if "user" in e and "not found" in e:
        return "user_id_wrong"
    if "not found" in e:
        return "other_not_found"
    if "invalid" in e:
        return "invalid_arg"
    if "required" in e or "missing" in e:
        return "missing_arg"
    return "other_error"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()
    d = json.load(open(args.dir.rstrip("/") + "/results.json", encoding="utf-8"))
    sims = d.get("simulations") or d.get("results") or []
    collapsed = [s for s in sims if term(s).lower() in ("too_many_errors", "max_steps") or "error" in term(s).lower()]
    print(f"collapsed tasks: {len(collapsed)} / {len(sims)}")
    cat = Counter(); tool_err = Counter(); rep = Counter()
    examples = []
    for s in collapsed:
        ms = msgs_of(s)
        calls = []  # (name,args) in order
        seen_err = []
        last_call = None
        for m in ms:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    last_call = (tc.get("name"), tc.get("arguments") or {})
                    calls.append(last_call)
            elif m.get("role") == "tool" and is_err(m) and last_call:
                name, a = last_call
                err = str(m.get("content") or "")
                c = classify(name, a, err)
                cat[c] += 1; tool_err[name] += 1
                seen_err.append((name, a, err[:80], c))
        # detect retry loop: same (tool,args) errored >=2x
        sig = Counter((n, json.dumps(a, sort_keys=True)) for n, a, _, _ in seen_err)
        for k, v in sig.items():
            if v >= 2:
                rep["retry_same_args>=2"] += 1
                break
        if seen_err and len(examples) < args.show:
            examples.append((s.get("task_id") or s.get("id"), term(s), seen_err[:4]))
    print("\n=== errored-call parameter classification (all collapsed) ===")
    for c, n in cat.most_common():
        print(f"  {c:>18}: {n}")
    print(f"\n=== tool of errored call ===")
    for t, n in tool_err.most_common():
        print(f"  {t:>26}: {n}")
    print(f"\nretry-same-args loop (>=2 identical errored calls): {rep['retry_same_args>=2']}/{len(collapsed)} collapsed tasks")
    print(f"\n=== examples ===")
    for tid, tm, errs in examples:
        print(f"  task {tid} [{tm}]")
        for n, a, e, c in errs:
            print(f"    [{c}] {n}({json.dumps(a, ensure_ascii=False)[:90]}) -> {e!r}")


if __name__ == "__main__":
    main()
