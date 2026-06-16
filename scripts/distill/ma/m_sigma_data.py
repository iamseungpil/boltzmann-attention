#!/usr/bin/env python3
"""M-sigma data pipeline (M_SIGMA_DESIGN_2026_06_16.md): cfb (binding-rich, grounded 2-hop)
-> (NL, config, DERIVATION-spec) training triples. The target tool-calls have COPY-THREADED
args (value derived from a prior tool output) REPLACED by a typed-reference {"$ref": "<prior_
tool>#<json-path>"}, so the model learns to emit the RELATION (not the concrete value); a
deterministic resolver fills it at inference. Literal/NL args stay literal. Optional
isotropization (consistent tool/field renaming) for surface-invariance.

Why cfb not TaskBench: census (M_SIGMA §3) — TaskBench outputs are stubs {status,ref} = NO
binding. cfb has prior-output->arg derivation. v7 trained cfb at CONCRETE level (failed);
M-sigma trains the DERIVATION level.

Round-trip validation: resolve($ref, prior_outputs) must reproduce gold concrete arg. Only
examples that round-trip cleanly are emitted (honest denominator).
"""
import json, argparse, random


def _walk(obj, path=""):
    """Yield (jsonpath, scalar_value) for every scalar in a nested json."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, obj


def _resolve(prior_outputs, ref):
    """ref = '<idx>#<path>' -> the scalar at that prior output path (for round-trip check)."""
    idx_s, path = ref.split("#", 1)
    idx = int(idx_s)
    for p, v in _walk(prior_outputs[idx]):
        if p == path:
            return v
    return None


def build_example(rec):
    """Return (triple|None, n_args, n_ref, n_roundtrip_ok). triple has messages with the
    assistant targets' threaded args replaced by $ref."""
    msgs = rec.get("messages", [])
    prior_outputs = []  # parsed tool outputs in order
    out_msgs = []
    n_args = n_ref = n_rt = 0
    for m in msgs:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            new_tcs = []
            for tc in m["tool_calls"]:
                fn = tc["function"]; args = fn.get("arguments")
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except Exception: args = {}
                new_args = {}
                for k, v in (args.items() if isinstance(args, dict) else []):
                    n_args += 1
                    ref = None
                    # copy-threading detection: value matches a scalar in a prior output
                    for oi, out in enumerate(prior_outputs):
                        for p, ov in _walk(out):
                            if ov == v and not isinstance(v, (bool, type(None))) and str(v) != "" and (not isinstance(v, str) or len(v) >= 3):
                                ref = f"{oi}#{p}"; break
                        if ref: break
                    if ref is not None:
                        n_ref += 1
                        if _resolve(prior_outputs, ref) == v:
                            n_rt += 1
                        new_args[k] = {"$ref": ref}
                    else:
                        new_args[k] = v  # literal / from-NL
                nf = dict(fn); nf["arguments"] = json.dumps(new_args, ensure_ascii=False)
                new_tcs.append({**tc, "function": nf})
            out_msgs.append({**m, "tool_calls": new_tcs})
        else:
            out_msgs.append(m)
            if role == "tool":
                try: prior_outputs.append(json.loads(m.get("content", "{}")))
                except Exception: prior_outputs.append({})
    triple = {"tools": rec.get("tools"), "messages": out_msgs, "supervise": "assistant",
              "_meta": {**(rec.get("_meta") or {}), "msigma": True}}
    return triple, n_args, n_ref, n_rt


def isotropize(triple, rng):
    """Consistent random renaming of tool names (surface-invariance augmentation). Field-level
    left for v2; tool-name alias is the cheapest covered dimension."""
    names = [t["function"]["name"] for t in triple["tools"]]
    alias = {nm: f"tool_{rng.randrange(10**6):06d}" for nm in names}
    s = json.dumps(triple, ensure_ascii=False)
    for nm, al in alias.items():
        s = s.replace(json.dumps(nm), json.dumps(al))
    return json.loads(s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/woori/scratch/fc_build/cfb.jsonl")
    ap.add_argument("--out", default="/home/woori/scratch/fc_build/msigma_train.jsonl")
    ap.add_argument("--iso", type=int, default=1, help="isotropization copies per example (0=off)")
    ap.add_argument("--validate_only", action="store_true")
    args = ap.parse_args()
    rng = random.Random(0)
    tot_args = tot_ref = tot_rt = n_rec = n_emit = 0
    with open(args.out, "w", encoding="utf-8") as w:
        for l in open(args.src, encoding="utf-8"):
            try: rec = json.loads(l)
            except Exception: continue
            n_rec += 1
            triple, na, nr, nrt = build_example(rec)
            tot_args += na; tot_ref += nr; tot_rt += nrt
            if nr == 0:  # no binding in this example -> skip (we want binding-rich)
                continue
            if nr != nrt:  # some ref doesn't round-trip -> skip (honest)
                continue
            if not args.validate_only:
                w.write(json.dumps(triple, ensure_ascii=False) + "\n"); n_emit += 1
                for _ in range(args.iso):
                    w.write(json.dumps(isotropize(triple, rng), ensure_ascii=False) + "\n"); n_emit += 1
    print(f"records={n_rec}  args={tot_args}  threaded-refs={tot_ref} ({tot_ref/max(tot_args,1):.2f})  "
          f"round-trip-ok={tot_rt}/{tot_ref} ({tot_rt/max(tot_ref,1):.2f})")
    print(f"emitted (binding-rich, clean round-trip) = {n_emit} lines -> {args.out}")
