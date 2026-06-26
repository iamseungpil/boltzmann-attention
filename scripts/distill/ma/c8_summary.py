#!/usr/bin/env python3
"""Aggregate C8 batch results into one table. Reads /home/woori/scratch/depth/c8/results/*.json
(written by c8_batch.sh). For each model (variant) prints held-out (S2=transfer) and in-dist
(S3=convergence) overall-B, recognition, and per-op recognition — vs BASE_S0 (gloss-off floor)
and BASE_S1 (gloss-on ceiling). Verdict per variant: did held-out comparative recognition move
from the S0 floor toward the S1 ceiling = procedure routing internalised in weights (transfer)?
Usage: python c8_summary.py [--dir /home/woori/scratch/depth/c8/results]
"""
import json, glob, argparse, os


def load(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return None


def frac(x):
    return x[0] / x[1] if x and x[1] else float("nan")


def row(d):
    if not d:
        return None
    bo = d.get("by_op_recog", {})
    return {
        "B": frac(d["overall"].get("B", [0, 0])),
        "recog": frac(d.get("recognition", [0, 0])),
        "cmp_recog": frac(bo.get("comparative", [0, 0])),
        "per_op": {k: frac(v) for k, v in bo.items()},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/woori/scratch/depth/c8/results")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, "*.json")))
    by_model = {}
    for f in files:
        name = os.path.basename(f).replace(".json", "")
        model, _, split = name.rpartition("__")  # TAG__heldout / TAG__indist
        by_model.setdefault(model, {})[split] = row(load(f))

    floor = (by_model.get("BASE_S0", {}).get("heldout") or {}).get("cmp_recog", float("nan"))
    ceil = (by_model.get("BASE_S1", {}).get("heldout") or {}).get("cmp_recog", float("nan"))
    print(f"=== C8 transfer summary ===  (held-out comparative recog: floor S0={floor:.2f}  ceiling S1={ceil:.2f})\n")
    hdr = f"{'variant':>16} | {'split':>8} | {'B':>5} {'recog':>6} {'cmpRecog':>8} | per-op recog"
    print(hdr); print("-" * len(hdr))
    for model in sorted(by_model):
        for split in ("heldout", "indist"):
            r = by_model[model].get(split)
            if not r:
                continue
            po = " ".join(f"{k[:4]}={r['per_op'].get(k, float('nan')):.2f}" for k in
                          ("filter", "argmax", "argmin", "rank", "comparative"))
            print(f"{model:>16} | {split:>8} | {r['B']:.2f}  {r['recog']:.2f}   {r['cmp_recog']:.2f}   | {po}")
        # verdict on transfer (held-out)
        h = by_model[model].get("heldout")
        if h and model not in ("BASE_S0", "BASE_S1"):
            c = h["cmp_recog"]
            v = ("TRANSFER" if c >= 0.8 else "PARTIAL" if c >= 0.3 else "NO-TRANSFER")
            print(f"{'':>16} |  -> held-out comparative recog={c:.2f}  [{v}]\n")
