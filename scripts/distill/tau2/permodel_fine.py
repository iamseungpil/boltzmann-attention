#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-MODEL fine failure decomposition across all 4 domains.
Dumps JSON: {domain: {model: {pass, nfail, G:{pct}}}}  -> for per-model comparison figure.
Reuses fine_function_decomp.decomp() (same leaf logic as pooled aggregate_fine.py).
usage: python3 permodel_fine.py /path/to/traj_dir [out.json]
"""
import sys, glob, os, json
from collections import Counter
import fine_function_decomp as F

GS = ["G1_COVERAGE", "G2_REACH", "G3_VERIFY", "G4_PERSISTENCE", "G5_SCOPE", "G6_OPERAND", "G7_REFERENCE"]

def gmap(leaf):
    l = leaf.lower()
    if "coverage" in l: return "G1_COVERAGE"
    if "reach" in l or "gather" in l: return "G2_REACH"
    if "verify" in l: return "G3_VERIFY"
    if "escalate" in l or "persist" in l: return "G4_PERSISTENCE"
    if "over-action" in l or "scope" in l: return "G5_SCOPE"
    if "guidance" in l: return "G9_GUIDANCE"
    if "arg_variant" in l or "arg_numeric" in l or l.startswith("arg_other") or "arg_categorical" in l: return "G6_OPERAND"
    if "arg_reference" in l: return "G7_REFERENCE"
    if "arg_freetext" in l: return "G6_OPERAND"
    if "wrong_op" in l: return "Xop_noise"
    return "Xother"

def dom_of(fn):
    for d in ("retail", "airline", "telecom", "banking"):
        if fn.endswith(f"_{d}"): return d
    return "?"

def model_of(lbl, dom):
    return lbl[:-(len(dom) + 1)] if lbl.endswith("_" + dom) else lbl

def main(d, out):
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    res = {dm: {} for dm in ("retail", "airline", "telecom", "banking")}
    for f in files:
        lbl = os.path.basename(f).replace(".json", "")
        dom = dom_of(lbl)
        if dom == "?":
            continue
        model = model_of(lbl, dom)
        try:
            _, leaf, nf = F.decomp(f, lbl)
        except Exception as e:
            print("ERR", lbl, e); continue
        if nf == 0:
            res[dom][model] = {"pass": None, "nfail": 0, "G": {g: 0.0 for g in GS}}
            continue
        gc = Counter()
        for k, v in leaf.items():
            gc[gmap(k)] += v
        # recover pass from decomp print? decomp doesn't return it; recompute cheaply
        res[dom][model] = {"nfail": nf, "G": {g: round(100 * gc[g] / nf, 1) for g in GS}}
    json.dump(res, open(out, "w"), indent=1)
    print("\nWROTE", out)
    # quick text table
    for dom in ("retail", "airline", "telecom", "banking"):
        print(f"\n#### {dom} ({len(res[dom])} models) ####")
        print("{:16}".format("model") + "".join(f"{g.split('_')[0]:>7}" for g in GS) + "  nfail")
        for m, v in res[dom].items():
            print(f"{m:16}" + "".join(f"{v['G'][g]:>6.0f} " for g in GS) + f" {v['nfail']}")

if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else r"C:\tmp\traj"
    out = sys.argv[2] if len(sys.argv) > 2 else r"C:\workspace\_cdp_private_local\permodel_fine.json"
    main(d, out)
