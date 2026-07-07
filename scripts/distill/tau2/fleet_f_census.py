#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fleet_f_census.py — reproducible denominator census for FLEET_FUNCTION_DELEGATION §4b (2026-07-07 review-fix).
Replaces the lost /tmp/f2.py. Measures the trajectory serving-cost token scale from persisted results,
so the fleet cost multiplier fleet/small = 1 + R*delegated/denom uses a correct denominator.

Denominators (per sim, over agent/assistant turns, from usage.prompt_tokens / completion_tokens):
  peak       = max single-turn prompt_tokens                 (context high-water mark)
  throughput = sum(prompt_tokens) + sum(completion_tokens)   (NAIVE no-cache serving cost; OVERSTATES)
  realistic ~= peak + decode  (with vLLM prefix-caching, prefill telescopes to ~peak)  <- use this
Old §4b used 3755 (a single mid-trajectory snapshot) = 3x too small vs peak, 27x vs throughput = ERROR.
"""
import json, gzip, glob, statistics as st, sys

BASE = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results/"
DELEGATED = 392 * 1.92   # ~753 tok: isolated cross-order sub-call * 1.92 per task

def census(tag):
    d = json.load(gzip.open(BASE + tag + ".results.json.gz"))
    peaks, thru, decode, nt = [], [], [], []
    for s in d["simulations"]:
        p, comp = [], 0
        for m in s.get("messages", []):
            if m.get("role") == "assistant":
                u = m.get("usage") or {}
                if u.get("prompt_tokens"):
                    p.append(u["prompt_tokens"]); comp += (u.get("completion_tokens") or 0)
        if not p:
            continue
        peaks.append(max(p)); thru.append(sum(p) + comp); decode.append(comp); nt.append(len(p))
    return peaks, thru, decode, nt

def sm(x): return f"median={int(st.median(x))} mean={int(st.mean(x))} max={max(x)}"

if __name__ == "__main__":
    for tag in (sys.argv[1:] or ["asmregen32b_regen_retail_t4", "fl32b_floor_retail_t4"]):
        peaks, thru, decode, nt = census(tag)
        real = [p + c for p, c in zip(peaks, decode)]   # realistic prefix-cached basis
        print(f"\n== {tag} (n={len(peaks)}) ==")
        print(" peak          :", sm(peaks))
        print(" throughput    :", sm(thru), "(naive no-cache; OVERSTATES)")
        print(" realistic(pk+dec):", sm(real), "<- correct denom")
        print(" agent turns   :", sm(nt))
        for name, denom in [("realistic", st.mean(real)), ("throughput", st.mean(thru)), ("peak", st.mean(peaks))]:
            print(f"   fleet/small @ {name} {int(denom)}:  R=2.25 -> {1+2.25*DELEGATED/denom:.2f}x   R=20 -> {1+20*DELEGATED/denom:.2f}x")
