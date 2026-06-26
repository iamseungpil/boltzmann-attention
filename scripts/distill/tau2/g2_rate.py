#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""g2_rate.py — §7-1: g2(confirm-before-write violation) PER-WRITE-OPPORTUNITY rate + CI, by scale.
Settles the §5.3 compliance backbone vs the completion-rate confound:
  compliance.json g2 = # SIMS with >=1 unconfirmed write -> confounded (higher-pass model reaches more writes).
  Correct denominator = # executed write tool-calls. rate = unconfirmed_writes / total_writes.
  If flat across scale (overlapping Wilson CIs) -> STRONG "scale-invariant". If falls -> DIFFERENTIAL only.
Reuses t2_compliance g2 logic (write not preceded by a confirming user turn). Floor runs (no gate -> all writes execute). gpt-4.1 0.
"""
import json, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_compliance import domain_constants
from gate_interpreter import CONFIRM_RE

DOM = "/home/woori/scratch/tau2-bench/data/simulations"
RUNS = [("7B", "on_n7b_floor_retail"), ("14B", "on_n14b_floor_retail"), ("32B", "on_n32int8_floor_retail")]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def count(run, C):
    d = json.load(open(f"{DOM}/{run}/results.json"))
    sims = d["simulations"] if isinstance(d, dict) else d
    WRITE = C["WRITE_TOOLS"]
    total_w = g2_w = sims_w = g2_sims = n = 0
    for s in sims:
        n += 1
        msgs = s.get("messages") or []
        rbyid = {m["id"]: m for m in msgs if m.get("role") == "tool" and m.get("id")}
        last_user = None
        had_w = had_g2 = False
        for m in msgs:
            role, mc = m.get("role"), m.get("content")
            if role == "user" and isinstance(mc, str) and mc.strip():
                last_user = mc
                continue
            if role != "assistant":
                continue
            for tc in (m.get("tool_calls") or []):
                name = tc.get("name")
                res = rbyid.get(tc.get("id"))
                content = (res or {}).get("content") or ""
                if not isinstance(content, str):
                    content = str(content)
                if "POLICY GATE" in content:
                    continue  # denied = not executed
                if name in WRITE:
                    total_w += 1
                    had_w = True
                    if last_user is None or not CONFIRM_RE.search(last_user):
                        g2_w += 1
                        had_g2 = True
        sims_w += had_w
        g2_sims += had_g2
    return n, total_w, g2_w, sims_w, g2_sims


def main():
    C = domain_constants("retail")
    print("=== g2 (confirm-before-write) PER-WRITE-OPPORTUNITY rate by scale (floor·gpt-4.1 0) ===\n")
    print(f"  {'scale':>5} {'n_sims':>6} {'writes':>7} {'g2_w':>5} {'rate=g2/write':>14} {'95% Wilson CI':>16}  {'g2_sims/n':>9} {'g2_sims/wrote':>13}")
    rows = []
    for tag, run in RUNS:
        n, tw, g2w, sw, g2s = count(run, C)
        lo, hi = wilson(g2w, tw)
        rate = g2w / tw if tw else float("nan")
        rows.append((tag, rate, lo, hi))
        print(f"  {tag:>5} {n:>6} {tw:>7} {g2w:>5} {rate:>14.3f} {f'[{lo:.3f},{hi:.3f}]':>16}  {g2s/n:>9.3f} {(g2s/sw if sw else float('nan')):>13.3f}")
    print("\n  해석: rate=g2/write가 scale 무관히 flat(CI 겹침) → STRONG scale-invariant.")
    print("        rate가 scale↑로 falls → DIFFERENTIAL만(g1 붕괴 vs g2 잔존).")
    # verdict
    r7, r32 = rows[0][1], rows[2][1]
    ci7, ci32 = (rows[0][2], rows[0][3]), (rows[2][2], rows[2][3])
    overlap = not (ci7[1] < ci32[0] or ci32[1] < ci7[0])
    print(f"\n  7B rate {r7:.3f} vs 32B rate {r32:.3f} · CI overlap(7B,32B)={overlap} → "
          f"{'FLAT (strong claim survives)' if overlap else 'rate moves (state differential)'}")


if __name__ == "__main__":
    main()
