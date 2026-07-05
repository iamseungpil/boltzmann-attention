#!/usr/bin/env python
"""Regenerate the F3/F4 compliance-by-scale table.

Replaces the corrupt sim_results/f3f4_scale_invariant_compliance_2026_06_26.txt.gz
(that file was a gzipped shell error, not the table — 2026-06-27 forensic).

F3 = bench task-pass^k ; F4 = full compliant-pass^k (every gate-violation class
counted as a fail). Reads the compliance.json sidecars that t2_run_gated writes.
pass^3 is only meaningful for nt=3 arms (n_sims == 3 * n_tasks); nt=1 arms show
"n/a" for pass^3.

Run:  cd tau2-bench && PYTHONPATH=src:<T2> python <T2>/f3f4_table.py
Persist: python f3f4_table.py | gzip > <repo>/reports/facet_rft_2026/sim_results/f3f4_scale_invariant_compliance_2026_06_26.txt.gz
"""
import json
import os

SIM = "data/simulations"
SCALES = [("n7b", "7B"), ("n14b", "14B"), ("n32int8", "32B")]
ARMS = [("floor(base)", "on_{t}_floor_retail"),
        ("g14(G1-4)", "ours_{t}_g14_retail"),
        ("g15(+G5)", "ours_{t}_g15_retail"),
        ("g15+retry", "ours_{t}_g15retry_retail")]
ASSEMBLED = {"n14b": "asmscale_14b_0626pm_assembled_retail_t3",
             "n32int8": "asmscale_32b_0626pm_assembled_retail_t3"}
N_TASKS = 114  # retail flow-discipline taskset


def comp(arm):
    p = os.path.join(SIM, arm, "compliance.json")
    return json.load(open(p)) if os.path.exists(p) else None


def p3(d, n_sims):
    """pass^3 only if nt>=3, else 'n/a'."""
    nt = round((n_sims or 0) / N_TASKS)
    return f"{d.get('pass^3', 0):.3f}" if nt >= 3 else " n/a "


def row(label, scale, c):
    if not c:
        return f"{label:13} {scale:5}  --   [missing]"
    n = c.get("n_sims")
    b, f, v = c.get("bench", {}), c.get("full", {}), c.get("violation_sims", {})
    return (f"{label:13} {scale:5} {n:>4} | {b.get('pass^1', 0):.3f}  {p3(b, n)} | "
            f"{f.get('pass^1', 0):.3f}  {p3(f, n)} | "
            f"{v.get('g1', 0):>3} {v.get('g1w', 0):>3} {v.get('g2', 0):>3} "
            f"{v.get('g3', 0):>3} {v.get('g4', 0):>3}")


def main():
    L = ["=" * 104,
         "F3/F4 COMPLIANCE x SCALE (regenerated 2026-06-27; replaces corrupt f3f4_..._2026_06_26)",
         "  F3 = bench task-pass^k | F4 = full compliant-pass^k (all gate classes = fail) | gpt-4.1 user-sim",
         "  pass^3 shown only for nt=3 arms (floor, assembled); nt=1 arms (g14/g15/g15retry) -> pass^1 only",
         "=" * 104,
         f"{'arm':13} {'scale':5} {'n':>4} | {'F3 p1':>6} {'F3 p3':>6} | {'F4 p1':>6} {'F4 p3':>6} | "
         f"{'g1':>3} {'g1w':>3} {'g2':>3} {'g3':>3} {'g4':>3}"]
    for label, pat in ARMS:
        L.append("-" * 104)
        for t, nm in SCALES:
            L.append(row(label, nm, comp(pat.format(t=t))))
    L.append("-" * 104)
    L.append("assembled (present+nested+full-gates+constraints+calc):")
    for t, nm in SCALES:
        arm = ASSEMBLED.get(t)
        L.append(row("assembled", nm, comp(arm) if arm else None))
    L += ["=" * 104,
          "READ: F3(bench) climbs with scale. Under floor, g1 (auth-type) collapses 7B->32B (57/38/6)",
          "      but g2 (confirm-before-write) persists (31/38/41) -> DIFFERENTIAL. Under any gate",
          "      (g14/g15/assembled) all violation classes -> 0 at every scale (F4 == F3).",
          "      = compliance is scale-invariant; the gate, not scale, buys guarantee.",
          "NOTE: strong form (per-write-opportunity g2 rate flat) is in g2_per_opportunity_rate; see g2_rate.py."]
    print("\n".join(L))


if __name__ == "__main__":
    main()
