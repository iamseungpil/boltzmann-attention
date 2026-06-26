#!/usr/bin/env python
"""밤샘 결과 → 테이블 (2026-06-22 사용자: 아침 확인용·1커맨드).
각 arm의 compliance.json(t2_run_gated 후크 자동생성·bench/full pass^k + 위반) 집계 + 실패-클래스 census.
Run: cd tau2-bench && PYTHONPATH=src:<T2> PY morning_tables.py
"""
import json, os, re, collections

SIM = "data/simulations"
SCALES = [("n7b", "7B"), ("n14b", "14B"), ("n32int8", "32B-int8")]


def comp(arm):
    p = os.path.join(SIM, arm, "compliance.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def cell(arm):
    c = comp(arm)
    if not c:
        return "  --  /  --  "
    b = c.get("bench", {}).get("pass^1")
    f = c.get("full", {}).get("pass^1")
    return f"{b:.3f}/{f:.3f}" if b is not None else "  --  "


def fclass(arm, domain="retail"):
    """floor 대비 비교용 실패-클래스 census (eligibility/loop/no-write·THIRTYB §6c)."""
    p = os.path.join(SIM, arm, "results.json")
    if not os.path.exists(p):
        return None
    j = json.load(open(p)); sims = j.get("simulations", j) if isinstance(j, dict) else j
    WR = {"modify_pending_order_items", "modify_pending_order_address", "modify_user_address",
          "cancel_pending_order", "return_delivered_order_items", "exchange_delivered_order_items",
          "modify_pending_order_payment"}
    elig = re.compile(r"non-pending|not pending|not delivered|non-delivered|must be (pending|delivered)|precondition|POLICY GATE G5", re.I)
    tot = elig_n = loop_n = nowrite_n = npass = 0
    for s in sims:
        r = (s.get("reward_info") or {}).get("reward")
        if r is None:
            continue
        tot += 1
        if r >= 1:
            npass += 1; continue
        nerr = ee = 0; hw = False
        for m in s.get("messages", []):
            if m.get("role") == "tool" and m.get("error"):
                nerr += 1
                if elig.search(str(m.get("content") or "")):
                    ee += 1
            for tc in (m.get("tool_calls") or []):
                nm = (tc.get("function") or {}).get("name") or tc.get("name")
                if nm in WR:
                    hw = True
        if ee:
            elig_n += 1
        if nerr >= 4:
            loop_n += 1
        if not hw:
            nowrite_n += 1
    return dict(n=tot, npass=npass, elig=elig_n, loop=loop_n, nowrite=nowrite_n)


print("=" * 78)
print("TABLE 2 — Retail flow-discipline × scale  (bench/compliant pass^1)")
print("=" * 78)
arms2 = [("floor", "on_{t}_floor_retail"), ("g14(G1-4)", "ours_{t}_g14_retail"),
         ("g15(+G5)", "ours_{t}_g15_retail"), ("g15+retry", "ours_{t}_g15retry_retail"),
         ("g5only*", "on_{t}_g5_retail"), ("g5+retry*", "on_{t}_g5retry_retail")]
print(f"{'arm':14s}" + "".join(f"{lab:>16s}" for _, lab in SCALES))
for name, tmpl in arms2:
    row = f"{name:14s}"
    for tag, _ in SCALES:
        row += f"{cell(tmpl.format(t=tag)):>16s}"
    print(row)
print("  (* on_n32int8_g5/g5retry = 32B G5-isolation arms·G1-4 OFF; 7/14B은 미실행)")

print()
print("=" * 78)
print("TABLE 1 — Cross-domain floor × scale  (bench/compliant pass^1; airline/bank compliant=caveat)")
print("=" * 78)
doms = [("retail", "on_{t}_floor_retail"), ("airline", "ours_{t}_floor_airline"),
        ("banking", "ours_{t}_floor_bank")]
print(f"{'domain':14s}" + "".join(f"{lab:>16s}" for _, lab in SCALES))
for name, tmpl in doms:
    row = f"{name:14s}"
    for tag, _ in SCALES:
        row += f"{cell(tmpl.format(t=tag)):>16s}"
    print(row)

print()
print("=" * 78)
print("FAILURE-CLASS census (retail) — floor vs g15 vs g15+retry  (scaffold-addressable Δ)")
print("=" * 78)
print(f"{'arm/model':22s}{'n':>5s}{'pass':>6s}{'elig/wrongtool':>16s}{'loop':>7s}{'no-write':>10s}")
for tag, lab in SCALES:
    for arm in (f"on_{tag}_floor_retail", f"ours_{tag}_g15_retail", f"ours_{tag}_g15retry_retail"):
        fc = fclass(arm)
        if fc:
            print(f"{arm:22s}{fc['n']:>5d}{fc['npass']:>6d}{fc['elig']:>16d}{fc['loop']:>7d}{fc['nowrite']:>10d}")
print("\n[done] morning_tables")
