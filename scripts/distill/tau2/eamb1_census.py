#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E-AMB-1/2 - |C| census + 실패율 단조성 (THEORY_AMBIGUITY T1·T2).

정본: reports/facet_rft_2026/E_AMB_MEASUREMENT_PLAN_2026_07_10.md §1-§2

T1: 결정점(write 인자)마다 후보집합 C(d) = 접두 문맥(도구출력)에 실재하는 같은-형식 값들.
    census 분포·|C|>=2 비율 -> C48 45.8%와 교차검증.
T2: 결정점 정오(gold action_checks 인자 대조·C20 노이즈 caveat) x |C| 층 단조성.
    교락 통제: within-task 대조(MH) + 공변량 조감 + 인자타입 층화.

[[05]] 준수: 엔진은 도메인-일반(형식 열거·집합 대조). 도메인 지식은 아래 A2 블록(데이터)에만.
[[08]] 준수: 종료사유 분포 출력 · per-case 덤프(각 층 실패 예시) 필수.

Run: py scripts/distill/tau2/eamb1_census.py [--sim fl32b_floor_retail_t4] [--ctx tool|tool+user]
"""
import argparse
import gzip
import json
import math
import re
from collections import Counter, defaultdict

import os as _os
SIM_DIR = _os.environ.get("EAMB_SIM_DIR", r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results")

# ---------------- A2 (도메인 데이터 · 엔진 아님) ----------------
WRITE = {"return_delivered_order_items", "exchange_delivered_order_items", "cancel_pending_order",
         "modify_pending_order_items", "modify_pending_order_address", "modify_pending_order_payment",
         "modify_user_address", "place_order"}
# 인자 -> 값 형식 (열거용 정규식) : C51 CAND_PAT와 동일 계열 (교차검증 정합 목적)
A2_FORMAT = {
    "payment_method_id": r"(?:credit_card|gift_card|paypal)_\d+",
    "address1": r"\d+ [A-Z][a-z]+ (?:Street|Avenue|Drive|Lane|Road|Boulevard|Way|Court)",
    "new_item_ids": r"\b\d{10}\b",
    "item_ids": r"\b\d{10}\b",
    "order_id": r"#?W\d{7}",
}
ARGS = tuple(A2_FORMAT)
# ----------------------------------------------------------------


def norm(x):
    return re.sub(r"\s+", " ", str(x).lower().replace("#", "")).strip()


def ctx_text(sim, idx, mode):
    roles = ("tool",) if mode == "tool" else ("tool", "user")
    return " ".join(str(m.get("content")) for m in sim["messages"][:idx]
                    if m.get("role") in roles and isinstance(m.get("content"), str))


def candidates(sim, idx, key, mode):
    pat = A2_FORMAT[key]
    seen, out = set(), []
    for m in re.findall(pat, ctx_text(sim, idx, mode)):
        n = norm(m)
        if n not in seen:
            seen.add(n)
            out.append(m)
    return out


def as_set(v):
    if v is None:
        return None
    vals = v if isinstance(v, list) else [v]
    return frozenset(norm(x) for x in vals if x is not None)


def gold_match(sim, tc, key):
    """같은 이름 gold action, order_id 일치 우선(min-diff 대용). return (status, gold_set)."""
    wname = tc.get("name")
    oid = norm((tc.get("arguments") or {}).get("order_id") or "")
    acts = [(a.get("action") or {}) for a in (sim.get("reward_info") or {}).get("action_checks") or []]
    cand = [a for a in acts if a.get("name") == wname]
    if not cand:
        return "NO-WRITE", None
    best = cand[0]
    for c in cand:
        if norm((c.get("arguments") or {}).get("order_id") or "") == oid:
            best = c
            break
    gv = (best.get("arguments") or {}).get(key)
    if gv is None:
        return "NO-ARG", None
    return "OK", as_set(gv)


def ca_trend(groups):
    """Cochran-Armitage trend z. groups: [(score, fail, n)]"""
    N = sum(n for _, _, n in groups)
    F = sum(f for _, f, n in groups)
    if N == 0 or F == 0 or F == N:
        return float("nan")
    p = F / N
    sbar = sum(s * n for s, _, n in groups) / N
    num = sum(f * (s - sbar) for s, f, _ in groups)
    var = p * (1 - p) * sum(n * (s - sbar) ** 2 for s, _, n in groups)
    return num / math.sqrt(var) if var > 0 else float("nan")


def bucket(c):
    return "0" if c == 0 else ("1" if c == 1 else ("2" if c == 2 else "3+"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="fl32b_floor_retail_t4")
    ap.add_argument("--ctx", default="tool", choices=["tool", "tool+user"])
    ap.add_argument("--dump", type=int, default=6)
    a = ap.parse_args()

    data = json.load(gzip.open(f"{SIM_DIR}\\{a.sim}.results.json.gz", "rt", encoding="utf-8"))
    sims = data["simulations"]

    # [[08]] (1) 종료사유 분포
    print("=== 종료사유 분포 (n=%d) ===" % len(sims))
    print(" ", dict(Counter(s.get("termination_reason") for s in sims)))

    rows = []
    for sim in sims:
        for idx, m in enumerate(sim["messages"]):
            if m.get("role") != "assistant":
                continue
            for tc in (m.get("tool_calls") or []):
                if tc.get("name") not in WRITE:
                    continue
                args = tc.get("arguments") or {}
                for key in ARGS:
                    if key not in args or args.get(key) in (None, [], ""):
                        continue
                    cand = candidates(sim, idx, key, a.ctx)
                    st, gset = gold_match(sim, tc, key)
                    vset = as_set(args.get(key))
                    correct = (vset == gset) if st == "OK" else None
                    gold_in = (set(gset) <= {norm(c) for c in cand}) if gset else None
                    rows.append({
                        "task": str(sim.get("task_id")), "trial": sim.get("trial"),
                        "tool": tc.get("name"), "arg": key, "idx": idx,
                        "val": sorted(vset or []), "gold": sorted(gset) if gset else None,
                        "C": len(cand), "cand": cand[:8], "status": st, "correct": correct,
                        "gold_in": gold_in,
                        "ctxlen": sum(len(str(mm.get("content") or "")) for mm in sim["messages"][:idx]),
                        "reads": sum(1 for mm in sim["messages"][:idx] if mm.get("role") == "assistant"
                                     for t in (mm.get("tool_calls") or []) if t.get("name") not in WRITE),
                    })

    print("\n=== T1 census (ctx=%s) ===" % a.ctx)
    print("결정점 %d개 (write-call x 인자)" % len(rows))
    cc = Counter(bucket(r["C"]) for r in rows)
    tot = len(rows)
    for b in ("0", "1", "2", "3+"):
        print("  |C|=%-2s : %4d  (%.1f%%)" % (b, cc[b], 100 * cc[b] / tot))
    ge2 = sum(1 for r in rows if r["C"] >= 2)
    print("  |C|>=2 비율: %.1f%%  (C48 교차검증 대상: 45.8%%)" % (100 * ge2 / tot))
    print("  인자타입별 |C|>=2:", {k: "%d/%d" % (sum(1 for r in rows if r["arg"] == k and r["C"] >= 2),
                                              sum(1 for r in rows if r["arg"] == k)) for k in ARGS})
    # gold in C 검사 (T1 반증조건: |C|>0인데 gold가 C 밖 >10%면 열거기 결함)
    ok = [r for r in rows if r["status"] == "OK" and r["gold"]]
    pos = [r for r in ok if r["C"] > 0]
    gout = sum(1 for r in pos if not r["gold_in"])
    print("  |C|>0 중 gold∉C: %d/%d (%.1f%%)  [반증조건 >10%%]" % (gout, len(pos), 100 * gout / max(len(pos), 1)))

    print("\n=== T2 실패율 x |C| (status=OK만 · n=%d) ===" % len(ok))
    grp = defaultdict(lambda: [0, 0])
    for r in ok:
        b = bucket(r["C"])
        grp[b][1] += 1
        if r["correct"] is False:
            grp[b][0] += 1
    for b in ("0", "1", "2", "3+"):
        f, n = grp[b]
        if n:
            print("  |C|=%-2s : fail %3d/%4d = %.3f" % (b, f, n, f / n))
    scores = {"0": 0, "1": 1, "2": 2, "3+": 3}
    z = ca_trend([(scores[b], grp[b][0], grp[b][1]) for b in grp if grp[b][1] > 0])
    print("  Cochran-Armitage trend z = %.2f  (양수=|C|↑ 실패↑)" % z)

    # 인자타입 층화
    print("\n  --- 인자타입별 (fail/n @ |C|=1 vs >=2) ---")
    for k in ARGS:
        r1 = [(r["correct"] is False) for r in ok if r["arg"] == k and r["C"] == 1]
        r2 = [(r["correct"] is False) for r in ok if r["arg"] == k and r["C"] >= 2]
        if r1 or r2:
            print("  %-18s |C|=1: %d/%d=%.2f   |C|>=2: %d/%d=%.2f" % (
                k, sum(r1), len(r1), sum(r1) / max(len(r1), 1),
                sum(r2), len(r2), sum(r2) / max(len(r2), 1)))

    # within-task 대조 (Mantel-Haenszel: task 층 고정, |C|>=2 vs |C|<=1)
    num = den = 0.0
    n_strata = 0
    for t, rs in defaultdict(list, {t: [r for r in ok if r["task"] == t] for t in set(r["task"] for r in ok)}).items():
        hi = [r for r in rs if r["C"] >= 2]
        lo = [r for r in rs if r["C"] <= 1]
        if not hi or not lo:
            continue
        n_strata += 1
        a1, n1 = sum(1 for r in hi if r["correct"] is False), len(hi)
        a0, n0 = sum(1 for r in lo if r["correct"] is False), len(lo)
        T = n1 + n0
        num += (a1 * (n0 - a0)) / T
        den += (a0 * (n1 - a1)) / T
    print("\n  within-task(MH) 층 %d개 · OR(|C|>=2 vs <=1) = %s" % (
        n_strata, ("%.2f" % (num / den)) if den > 0 else "inf(num=%.1f)" % num))

    # 공변량 조감: ctxlen·reads 사분위별 실패율 (교락 방향 확인)
    for cov in ("ctxlen", "reads"):
        vals = sorted(r[cov] for r in ok)
        qs = [vals[int(len(vals) * q)] for q in (0.25, 0.5, 0.75)]
        def qb(v):
            return sum(v > q for q in qs)
        t = defaultdict(lambda: [0, 0])
        for r in ok:
            b = qb(r[cov])
            t[b][1] += 1
            t[b][0] += (r["correct"] is False)
        print("  공변량 %s 사분위 실패율: %s" % (cov, {q: "%.2f" % (f / n) for q, (f, n) in sorted(t.items())}))

    # NO-WRITE/NO-ARG 계수
    print("\n  status 분포:", dict(Counter(r["status"] for r in rows)))

    # [[08]] per-case 덤프
    print("\n=== per-case 덤프 (각 층 실패 예시 · 정독 대상) ===")
    for b in ("1", "2", "3+"):
        ex = [r for r in ok if bucket(r["C"]) == b and r["correct"] is False][:a.dump]
        print("--- |C|=%s 실패 %d건 중 %d건 ---" % (b, sum(1 for r in ok if bucket(r["C"]) == b and r["correct"] is False), len(ex)))
        for r in ex:
            print("  t%s tr%s %s.%s val=%s gold=%s C=%d cand=%s" % (
                r["task"], r["trial"], r["tool"], r["arg"], r["val"], r["gold"], r["C"], r["cand"][:4]))

    out = f"{SIM_DIR}\\eamb1_{a.sim}_{a.ctx.replace('+', '_')}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
