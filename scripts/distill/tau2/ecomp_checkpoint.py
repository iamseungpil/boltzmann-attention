#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E-COMP 체크포인트 — COMP arm vs {floor, prov} 짝 비교 (설계 §3 판정 ①·§3c p4 축).

산출: (a) reward/db 기준 pass^1..4 (공식 pass_hat_k·공통 태스크 짝) (b) robust(4/4) 교차·flip 목록
     (c) prov/floor-pass → COMP-fail 태스크(Δspurious 정독 후보) (d) 요약 판정 신호.
usage: ecomp_checkpoint.py --arm <gz> [--name COMP]
"""
import argparse, gzip, json, math
from collections import Counter

SIM = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results/"


def per_task(path, basis):
    d = json.load(gzip.open(SIM + path))
    pt = {}
    for s in d["simulations"]:
        ri = s.get("reward_info") or {}
        if basis == "db":
            dc = ri.get("db_check")
            ok = bool(dc.get("db_match")) if isinstance(dc, dict) else None
        else:
            r = ri.get("reward")
            ok = (r >= 1) if r is not None else None
        if ok is None:
            continue
        pt.setdefault(str(s["task_id"]), []).append(1 if ok else 0)
    return {t: rs[:4] for t, rs in pt.items() if len(rs) >= 4}


def p_hat(pt, ts, k):
    tot = n = 0
    for t in ts:
        rs = pt[t]
        c = sum(rs)
        tot += math.comb(c, k) / math.comb(len(rs), k)
        n += 1
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--name", default="COMP")
    a = ap.parse_args()

    for basis in ("reward", "db"):
        arm = per_task(a.arm, basis)
        fl = per_task("fl32b_floor_retail_t4.results.json.gz", basis)
        pv = per_task("prov_e2e_retail_t4.results.json.gz", basis)
        common = set(arm) & set(fl) & set(pv)
        ts = sorted(common, key=int)
        print("\n===== basis=%s · common=%d =====" % (basis, len(ts)))
        for nm, pt in (("floor", fl), ("prov", pv), (a.name, arm)):
            print("%-6s " % nm + " ".join("p%d=%.4f" % (k, p_hat(pt, ts, k)) for k in (1, 2, 3, 4)))
        # robust 교차 (arm vs 각 ref)
        for refname, ref in (("floor", fl), ("prov", pv)):
            lost = [t for t in ts if sum(ref[t]) == 4 and sum(arm[t]) < 4]
            gained = [t for t in ts if sum(ref[t]) < 4 and sum(arm[t]) == 4]
            print("  vs %-5s robust: lost=%d %s gained=%d %s" % (
                refname, len(lost),
                [(t, sum(arm[t])) for t in sorted(lost, key=int)][:12],
                len(gained), sorted(gained, key=int)[:12]))
        # Δspurious 후보: ref가 4/4·arm 0/4 (체계 파손 의심)
        for refname, ref in (("floor", fl), ("prov", pv)):
            broke = [t for t in ts if sum(ref[t]) == 4 and sum(arm[t]) == 0]
            print("  vs %-5s systematic-break(4/4->0/4): %s" % (refname, sorted(broke, key=int)))
        # flaky 구조
        wins = Counter(sum(arm[t]) for t in ts)
        print("  %s wins hist(0..4)=%s flaky=%d rpass=%d" % (
            a.name, [wins.get(i, 0) for i in range(5)],
            sum(1 for t in ts if 0 < sum(arm[t]) < 4), wins.get(4, 0)))


if __name__ == "__main__":
    main()
