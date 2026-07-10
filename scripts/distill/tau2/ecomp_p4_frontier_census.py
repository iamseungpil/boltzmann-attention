#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""p4/robust 축 3-arm 전수 교차 — floor vs prov vs gpt-4.1(공식 nt=4) (E-COMP §3c 후속·무료).

기준 = **DB-only**(db_check.db_match·C19/C21: 채점기준 공통분모·C57: 각자-gold라 per-task 대조는
구조 비교로 한정·t18/t91/t107 caveat).

산출:
  1) arm별 db-기준 p1/p4·flaky 구조(0<wins<4 태스크 수·robust-pass·robust-fail)
  2) prov가 잃은 robust 태스크에서 gpt-4.1의 패턴
  3) flaky 태스크 겹침(floor∩gpt41 = user-sim 공통 노이즈 vs arm-특이)
  4) p1→p4 감쇠 분해
"""
import gzip, json, math, sys
from collections import Counter

SIM = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results/"
FR = "/home/woori/workspace_common/boltzmann-attention/external/tau2-bench/data/tau2/results/final/"


def db_ok(s):
    ri = s.get("reward_info") or {}
    dc = ri.get("db_check")
    if isinstance(dc, dict):
        return bool(dc.get("db_match"))
    return None


def per_task(sims):
    pt = {}
    for s in sims:
        ok = db_ok(s)
        if ok is None:
            continue
        pt.setdefault(str(s["task_id"]), []).append(1 if ok else 0)
    return {t: rs[:4] for t, rs in pt.items() if len(rs) >= 4}


def p_hat(pt, k):
    tot = n = 0
    for rs in pt.values():
        c = sum(rs)
        tot += math.comb(c, k) / math.comb(len(rs), k)
        n += 1
    return tot / max(n, 1)


arms = {
    "floor": per_task(json.load(gzip.open(SIM + "fl32b_floor_retail_t4.results.json.gz"))["simulations"]),
    "prov": per_task(json.load(gzip.open(SIM + "prov_e2e_retail_t4.results.json.gz"))["simulations"]),
    "gpt41": per_task(json.load(open(FR + "gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json"))["simulations"]),
}
common = set(arms["floor"]) & set(arms["prov"]) & set(arms["gpt41"])
print("common tasks:", len(common))

print("\n== 1) db-기준 p1/p4 + flaky 구조 (공통 태스크) ==")
for name, pt in arms.items():
    sub = {t: pt[t] for t in common}
    wins = Counter(sum(rs) for rs in sub.values())
    flaky = sum(1 for rs in sub.values() if 0 < sum(rs) < 4)
    print("%-6s p1=%.4f p4=%.4f decay=%.1fpp | wins hist(0..4)=%s flaky=%d rpass=%d rfail=%d"
          % (name, p_hat(sub, 1), p_hat(sub, 4), 100 * (p_hat(sub, 1) - p_hat(sub, 4)),
             [wins.get(i, 0) for i in range(5)], flaky, wins.get(4, 0), wins.get(0, 0)))

print("\n== 2) prov가 잃은 robust 태스크(floor 4/4 & prov <4)의 3-arm 패턴 ==")
lost = sorted((t for t in common if sum(arms["floor"][t]) == 4 and sum(arms["prov"][t]) < 4), key=int)
for t in lost:
    print("  t%-4s floor=4/4 prov=%d/4 gpt41=%d/4" % (t, sum(arms["prov"][t]), sum(arms["gpt41"][t])))
g41_robust_on_lost = sum(1 for t in lost if sum(arms["gpt41"][t]) == 4)
print("  => lost %d개 중 gpt-4.1이 4/4인 것: %d" % (len(lost), g41_robust_on_lost))

print("\n== 3) flaky 겹침 (user-sim 공통 노이즈 vs arm-특이) ==")
fl_flaky = {t for t in common if 0 < sum(arms["floor"][t]) < 4}
pv_flaky = {t for t in common if 0 < sum(arms["prov"][t]) < 4}
g4_flaky = {t for t in common if 0 < sum(arms["gpt41"][t]) < 4}
print("  flaky: floor=%d prov=%d gpt41=%d" % (len(fl_flaky), len(pv_flaky), len(g4_flaky)))
print("  floor∩gpt41=%d floor∩prov=%d prov-only(신규 flaky)=%d"
      % (len(fl_flaky & g4_flaky), len(fl_flaky & pv_flaky), len(pv_flaky - fl_flaky - g4_flaky)))
print("  prov 신규-flaky 태스크:", sorted(pv_flaky - fl_flaky, key=int))

print("\n== 4) gpt-4.1이 robust-pass인데 우리 floor가 flaky/실패인 태스크 수 ==")
g4r = {t for t in common if sum(arms["gpt41"][t]) == 4}
print("  gpt41 4/4=%d | 그중 floor 4/4=%d floor flaky=%d floor 0/4=%d"
      % (len(g4r), sum(1 for t in g4r if sum(arms["floor"][t]) == 4),
         sum(1 for t in g4r if 0 < sum(arms["floor"][t]) < 4),
         sum(1 for t in g4r if sum(arms["floor"][t]) == 0)))
