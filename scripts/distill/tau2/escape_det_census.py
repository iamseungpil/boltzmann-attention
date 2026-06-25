#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
escape_det_census.py — [[08]] 결정론 포렌식 회수 (집계 직행 금지).

핸드오프 2026-06-25 PM §2.1: present+g15(precondition)이 operator/over-action 잔여를
*닫나*를, **결정론 행동지표**(pass^1=user-sim 노이즈 폐기)로 floor/g15/present/present+g15
*누적* 비교한다.

방법(전 trial·고정 task셋·순수 기계적):
  각 sim마다 gold write-actions ↔ 궤적 write-calls 정렬(escape_layer_decomp.align_and_layer)
  → 각 gold write의 first-divergence 층:
       MISS  write 미도달(상류 실패)
       L0    operator 틀림
       L1    order_id(order-pick) 틀림
       L2    item 틀림
       L3    variant/attr/payment 틀림
       MATCH write 완전 일치
  + OVER  여분 write(over-action)
누적(분모=task셋 전 sim의 gold-write 합):
  operator_correct  = 1 - (MISS+L0)/N      (올바른 operator 도달)
  orderpick_correct = (L2+L3+MATCH)/N       (order_id 맞음·이후서 갈림 or 일치)
  write_match       = MATCH/N               (write 완전 정답)
over-action(분모=sim 수): over_rate = #sim(≥1 여분)/#sim
pass^k(보조·노이즈): per-trial reward≥0.999. pass1=평균·pass_all·pass_any.

사용:
  python escape_det_census.py --tasks all \
    --dirs 32B/floor:on_n32int8_floor_retail,32B/g15:on_n32int8_g15_retail_t3,\
32B/present:on_n32int8_presentread_retail_t3,32B/present+g15:on_n32int8_presentg15_retail_t3
"""
import os, argparse
from collections import defaultdict
from escape_scope_diag import load_json, per_task_pass, compute_gap, SIM, DOM
from escape_layer_decomp import gold_writes, traj_writes, align_and_layer, WRITES

LAYERS = ["MISS", "L0", "L1", "L2", "L3", "MATCH"]


def all_sims(sim_dir):
    r = load_json(os.path.join(SIM, sim_dir, "results.json"))
    return r["simulations"]


def census(sim_dir, taskset):
    tasks = {str(t["id"]): t for t in load_json(os.path.join(DOM, "tasks.json"))}
    lc = defaultdict(int)          # layer -> gold-write 수
    n_gold = 0                     # gold-write 합
    n_sims = 0                     # 대상 sim 수
    over_writes = 0                # 여분 write 총수
    sims_with_over = 0
    for s in all_sims(sim_dir):
        tid = str(s["task_id"])
        if tid not in taskset:
            continue
        t = tasks.get(tid)
        if t is None:
            continue
        golds = gold_writes(t)
        if not golds:
            continue
        n_sims += 1
        trajs = traj_writes(s)
        rows, over = align_and_layer(golds, trajs)
        for _gi, _gn, layer, _why in rows:
            lc[layer] += 1
            n_gold += 1
        if over:
            sims_with_over += 1
            over_writes += len(over)
    # pass^k (보조)
    bp = per_task_pass(sim_dir)
    bp = {tid: v for tid, v in bp.items() if tid in taskset}
    flat = [x for v in bp.values() for x in v]
    pass1 = sum(flat) / len(flat) if flat else 0.0
    pass_all = sum(1 for v in bp.values() if v and all(v)) / len(bp) if bp else 0.0
    pass_any = sum(1 for v in bp.values() if any(v)) / len(bp) if bp else 0.0
    N = max(n_gold, 1)
    return {
        "n_sims": n_sims, "n_gold": n_gold,
        "MISS": lc["MISS"], "L0": lc["L0"], "L1": lc["L1"],
        "L2": lc["L2"], "L3": lc["L3"], "MATCH": lc["MATCH"],
        "operator_correct": 1 - (lc["MISS"] + lc["L0"]) / N,
        "orderpick_correct": (lc["L2"] + lc["L3"] + lc["MATCH"]) / N,
        "write_match": lc["MATCH"] / N,
        "over_rate": sims_with_over / max(n_sims, 1),
        "over_writes": over_writes,
        "pass1": pass1, "pass_all": pass_all, "pass_any": pass_any, "n_tasks": len(bp),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True, help="label:dir,label:dir,...")
    ap.add_argument("--tasks", default="all", choices=["all", "gap"])
    ap.add_argument("--floor", default="on_n32int8_floor_retail", help="gap anchor (--tasks gap)")
    args = ap.parse_args()

    if args.tasks == "gap":
        taskset = set(compute_gap(args.floor))
    else:
        tasks = load_json(os.path.join(DOM, "tasks.json"))
        taskset = set(str(t["id"]) for t in tasks
                      if any((a.get("name") in WRITES)
                             for a in (t.get("evaluation_criteria", {}) or {}).get("actions", [])))
    pairs = [p.split(":", 1) for p in args.dirs.split(",")]

    print(f"# escape_det_census — taskset={args.tasks} (|tasks|={len(taskset)}·gold-write 보유)")
    print(f"# 결정론 지표(전 trial). over=여분write·MISS=write미도달·L0=operator·L1=orderpick·L2=item·L3=variant·MATCH=완전\n")
    hdr = ("label", "opCorr", "ordPick", "wMatch", "over%", "|L0", "|L1", "|L2", "|L3", "MISS", "MATCH", "pass1", "passAll")
    print("{:<16}{:>8}{:>8}{:>8}{:>7}{:>5}{:>5}{:>5}{:>5}{:>6}{:>7}{:>7}{:>8}".format(*hdr))
    rows = []
    for label, d in pairs:
        try:
            c = census(d, taskset)
        except Exception as e:
            print(f"{label:<16} ERROR {e}")
            continue
        rows.append((label, c))
        print("{:<16}{:>8.3f}{:>8.3f}{:>8.3f}{:>6.1f}%{:>5}{:>5}{:>5}{:>5}{:>6}{:>7}{:>7.3f}{:>8.3f}".format(
            label, c["operator_correct"], c["orderpick_correct"], c["write_match"], 100 * c["over_rate"],
            c["L0"], c["L1"], c["L2"], c["L3"], c["MISS"], c["MATCH"], c["pass1"], c["pass_all"]))

    # 누적 delta vs 첫 항(floor)
    if len(rows) >= 2:
        base = rows[0][1]
        print(f"\n# Δ vs {rows[0][0]} (결정론 행동지표 변화)")
        for label, c in rows[1:]:
            print("  {:<14} ΔopCorr={:+.3f} ΔordPick={:+.3f} ΔwMatch={:+.3f} Δover={:+.1f}pp ΔL0={:+d} Δpass1={:+.3f}".format(
                label, c["operator_correct"] - base["operator_correct"],
                c["orderpick_correct"] - base["orderpick_correct"],
                c["write_match"] - base["write_match"],
                100 * (c["over_rate"] - base["over_rate"]),
                c["L0"] - base["L0"], c["pass1"] - base["pass1"]))


if __name__ == "__main__":
    main()
