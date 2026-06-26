#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""load_obs.py — LOAD THEORY 무료 관측연구 (construct-validity 게이트·gpt-4.1 0·기존 데이터만).

리뷰어 directive: 새 생성·유료 0. 기존 results.json으로
  Study1 (construct validity): fail이 load-feature와 상관하나? (상관 0 → 이론 무근거 → 중단)
  Study2 (약한 ΔL 스크린): floor vs scaffold서 rescued-task의 load-feature → scaffold가 어느 차원 깎았나
  + collinearity 행렬 (리뷰어 예측: 5차원→2-3 붕괴) + L_contra 희소 점검
★주의([[08]]): 관측=약한 스크린·난이도 교락(operand 공변량 통제)·인과 아님. 상관0이면 중단·아니면 통제생성 필요.

사용: python load_obs.py --floor <dir> --scaffold <dir> [--domain retail]
"""
import argparse, json, re, os
import numpy as np
from collections import defaultdict

DOM_BASE = "/home/woori/scratch/tau2-bench/data/simulations"
RETAIL = "/home/woori/scratch/tau2-bench/data/tau2/domains/retail"

WRITES = {"modify_pending_order_items", "exchange_delivered_order_items",
          "return_delivered_order_items", "modify_pending_order_address",
          "cancel_pending_order", "modify_pending_order_payment"}
COND = re.compile(r"\bif\b|otherwise|unless|in case|if not|if possible|if the agent|"
                  r"if that|if it|in which case|else", re.I)
CONTRA = re.compile(r"change my mind|instead of|but when|but if|initially.{0,40}then|"
                    r"decide to|actually,|on second thought|i change|rather than", re.I)


def load_tasks_db():
    tasks = {str(t["id"]): t for t in json.load(open(RETAIL + "/tasks.json"))}
    db = json.load(open(RETAIL + "/db.json"))
    return tasks, db["orders"], db["products"]


def task_features(task, orders, prods):
    """task-intrinsic load 후보 feature (realized 아님·outcome 비결합). + operand 공변량."""
    reason = str(task.get("user_scenario", {}).get("instructions", {}).get("reason_for_call", ""))
    acts = [a for a in (task.get("evaluation_criteria", {}) or {}).get("actions", []) if a.get("name") in WRITES]
    # 관여 user들의 전 주문 (모델이 들고가야 할 맥락)
    uids = set()
    for a in acts:
        oid = (a.get("arguments") or {}).get("order_id")
        if oid in orders:
            uids.add(orders[oid].get("user_id"))
    uords = {oid: o for oid, o in orders.items() if o.get("user_id") in uids}
    n_user_orders = len(uords)
    total_items = sum(len(o.get("items", [])) for o in uords.values())
    # 같은 product type 중복 (간섭)
    type_counts = defaultdict(int)
    for o in uords.values():
        for it in o.get("items", []):
            type_counts[it.get("name")] += 1
    max_same_type = max(type_counts.values()) if type_counts else 0
    # write 슬롯
    n_orders_touched = len(set((a.get("arguments") or {}).get("order_id") for a in acts))
    n_write_items = sum(len((a.get("arguments") or {}).get("item_ids") or []) for a in acts)
    n_new_variants = sum(len((a.get("arguments") or {}).get("new_item_ids") or []) for a in acts)
    return {
        "L_len":    len(reason.split()) + total_items,        # 보존 맥락 proxy
        "L_state":  n_write_items + n_orders_touched,         # 상호의존 슬롯
        "L_branch": len(COND.findall(reason)),                # 조건분기
        "L_interf": n_user_orders + max_same_type,            # 헷갈릴 엔티티
        "L_contra": len(CONTRA.findall(reason)),              # 모순-개정 (희소 예상)
        "operand":  n_new_variants,                           # 공변량 (load 아님·난이도)
        "n_actions": len(acts),
    }


def pass_rate(run_dir):
    """results.json → {task_id: (n_pass, n_trial)} reward==1.0 = pass."""
    f = os.path.join(run_dir, "results.json")
    d = json.load(open(f))
    sims = d["simulations"] if isinstance(d, dict) else d
    agg = defaultdict(lambda: [0, 0])
    for s in sims:
        ri = s.get("reward_info") or {}
        r = ri.get("reward")
        if r is None:
            continue
        tid = str(s.get("task_id"))
        agg[tid][1] += 1
        agg[tid][0] += (1 if r >= 1.0 else 0)
    return {tid: (p, n) for tid, (p, n) in agg.items()}


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def partial(x, y, z):
    """r(x,y | z) — operand 공변량 통제."""
    rxy, rxz, ryz = pearson(x, y), pearson(x, z), pearson(y, z)
    denom = ((1 - rxz**2) * (1 - ryz**2))
    if denom <= 0 or np.isnan(rxy):
        return float("nan")
    return (rxy - rxz * ryz) / np.sqrt(denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", default="on_n32int8_floor_retail_t5")
    ap.add_argument("--scaffold", default="on_n32int8_presentnest_g15_retail_t3")
    a = ap.parse_args()
    tasks, orders, prods = load_tasks_db()
    floor = pass_rate(os.path.join(DOM_BASE, a.floor))
    scaf = pass_rate(os.path.join(DOM_BASE, a.scaffold))
    common = sorted(set(floor) & set(scaf) & set(tasks), key=int)
    FEATS = ["L_len", "L_state", "L_branch", "L_interf", "L_contra"]

    rows = []
    for tid in common:
        ft = task_features(tasks[tid], orders, prods)
        fp, fn = floor[tid]; sp, sn = scaf[tid]
        ft["floor_fail"] = 1 - fp / fn          # fail-rate (0..1)
        ft["scaf_fail"] = 1 - sp / sn
        ft["tid"] = tid
        rows.append(ft)
    print(f"=== LOAD 관측연구 (n={len(rows)} tasks·floor={a.floor}·scaffold={a.scaffold}·gpt-4.1 0) ===\n")

    # collinearity (리뷰어: 5→2-3 붕괴 예측)
    print("## collinearity (feature 간 Pearson·붕괴 점검)")
    mat = {f: [r[f] for r in rows] for f in FEATS}
    print("        " + " ".join(f"{f:>8}" for f in FEATS))
    for f in FEATS:
        print(f"{f:>8} " + " ".join(f"{pearson(mat[f], mat[g]):>8.2f}" for g in FEATS))
    print(f"  L_contra nonzero: {sum(1 for v in mat['L_contra'] if v>0)}/{len(rows)} (희소?)\n")

    # Study 1: fail ↔ load (floor·model-alone) + operand 통제
    print("## Study1 construct-validity: floor fail-rate ↔ load-feature")
    ff = [r["floor_fail"] for r in rows]; op = [r["operand"] for r in rows]
    print(f"  (공변량 operand: r(operand,fail)={pearson(op,ff):+.2f})")
    print(f"  {'feature':>8} {'r_raw':>7} {'r|operand':>10}   fail by tertile(low/mid/high)")
    for f in FEATS:
        xs = [r[f] for r in rows]
        r_raw = pearson(xs, ff); r_par = partial(xs, ff, op)
        order = np.argsort(xs); n = len(order); t = n // 3
        terts = [order[:t], order[t:2*t], order[2*t:]]
        tf = [np.mean([ff[i] for i in g]) if len(g) else float("nan") for g in terts]
        print(f"  {f:>8} {r_raw:>+7.2f} {r_par:>+10.2f}   {tf[0]:.2f}/{tf[1]:.2f}/{tf[2]:.2f}")

    # Study 2: 약한 ΔL — scaffold가 rescue한 task의 feature
    print("\n## Study2 약한 ΔL 스크린: floor-fail 중 scaffold가 rescue(개선)한 task vs 잔존")
    floorfail = [r for r in rows if r["floor_fail"] >= 0.5]
    rescued = [r for r in floorfail if r["scaf_fail"] < r["floor_fail"] - 1e-9]
    stayed = [r for r in floorfail if r["scaf_fail"] >= r["floor_fail"] - 1e-9]
    print(f"  floor-fail(≥.5): {len(floorfail)} | rescued(scaffold↓): {len(rescued)} | stayed: {len(stayed)}")
    print(f"  {'feature':>8} {'rescued_mean':>13} {'stayed_mean':>12}  (큰 차이=scaffold가 그 차원 깎음 시사)")
    for f in FEATS + ["operand"]:
        rm = np.mean([r[f] for r in rescued]) if rescued else float("nan")
        sm = np.mean([r[f] for r in stayed]) if stayed else float("nan")
        print(f"  {f:>8} {rm:>13.2f} {sm:>12.2f}")
    print("\n★[[08]] 해석규율: 관측=약한 스크린(상관0이면 중단). 상관 있어도 난이도 교락→인과는 통제생성(operand고정·한차원) 필요. Study2 ΔL=분포비교일 뿐·비순환 예측검정 아님(L1+ 필요).")


if __name__ == "__main__":
    main()
