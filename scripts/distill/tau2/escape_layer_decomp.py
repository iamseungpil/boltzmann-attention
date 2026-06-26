#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S1b: 층화 분해 (ESCAPE_SCOPE_DIAGNOSTIC_LAYERS_AUG §1·§4).
gold write-actions ↔ 궤적 write-calls 정렬(폴백: gold action-index 우선·동률시 best-field-overlap)
→ first-divergence 층(L0 operator / L1 order / L2 item / L3 variant·attr / OVER 여분) 판정.
순수 기계적(술어 불요) → 15 gap의 *층 분포* = "어느 층 지배" 신호(재설계 트리거 §6.5).
σ 분류(ⓐ/ⓑ)는 별도(escape_scope_diag.py·층별 술어 큐레이션 후).
"""
import json, os, argparse
from escape_scope_diag import load_json, compute_gap, first_sim, DOM

# 각 write tool의 층별 인자
LAYER_FIELDS = {
    "modify_pending_order_address":  {"L1": ["order_id"], "L3": ["address1","address2","city","state","country","zip"]},
    "modify_pending_order_items":    {"L1": ["order_id"], "L2": ["item_ids"], "L3": ["new_item_ids"], "pay": ["payment_method_id"]},
    "modify_pending_order_payment":  {"L1": ["order_id"], "pay": ["payment_method_id"]},
    "cancel_pending_order":          {"L1": ["order_id"]},
    "return_delivered_order_items":  {"L1": ["order_id"], "L2": ["item_ids"], "pay": ["payment_method_id"]},
    "exchange_delivered_order_items":{"L1": ["order_id"], "L2": ["item_ids"], "L3": ["new_item_ids"], "pay": ["payment_method_id"]},
}
WRITES = set(LAYER_FIELDS)

def gold_writes(task):
    out = []
    for a in (task.get("evaluation_criteria", {}) or {}).get("actions", []):
        if a.get("name") in WRITES:
            out.append((a["name"], a.get("arguments") or {}))
    return out

def traj_writes(sim):
    out = []
    for m in sim.get("messages", []):
        for tc in (m.get("tool_calls") or []):
            if tc.get("name") in WRITES:
                out.append((tc.get("name"), tc.get("arguments") or {}))
    return out

def _norm(v):
    if isinstance(v, list): return sorted(str(x) for x in v)
    return str(v)

def field_overlap(ga, ta):
    """두 write call 인자의 일치 필드 수(정렬 점수)."""
    s = 0
    for k in set(ga) | set(ta):
        if k in ga and k in ta and _norm(ga[k]) == _norm(ta[k]): s += 1
    return s

def first_divergence(gname, gargs, tname, targs):
    """설계 §1: L0→L1→L2→L3→pay 순 first-divergence."""
    if gname != tname:
        return "L0", f"operator {tname}≠{gname}"
    fields = LAYER_FIELDS[gname]
    for layer in ["L1", "L2", "L3", "pay"]:
        for f in fields.get(layer, []):
            if _norm(gargs.get(f)) != _norm(targs.get(f)):
                lab = layer if layer != "pay" else "L3"  # payment=L3-attr 취급
                return lab, f"{f}: {_norm(targs.get(f))}≠{_norm(gargs.get(f))}"
    return "MATCH", "write 일치(실패=communicate/하류 비-write)"

def align_and_layer(golds, trajs):
    """gold action-index 우선·동률시 best-overlap. 미매칭 gold=MISS·미매칭 traj=OVER."""
    used = set(); rows = []
    for gi, (gn, ga) in enumerate(golds):
        # 후보: 같은 tool 우선, 그 안에서 best overlap
        best, bj = None, -1
        for tj, (tn, ta) in enumerate(trajs):
            if tj in used: continue
            score = (2 if tn == gn else 0) + field_overlap(ga, ta) + (1 if _norm(ga.get("order_id")) == _norm(ta.get("order_id")) else 0)
            if score > bj: bj, best = score, tj
        if best is None:
            rows.append((gi, gn, "MISS", "대응 traj-write 없음(상류 실패로 write 미도달)")); continue
        used.add(best); tn, ta = trajs[best]
        layer, why = first_divergence(gn, ga, tn, ta)
        rows.append((gi, gn, layer, why))
    over = [trajs[j] for j in range(len(trajs)) if j not in used]
    return rows, over

LEVER = {  # 층 → 레버 (§3.5)
    "L0": "결정론(eligibility/status gate)", "OVER": "결정론(stop/commit gate)",
    "L1": "σ분류(ⓐ-ask/ⓑ)", "L2": "σ분류(item·ⓐ/ⓑ)", "L3": "σ분류(variant·ⓐ-ask/resolve/ⓑ)",
    "MISS": "상류(또 다른 층)", "MATCH": "비-write(communicate/operand-equal)",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b32", default="on_n32int8_floor_retail")
    ap.add_argument("--task", default=None)
    args = ap.parse_args()
    tasks = {str(t["id"]): t for t in load_json(os.path.join(DOM, "tasks.json"))}
    gap = compute_gap(args.b32)
    print(f"# S1b 층화 분해 (first-divergence) — gap {len(gap)} task\n")
    dist = {}
    targets = [args.task] if args.task else gap
    for tid in targets:
        t = tasks[tid]; sim = first_sim(args.b32, tid)
        golds, trajs = gold_writes(t), traj_writes(sim)
        rows, over = align_and_layer(golds, trajs)
        layers = [r[2] for r in rows] + (["OVER"] if over else [])
        prim = next((l for l in ["L0","L1","L2","L3","MISS"] if l in layers), ("OVER" if over else "MATCH"))
        dist[prim] = dist.get(prim, 0) + 1
        print(f"[task {tid}] gold={len(golds)}w traj={len(trajs)}w  primary={prim} ({LEVER[prim]})")
        for gi, gn, layer, why in rows:
            print(f"    g{gi} {gn:30s} → {layer:5s} {why}")
        if over:
            for tn, ta in over:
                print(f"    OVER {tn:30s} → 여분 write(order={ta.get('order_id')}) = stop/commit gate")
    print("\n# primary-layer 분포(정성·n=15·비율 아님):", dict(sorted(dist.items(), key=lambda x:-x[1])))
    print("# 레버 함의: L0/OVER=결정론게이트 · L1/L2/L3=σ분류(ⓐ-ask 학습 vs ⓑ formalize vs resolve) · MISS=상류")

if __name__ == "__main__":
    main()
