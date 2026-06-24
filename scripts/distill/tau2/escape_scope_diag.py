#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Escape-Scope Diagnostic (ESCAPE_SCOPE_DIAGNOSTIC_DESIGN rev2).
make-or-break 첫 측정: 32B 실패가 DB-카디널리티로 표면화되나(ⓐ) vs 침묵(ⓑ).

Arm-I (아키텍처·escape 너비): faithful 술어 → σ(참 DB, 결정점 state) → 궤적 선택 대조 분류.
Arm-II (capability): tie·ⓑ 케이스에 32B select-probe (별도 단계·--probe).

설계 §2 분류(궤적-선택 기반·predicate 재추출 *안 함*):
  |σ|=0          → no-change → ⓐ
  |σ|>1          → tie       → ⓐ
  |σ|=1, traj가 그 유일정답 entity 고름·op/arg 틀림 → ⓑ-act/ⓑ-op
  |σ|=1, traj가 딴 entity 고름                      → ⓑ (mis-ground)
discipline: tau2 학습0·A2 관계(σ)만·도메인분기0·gpt-4.1 불요(Arm-I).
"""
import json, os, argparse, re

TB = "/home/woori/scratch/tau2-bench"
SIM = TB + "/data/simulations"
DOM = TB + "/data/tau2/domains/retail"

# write tools whose key id-arg = the disambiguated entity (order-level)
ORDER_WRITE = {
    "modify_pending_order_address": "order_id",
    "modify_pending_order_items": "order_id",
    "cancel_pending_order": "order_id",
    "return_delivered_order_items": "order_id",
    "exchange_delivered_order_items": "order_id",
    "modify_pending_order_payment": "order_id",
}

def load_json(p): return json.load(open(p, encoding="utf-8"))

def per_task_pass(sim_dir):
    r = load_json(os.path.join(SIM, sim_dir, "results.json"))
    by = {}
    for s in r["simulations"]:
        ri = s.get("reward_info") or {}
        rew = ri.get("reward"); rew = float(rew) if rew is not None else 0.0
        by.setdefault(str(s["task_id"]), []).append(1 if rew >= 0.999 else 0)
    return by

def compute_gap(b32_dir="on_n32int8_floor_retail", gpt_dir="retail_gpt41_nogate"):
    b32 = per_task_pass(b32_dir); bg = per_task_pass(gpt_dir)
    gap = [t for t in b32 if all(v == 0 for v in b32[t]) and bg.get(t) and any(bg[t])]
    return sorted(gap, key=int)

def first_sim(sim_dir, tid):
    r = load_json(os.path.join(SIM, sim_dir, "results.json"))
    for s in r["simulations"]:
        if str(s["task_id"]) == str(tid):
            return s
    return None

def write_calls(sim):
    """모델이 실제 호출한 write tool calls (name, args)."""
    out = []
    for m in sim.get("messages", []):
        for tc in (m.get("tool_calls") or []):
            nm = tc.get("name", "")
            if nm in ORDER_WRITE or nm.startswith(("modify_", "cancel_", "return_", "exchange_")):
                out.append((nm, tc.get("arguments") or {}))
    return out

def resolve_user_orders(db, task, sim):
    """user_id 해석 → 그 user의 orders(enriched)를 candidate collection으로."""
    users, orders = db["users"], db["orders"]
    uid = None
    # 1) 궤적의 find_user 결과 args에서 이름/zip → user 매칭
    name = zipc = None
    for m in sim.get("messages", []):
        for tc in (m.get("tool_calls") or []):
            if tc.get("name", "").startswith("find_user_id"):
                a = tc.get("arguments") or {}
                name = (a.get("first_name"), a.get("last_name")); zipc = a.get("zip")
    if name and name[0]:
        for k, v in users.items():
            nm = v.get("name", {})
            if nm.get("first_name") == name[0] and nm.get("last_name") == name[1]:
                uid = k; break
    if uid is None:  # fallback: known_info 파싱
        ki = task["user_scenario"]["instructions"].get("known_info", "")
        m = re.search(r"name is (\w+)\s+(\w+)", ki or "")
        if m:
            for k, v in users.items():
                nm = v.get("name", {})
                if nm.get("first_name") == m.group(1) and nm.get("last_name") == m.group(2):
                    uid = k; break
    cands = []
    if uid:
        for oid in users[uid].get("orders", []):
            o = orders.get(oid, {})
            ad = o.get("address", {})
            cands.append({
                "order_id": oid, "status": o.get("status"),
                "city": ad.get("city"), "state": ad.get("state"), "zip": ad.get("zip"),
                "n_items": len(o.get("items", [])),
                "products": sorted({it.get("name") for it in o.get("items", [])}),
            })
    return uid, cands

def sigma(cands, flt):
    """A2 σ: candidate dict 집합에 구조적 필터(field==value, 대소문자 무시 부분일치 옵션) 적용."""
    def match(c):
        for k, v in flt.items():
            cv = c.get(k)
            if isinstance(v, dict) and "contains" in v:
                if cv is None or v["contains"].lower() not in str(cv).lower():
                    return False
            else:
                if str(cv).lower() != str(v).lower():
                    return False
        return True
    return [c for c in cands if match(c)]

def gold_order_target(task):
    for a in (task.get("evaluation_criteria", {}) or {}).get("actions", []):
        if a.get("name") in ORDER_WRITE:
            return a["arguments"].get("order_id")
    return None

def classify(sig, gold_tgt, traj_oids):
    """설계 §2: 궤적 선택 vs 참 정답 + |σ|."""
    n = len(sig)
    sig_ids = {c["order_id"] for c in sig}
    picked_gold = gold_tgt in traj_oids
    if gold_tgt not in sig_ids and n > 0:
        return "PRED_ERR", f"gold {gold_tgt} ∉ σ(={sorted(sig_ids)}) — faithful 술어 점검(누락/발명)"
    if n == 0:
        return "ⓐ:no-change", "참 DB에 매치 없음 → 'none→ASK'이어야"
    if n > 1:
        return "ⓐ:tie", f"|σ|={n} 후보 여럿 {sorted(sig_ids)} → '어느 것?→ASK'이어야"
    # n == 1 (유일 정답)
    if picked_gold:
        return "ⓑ-act/op", "정답 entity는 골랐으나 operator/arg 틀림(B2/action·escape 밖)"
    return "ⓑ:mis-ground", f"|σ|=1 유일정답 {gold_tgt} 인데 모델은 {sorted(traj_oids)} 선택 → 침묵 잔여"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predicates", default=os.path.join(os.path.dirname(__file__), "escape_predicates.json"))
    ap.add_argument("--b32", default="on_n32int8_floor_retail")
    ap.add_argument("--task", default=None, help="단일 task만")
    args = ap.parse_args()

    db = load_json(os.path.join(DOM, "db.json"))
    tasks = {str(t["id"]): t for t in load_json(os.path.join(DOM, "tasks.json"))}
    preds = load_json(args.predicates) if os.path.exists(args.predicates) else {}

    gap = compute_gap(args.b32)
    print(f"# Escape-Scope Diagnostic — Stage-1 (정성 카탈로그·비율결론 금지)")
    print(f"# gap = gpt4.1 pass ∧ 32B fail-all-3 = {len(gap)} task: {gap}\n")
    targets = [args.task] if args.task else gap
    cat = {"ⓐ": 0, "ⓑ": 0, "PRED_ERR": 0, "no-pred": 0, "non-order": 0}
    for tid in targets:
        if tid not in tasks:
            continue
        t = tasks[tid]; sim = first_sim(args.b32, tid)
        gold = gold_order_target(t)
        if gold is None:
            print(f"[task {tid}] non-order-disambiguation (gold write=order 아님) → 별도 분류")
            cat["non-order"] += 1; continue
        uid, cands = resolve_user_orders(db, t, sim)
        wc = write_calls(sim)
        traj_oids = {a.get("order_id") for _, a in wc if a.get("order_id")}
        pr = preds.get(tid)
        print(f"[task {tid}] user={uid} gold={gold} traj_pick={sorted(traj_oids) or '∅'} "
              f"cands={len(cands)}")
        for c in cands:
            print(f"    - {c['order_id']} status={c['status']} {c['city']}/{c['state']} items={c['n_items']}")
        if not pr:
            print(f"    → faithful 술어 미작성(S3 큐레이션 필요): predicate 추가 후 분류\n")
            cat["no-pred"] += 1; continue
        sig = sigma(cands, pr["filter"])
        label, why = classify(sig, gold, traj_oids)
        print(f"    faithful σ filter={pr['filter']}  note={pr.get('note','')}")
        print(f"    → {label}  | {why}\n")
        cat[label.split(":")[0].split("-")[0]] = cat.get(label.split(":")[0].split("-")[0], 0) + 1
    print("# 분류 누적(정성·참고용·n작음):", {k: v for k, v in cat.items() if v})

if __name__ == "__main__":
    main()
