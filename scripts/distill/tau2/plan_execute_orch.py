#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""plan_execute_orch.py — Phase-1 C1: plan/execute 분리 + 결정론 controller (gpt-4.1 0).

Phase-0(plan_probe) 판정: 부하-유발 실패는 planning 능력이 아니라 plan+execute 동시수행 부하(H1).
이 하네스는 그 가설을 검정한다: **격리 plan-spec를 결정론 controller로 walk**하면 구조 실패
(BATCH_SPLIT·wrong-action-by-status·fabricated-order)가 회복되나?

controller = 도메인-일반 IR([[05]] 준수). retail 특이 지식은 전부 ACTION_SPEC(ABox)에서만 온다:
  - batch-merge   : 같은 (intent-class, order) 다중 콜 → 1콜 병합(union item_ids)   [BATCH_SPLIT 고침]
  - status-fix    : action의 applies-status ≠ 실제 주문 status면 동일 intent-class의
                    올바른 action으로 remap (pending↔delivered)                    [wrong-action 고침]
  - provenance    : order_id ∉ 유저 주문 = 날조 → drop                             [fabrication 고침]
  - 남기는 것(결정론 불가): ⋈ wrong-VALID-order(scale)·over-reach(LLM-scope)·operand

두 모드:
  (a) --replay <plans.json>  : 기존 plan_probe 산출 plan을 controller에 통과(신규 추론 0·완전 무료·오프라인).
  (b) live(기본)             : k-trial planning(32B) → controller → gold 대조 robust. (32B 필요·유료런과 경합 회피)

사용:
  python plan_execute_orch.py --replay plans_phase0.json          # 오프라인 controller 검증
  python plan_execute_orch.py --tasks 20,36,37,99,109,111,71 --k 4  # live robust (PERSISTED 후)
"""
import json, argparse, sys
from collections import defaultdict

# plan_probe(같은 디렉터리)는 모듈 로드 시 원격 db.json을 읽으므로 **지연 임포트**한다
# (controller 순수로직은 오프라인·db 없이 단위테스트 가능해야 함).
_PP = None
def pp():
    global _PP
    if _PP is None:
        import plan_probe as _m
        _PP = _m
    return _PP

# ── ACTION_SPEC = ABox(retail 인스턴스·엔진 아님·[[05]]) ────────────────────────
# intent-class × order-status → concrete action name.  controller 로직은 이걸 읽을 뿐 하드코딩 0.
ACTION_SPEC = {
    # action_name: (intent_class, applies_status, batchable)
    "modify_pending_order_items":    ("item_change", "pending",   True),
    "exchange_delivered_order_items":("item_change", "delivered", True),
    "return_delivered_order_items":  ("item_return", "delivered", True),
    "modify_pending_order_address":  ("address",     "pending",   False),
    "cancel_pending_order":          ("cancel",      "pending",   False),
    "modify_pending_order_payment":  ("payment",     "pending",   False),
}
# intent_class × status → action (status-fix 조회표·ABox 파생)
_CLASS_STATUS_TO_ACTION = {}
for _a, (_c, _s, _b) in ACTION_SPEC.items():
    _CLASS_STATUS_TO_ACTION[(_c, _s)] = _a


def user_order_ids(task):
    """gold가 건드리는 주문의 소유 유저(들)의 전 주문 id 집합 = provenance 화이트리스트."""
    P = pp(); uids, valid = set(), set()
    for a in (task.get("evaluation_criteria", {}) or {}).get("actions", []):
        oid = (a.get("arguments") or {}).get("order_id")
        if oid in P.orders:
            uids.add(P.orders[oid].get("user_id"))
    for oid, o in P.orders.items():
        if o.get("user_id") in uids:
            valid.add(P.norm_oid(oid))
    return valid


def order_status(oid):
    P = pp()
    for k, v in P.orders.items():
        if P.norm_oid(k) == P.norm_oid(oid):
            return v.get("status")
    return None


def controller(plan, valid_oids, status_fn=order_status):
    """결정론 walk. 반환 (normalized_plan, fixlog).
    plan = [(action, oid, frozenset(items))].  ACTION_SPEC(ABox)만 참조·retail 하드코딩 0.
    status_fn = oid→status (기본=db 조회·테스트 시 stub 주입)."""
    fix = {"batch_merge": 0, "status_fix": 0, "provenance_drop": 0}
    # 1) provenance: 날조 주문 drop
    kept = []
    for (n, oid, items) in plan:
        if valid_oids and oid not in valid_oids:
            fix["provenance_drop"] += 1
            continue
        kept.append((n, oid, items))
    # 2) status-fix: action-class × 실제 status → 올바른 action
    fixed = []
    for (n, oid, items) in kept:
        spec = ACTION_SPEC.get(n)
        if spec:
            cls, applies, _ = spec
            st = status_fn(oid)
            if st and st != applies:
                target = _CLASS_STATUS_TO_ACTION.get((cls, st))
                if target and target != n:
                    n = target; fix["status_fix"] += 1
        fixed.append((n, oid, items))
    # 3) batch-merge: 같은 (batchable action, oid) 다중 → 1콜(union)
    groups = defaultdict(list)
    order_seen = []
    for (n, oid, items) in fixed:
        key = (n, oid)
        if key not in groups:
            order_seen.append(key)
        groups[key].append(items)
    normalized = []
    for key in order_seen:
        n, oid = key
        calls = groups[key]
        batchable = ACTION_SPEC.get(n, (None, None, False))[2]
        if batchable and len(calls) > 1:
            union = frozenset().union(*calls)
            fix["batch_merge"] += (len(calls) - 1)
            normalized.append((n, oid, union))
        else:
            for items in calls:
                normalized.append((n, oid, items))
    return normalized, fix


def score(task, plan):
    """plan → gold 대조. core_ok(필수구조·batching) / over_reach(EXTRA on valid). plan_probe.grade 재사용."""
    gold = pp().gold_structure(task)
    core_ok, has_extra, label, issues = pp().grade(gold, plan)
    return core_ok, has_extra, label, issues


def run_offline(replay_path, tids):
    """--replay: 기존 plan(신규 추론 0)을 controller에 통과 → pre/post core_ok."""
    plans = json.load(open(replay_path, encoding="utf-8"))  # {tid: [[action,oid,[items]], ...]}
    rows = []
    for tid in tids:
        task = pp().TASKS.get(tid)
        if not task or tid not in plans:
            print(f"### t{tid}: SKIP (no task/plan)"); continue
        plan = [(n, pp().norm_oid(o), frozenset(it)) for (n, o, it) in plans[tid]]
        valid = user_order_ids(task)
        pre_ok, pre_x, pre_l, _ = score(task, plan)
        norm, fix = controller(plan, valid)
        post_ok, post_x, post_l, post_iss = score(task, norm)
        rows.append((tid, pre_ok, post_ok, pre_x, post_x, fix, pre_l, post_l))
        print(f"\n### t{tid}: pre[{pre_l}] → post[{post_l}]  fixes={ {k:v for k,v in fix.items() if v} }")
        if post_iss:
            print(f"  잔여: {post_iss}")
    return rows


def run_live(tids, k, base, model):
    """live k-trial: planning(32B) → controller → gold. robust core_ok(all-k)."""
    rows = []
    for tid in tids:
        task = pp().TASKS.get(tid)
        if not task:
            print(f"### t{tid}: NO TASK"); continue
        reason = str(task.get("user_scenario", {}).get("instructions", {}).get("reason_for_call", ""))
        ctx = pp().user_orders_context(task); valid = user_order_ids(task)
        prompt = _plan_prompt(reason, ctx)
        pre_oks, post_oks, fixes = [], [], defaultdict(int)
        for _ in range(k):
            try:
                out = pp().ask(prompt, model, base)
            except Exception as e:
                print(f"### t{tid}: ASK_ERR {type(e).__name__}"); pre_oks.append(False); post_oks.append(False); continue
            plan = pp().parse_plan(out)
            if plan is None:
                pre_oks.append(False); post_oks.append(False); continue
            pre_ok, _, _, _ = score(task, plan)
            norm, fix = controller(plan, valid)
            post_ok, _, _, _ = score(task, norm)
            pre_oks.append(pre_ok); post_oks.append(post_ok)
            for kk, vv in fix.items(): fixes[kk] += vv
        rob_pre = all(pre_oks) and len(pre_oks) == k
        rob_post = all(post_oks) and len(post_oks) == k
        rows.append((tid, rob_pre, rob_post, sum(pre_oks), sum(post_oks), dict(fixes)))
        print(f"### t{tid}: pre core_ok {sum(pre_oks)}/{k} (robust={rob_pre}) → post {sum(post_oks)}/{k} (robust={rob_post})  fixes={dict(fixes)}")
    return rows


def _plan_prompt(reason, ctx):
    return (
        "You are PLANNING (not executing) a retail customer-service request. "
        "Do NOT call tools or pick specific replacement variants — only lay out the WRITE actions you WOULD take.\n\n"
        f"{pp().TOOLDOC}\n\nCustomer's request:\n{reason[:800]}\n\n"
        f"Customer's orders (already looked up for you):\n{ctx}\n\n"
        "Output ONLY a JSON array. One element per tool call you would make, in order. "
        'Each element: {"action": "<write action name>", "order_id": "<#W...>", "items": ["<item_id>", ...], "why": "<short>"}. '
        "For replacement variants use \"items\" = the item_ids being changed (NOT the new variant). "
        "Group correctly: all item changes for one order go in ONE element. Omit actions that are not possible/needed."
    )


def summarize(rows, live):
    print("\n\n=== SUMMARY C1 (plan/execute 분리 + 결정론 controller·gpt-4.1 0) ===")
    n = len(rows)
    pre = sum(1 for r in rows if r[1]); post = sum(1 for r in rows if r[2])
    tot = defaultdict(int)
    for r in rows:
        fx = r[5]
        for k, v in fx.items(): tot[k] += v
    print(f"  core_ok (robust): pre {pre}/{n} → post {post}/{n}   (Δ = controller가 회복한 구조)")
    print(f"  결정론 controller 발화(전 과제 합): {dict(tot)}")
    print("  해석: post >> pre 면 = 부하-유발 구조실패를 결정론 walk가 회복(H1 지지·C1 승리·learn NO-GO).")
    print("        post 잔여 실패 = ⋈ wrong-VALID-order(scale)·over-reach(LLM-scope)·operand — 결정론 불가.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="20,36,37,99,109,111,71,17,92,105")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--replay", default=None, help="기존 plans.json(오프라인·추론 0)")
    ap.add_argument("--agent_base", default="http://localhost:8360/v1")
    ap.add_argument("--agent_model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    a = ap.parse_args()
    tids = [t.strip() for t in a.tasks.split(",") if t.strip()]
    if a.replay:
        rows = run_offline(a.replay, tids)
        rows = [(r[0], r[1], r[2], None, None, r[5]) for r in rows]  # summarize 포맷 정규화
    else:
        rows = run_live(tids, a.k, a.agent_base, a.agent_model)
    summarize(rows, live=not a.replay)


if __name__ == "__main__":
    main()
