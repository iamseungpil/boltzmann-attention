#!/usr/bin/env python3
"""E10 §5 격리검증 — 정책-precondition 게이트가 DB-결정론으로 성립하는가 (GPU-free).

정본 doc: reports/facet_rft_2026/E10_PRECONDITION_GATE_DESIGN_2026_07_09.md §5
결과(2026-07-09·arm ours=asmregen32b_regen_retail_t4·retail/db.json): NO-GO.
  P1(refund-target): TP(위반&db_fail)=5(전부 t99) < over-block FP(위반&db_pass)=6, 비판별.
  P2(status-eligibility): 실행 write 중 ineligible 0/602 = 환경이 이미 집행(C12·redundant).
  per-case: t10/t12 환불 PM이 주문 pm_orig와 일치·t57/t32/t99 status-eligible
            → over-action 불가능성은 DB-state 아니라 대화(철회·부분의도)=semantic([[06]] 게이트금지).

사용: python3 e10_precond_probe.py [--stage cases|scan]
"""
import argparse, json, gzip

SIM = ("/home/woori/workspace_common/boltzmann-attention-pi/reports/"
       "facet_rft_2026/sim_results/asmregen32b_regen_retail_t4.results.json.gz")
DB = "/home/woori/scratch/tau2-bench/data/tau2/domains/retail/db.json"

WRITE = {"return_delivered_order_items", "exchange_delivered_order_items", "cancel_pending_order",
         "modify_pending_order_items", "modify_pending_order_address",
         "modify_pending_order_payment", "modify_user_address", "place_order"}
REF = {"return_delivered_order_items", "exchange_delivered_order_items"}
STATUS_REQ = {"return_delivered_order_items": "delivered", "exchange_delivered_order_items": "delivered",
              "cancel_pending_order": "pending", "modify_pending_order_items": "pending",
              "modify_pending_order_address": "pending", "modify_pending_order_payment": "pending"}


def _load():
    db = json.load(open(DB))
    sims = json.load(gzip.open(SIM))["simulations"]
    return db["orders"], db["users"], sims


def ri(s):
    return s.get("reward_info") or {}


def db_match(s):
    return (ri(s).get("db_check") or {}).get("db_match")


def tid(s):
    return str(s.get("task_id"))


def gold_writes(s):
    return [(a.get("action", {}).get("name"), a.get("action", {}).get("arguments") or {})
            for a in (ri(s).get("action_checks") or []) if a.get("action", {}).get("name") in WRITE]


def exec_writes(s):
    by = {m.get("id") or m.get("tool_call_id"): m for m in s.get("messages", []) if m.get("role") == "tool"}
    out = []
    for m in s.get("messages", []):
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            if tc.get("name") not in WRITE:
                continue
            tm = by.get(tc.get("id"))
            if tm is not None and not tm.get("error"):
                out.append((tc["name"], tc.get("arguments") or {}))
    return out


def pm_orig(orders, oid):
    o = orders.get(oid) or {}
    return {p.get("payment_method_id") for p in (o.get("payment_history") or [])
            if p.get("transaction_type") == "payment"}


def gcs(users, uid):
    u = users.get(uid) or {}
    return {k for k, v in (u.get("payment_methods") or {}).items()
            if isinstance(v, dict) and v.get("source") == "gift_card"}


def stage_cases():
    orders, users, sims = _load()
    target = {"10", "12", "32", "57", "99"}
    rw = {"return_delivered_order_items", "exchange_delivered_order_items", "cancel_pending_order"}
    for s in sims:
        if tid(s) not in target:
            continue
        ew = exec_writes(s)
        rel = [(n, a) for n, a in ew if n in rw]
        if not rel:
            continue
        uid = next((a.get("user_id") for _, a in ew if a.get("user_id")), None)
        print(f"\n### t{tid(s)} tr{s.get('trial')} db_match={db_match(s)} "
              f"gold={len(gold_writes(s))} exec={len(ew)}")
        for n, a in rel:
            oid = a.get("order_id")
            print(f"   {n} order={oid} pm={a.get('payment_method_id')} | "
                  f"status={(orders.get(oid) or {}).get('status')} pm_orig={sorted(pm_orig(orders, oid))}")
        if uid:
            print(f"   user={uid} giftcards={sorted(gcs(users, uid))}")


def stage_scan():
    orders, users, sims = _load()
    tp = fp = tot = 0
    for s in sims:
        uid = next((a.get("user_id") for _, a in exec_writes(s) if a.get("user_id")), None)
        g = gcs(users, uid) if uid else set()
        for n, a in exec_writes(s):
            if n in REF and a.get("payment_method_id") and a.get("order_id"):
                tot += 1
                if a["payment_method_id"] not in (pm_orig(orders, a["order_id"]) | g):
                    if db_match(s) is False:
                        tp += 1
                    else:
                        fp += 1
    print(f"P1(refund-target) 전수 refund={tot} · TP(위반&db_fail)={tp} · over-block FP(위반&db_pass)={fp}")
    inelig = tot_st = 0
    for s in sims:
        for n, a in exec_writes(s):
            if n in STATUS_REQ and a.get("order_id"):
                tot_st += 1
                st = (orders.get(a["order_id"]) or {}).get("status")
                if st and STATUS_REQ[n] not in st:
                    inelig += 1
    print(f"P2(status) 실행 write 중 ineligible={inelig}/{tot_st} (0=환경이 이미 집행·C12 redundant)")
    print(f"VERDICT: TP({tp}) {'<' if tp < fp else '>='} FP({fp}) · P2 redundant={inelig==0} "
          f"=> {'NO-GO' if (tp <= fp or inelig == 0) else 'check'}")


STAGES = {"cases": stage_cases, "scan": stage_scan}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="scan", choices=sorted(STAGES))
    STAGES[ap.parse_args().stage]()
