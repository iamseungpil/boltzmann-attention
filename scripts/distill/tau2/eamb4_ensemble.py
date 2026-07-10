#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E-AMB-4 (T4) - 해독기-앙상블 불일치 = H(gold|X) 대리량 검정.

정본: reports/facet_rft_2026/E_AMB_MEASUREMENT_PLAN_2026_07_10.md §4
해독기 = frontier baseline 4종(raw 보존·gpt-4.1 sim) + 우리 arm들(gz).
슬롯 = (task, gold write action, 인자). 각 해독기의 답 = trial 다수결 값.
불일치 H = 해독기 답 분포의 섀넌 엔트로피.

P4a: frontier-공통-실패 슬롯이 고-H에 집중하는가.
반증조건: 공통-실패가 저-H(전원이 '같은 오답')에 집중 → 공유-prior 섹터(미결정 아님).
둘 다 계수한다: scatter(미결정형) vs same-wrong(공유-prior형).

Run (remote): python3 eamb4_ensemble.py
"""
import glob
import gzip
import json
import math
import os
import re
from collections import Counter, defaultdict

REPO = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results"
FRONTIER_DIR = "/home/woori/scratch/tau2-bench/data/tau2/results/final"

OURS = {
    "ours_fl32b": REPO + "/fl32b_floor_retail_t4.results.json.gz",
    "ours_fl14b": REPO + "/fl14b_floor_retail_t4.results.json.gz",
    "ours_qwq32b": REPO + "/qwq32b_floor_retail_t4.results.json.gz",
    "ours_asm32b": REPO + "/asmregen32b_regen_retail_t4.results.json.gz",
    "ours_prov32b": REPO + "/prov_e2e_retail_t4.results.json.gz",
}
WRITE = {"return_delivered_order_items", "exchange_delivered_order_items", "cancel_pending_order",
         "modify_pending_order_items", "modify_pending_order_address", "modify_pending_order_payment",
         "modify_user_address", "place_order"}
ARGS = ("new_item_ids", "item_ids", "payment_method_id", "order_id", "address1")


def norm(x):
    return re.sub(r"\s+", " ", str(x).lower().replace("#", "")).strip()


def as_key(v):
    vals = v if isinstance(v, list) else [v]
    return "|".join(sorted(norm(x) for x in vals if x is not None))


def load(path):
    if path.endswith(".gz"):
        return json.load(gzip.open(path, "rt", encoding="utf-8"))
    return json.load(open(path, encoding="utf-8"))


def gold_slots(sims):
    """(task_id) -> [(tool, gold_oid, argkey, gold_value_key)]"""
    slots = {}
    for sim in sims:
        tid = str(sim.get("task_id"))
        if tid in slots:
            continue
        out = []
        for a in (sim.get("reward_info") or {}).get("action_checks") or []:
            act = a.get("action") or {}
            if act.get("name") not in WRITE:
                continue
            args = act.get("arguments") or {}
            oid = norm(args.get("order_id") or "")
            for k in ARGS:
                if k in args and args.get(k) not in (None, [], ""):
                    out.append((act.get("name"), oid, k, as_key(args.get(k))))
        slots[tid] = out
    return slots


def decoder_answers(sims):
    """(task, tool, gold_oid, argkey) -> 이 해독기의 trial별 값 -> 다수결.
    매칭: 같은 tool 호출 중 order_id 일치 우선, 없으면 마지막 호출(최종 결정)."""
    per_trial = defaultdict(dict)
    for sim in sims:
        tid = str(sim.get("task_id"))
        trial = sim.get("trial", 0)
        calls = []
        for m in sim.get("messages", []):
            if m.get("role") != "assistant":
                continue
            for tc in (m.get("tool_calls") or []):
                if tc.get("name") in WRITE:
                    calls.append(tc)
        gold = (sim.get("reward_info") or {}).get("action_checks") or []
        for a in gold:
            act = a.get("action") or {}
            if act.get("name") not in WRITE:
                continue
            gargs = act.get("arguments") or {}
            goid = norm(gargs.get("order_id") or "")
            cand = [c for c in calls if c.get("name") == act.get("name")]
            pick = None
            for c in cand:
                if norm((c.get("arguments") or {}).get("order_id") or "") == goid:
                    pick = c
            if pick is None and cand:
                pick = cand[-1]
            for k in ARGS:
                if k not in gargs or gargs.get(k) in (None, [], ""):
                    continue
                slot = (tid, act.get("name"), goid, k)
                v = (pick.get("arguments") or {}).get(k) if pick else None
                per_trial[slot][trial] = as_key(v) if v not in (None, [], "") else "<NO-WRITE>"
    ans = {}
    for slot, tv in per_trial.items():
        cnt = Counter(tv.values())
        ans[slot] = cnt.most_common(1)[0][0]
    return ans


def main():
    frontier = {}
    for f in glob.glob(FRONTIER_DIR + "/*_retail_default_*4trials.json"):
        name = "fr_" + os.path.basename(f).split("_retail_")[0]
        frontier[name] = load(f)["simulations"]
    ours = {k: load(p)["simulations"] for k, p in OURS.items() if os.path.exists(p)}
    print("frontier decoders:", sorted(frontier), "\nours:", sorted(ours))

    ref = next(iter(ours.values()))
    slots_by_task = gold_slots(ref)
    all_slots = [(t,) + s for t, ss in slots_by_task.items() for s in ss]
    print("gold slots:", len(all_slots))

    answers = {}
    for name, sims in list(frontier.items()) + list(ours.items()):
        answers[name] = decoder_answers(sims)

    fr_names = sorted(frontier)
    all_names = fr_names + sorted(ours)
    rows = []
    for (tid, tool, goid, k, gkey) in all_slots:
        slot = (tid, tool, goid, k)
        vals = {n: answers[n].get(slot, "<ABSENT>") for n in all_names}
        present = {n: v for n, v in vals.items() if v not in ("<ABSENT>",)}
        if len(present) < 6:
            continue
        cnt = Counter(present.values())
        N = sum(cnt.values())
        H = -sum(c / N * math.log2(c / N) for c in cnt.values())
        fr_vals = [vals[n] for n in fr_names if vals[n] != "<ABSENT>"]
        fr_wrong = [v for v in fr_vals if v != gkey and v != "<NO-WRITE>"]
        fr_common_fail = len(fr_vals) >= 3 and all(v != gkey for v in fr_vals)
        same_wrong = fr_common_fail and len(set(fr_wrong)) == 1 and fr_wrong
        rows.append({"task": tid, "tool": tool, "arg": k, "gold": gkey, "H": round(H, 3),
                     "n_dec": N, "fr_common_fail": fr_common_fail,
                     "same_wrong": bool(same_wrong), "vals": vals})

    print("\n=== 슬롯 %d개 (해독기>=6 응답) ===" % len(rows))
    med = sorted(r["H"] for r in rows)[len(rows) // 2]
    hi = [r for r in rows if r["H"] > med]
    lo = [r for r in rows if r["H"] <= med]
    for lbl, grp in (("고-H(>%.2f)" % med, hi), ("저-H", lo)):
        cf = sum(1 for r in grp if r["fr_common_fail"])
        print("  %-12s n=%4d  frontier-공통실패 %d (%.1f%%)" % (lbl, len(grp), cf, 100 * cf / max(len(grp), 1)))
    cf_rows = [r for r in rows if r["fr_common_fail"]]
    sw = sum(1 for r in cf_rows if r["same_wrong"])
    print("  공통실패 %d개 중 same-wrong(공유-prior형) %d · scatter(미결정형) %d" % (len(cf_rows), sw, len(cf_rows) - sw))
    print("  공통실패 H 평균 %.2f vs 나머지 %.2f" % (
        sum(r["H"] for r in cf_rows) / max(len(cf_rows), 1),
        sum(r["H"] for r in rows if not r["fr_common_fail"]) / max(len(rows) - len(cf_rows), 1)))

    print("\n=== frontier-공통실패 슬롯 전수 (per-case 정독 대상) ===")
    for r in sorted(cf_rows, key=lambda x: -x["H"]):
        print(" t%s %s.%s gold=%s H=%.2f %s" % (r["task"], r["tool"], r["arg"], r["gold"][:34], r["H"],
              "SAME-WRONG" if r["same_wrong"] else "scatter"))
        for n, v in r["vals"].items():
            if v != gkey:
                print("    %-16s -> %s" % (n, str(v)[:60]))

    out = REPO + "/eamb4_ensemble_rows.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
