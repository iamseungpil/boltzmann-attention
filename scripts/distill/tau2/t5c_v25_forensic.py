#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T5-C V2.5 표적 6 전수 궤적 포렌식 — 원인별 부작용-없는 해소 판정 (무료).

각 태스크: gold write · v25b 전 turn(user/assistant/tool·write 강조) · 레버발화 마커 ·
floor/COMP 동일태스크 write 대조. [[08]] per-case 정독.
usage: t5c_v25_forensic.py --arm t5c_v25b [--tasks 0,17,40,47,61,95]
"""
import argparse, gzip, json, sys

SIM = "/home/woori/scratch/tau2-bench/data/simulations/"
PERS = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results/"
TASKS = "/home/woori/scratch/tau2-bench/data/tau2/domains/retail/tasks.json"
WR = ("modify", "exchange", "return", "cancel")


def args_of(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {}
    return a if isinstance(a, dict) else {}


def load_dir(tag):
    return json.load(open(SIM + tag + "/results.json"))["simulations"]


def load_gz(tag):
    return json.load(gzip.open(PERS + tag + ".results.json.gz"))["simulations"]


def gold_writes(task):
    return [(x.get("name"), args_of(x.get("arguments")))
            for x in ((task.get("evaluation_criteria") or {}).get("actions") or [])
            if x.get("requestor", "assistant") == "assistant" and any(w in (x.get("name") or "") for w in WR)]


def writes_of(sim):
    msgs = sim.get("messages") or []
    res = {m.get("id"): m for m in msgs if m.get("role") == "tool"}
    out = []
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            nm = tc.get("name") or ""
            if any(w in nm for w in WR):
                tm = res.get(tc.get("id"))
                err = (tm or {}).get("error") if tm else None
                out.append((nm, args_of(tc.get("arguments")), "ERR" if err else "ok"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="t5c_v25b")
    ap.add_argument("--tasks", default="0,17,40,47,61,95")
    a = ap.parse_args()
    tasks = {str(t["id"]): t for t in json.load(open(TASKS))}
    arm = {str(s["task_id"]): s for s in load_dir(a.arm)}
    floor = {str(s["task_id"]): s for s in load_gz("fl32b_floor_retail_t4")}
    comp = {str(s["task_id"]): s for s in load_gz("comp_retail_t4")}
    tids = a.tasks.split(",")

    for tid in tids:
        s = arm.get(tid)
        if s is None:
            print("== t%s MISSING ==" % tid); continue
        t = tasks[tid]
        ri = s.get("reward_info") or {}
        dc = ri.get("db_check")
        db = dc.get("db_match") if isinstance(dc, dict) else None
        print("\n" + "=" * 78)
        print("t%s  reward=%s db_match=%s term=%s" % (tid, ri.get("reward"), db, s.get("termination_reason")))
        print("GOLD writes:")
        for nm, ar in gold_writes(t):
            print("   %s %s" % (nm, json.dumps(ar, ensure_ascii=False)[:180]))
        print("v25b(COMP+D-v2) writes:")
        for nm, ar, st in writes_of(s):
            print("   [%s] %s %s" % (st, nm, json.dumps(ar, ensure_ascii=False)[:170]))
        # floor/comp(t0..3 aggregate: 몇 trial write 있었나·pass)
        for nm2, src in (("floor", floor), ("COMP", comp)):
            grp = [x for x in src.values() if False]  # placeholder
        for label, src in (("floor", floor), ("COMP", comp)):
            xs = [x for x in (load_gz("fl32b_floor_retail_t4") if label == "floor" else load_gz("comp_retail_t4"))
                  if str(x["task_id"]) == tid]
            summ = []
            for x in xs:
                r = (x.get("reward_info") or {}).get("reward")
                nw = len(writes_of(x))
                summ.append("t%s:r%s/w%d" % (x.get("trial"), r, nw))
            print("  %s(nt4): %s" % (label, " ".join(summ)))
        # 레버 발화 (이 sim 안의 assistant content에 개입 흔적? — 없음이 정상[replay-clean]) + 대화 마지막 2 assistant
        ass = [m.get("content") for m in (s.get("messages") or [])
               if m.get("role") == "assistant" and isinstance(m.get("content"), str) and m.get("content").strip()]
        print("  last assistant: %s" % (ass[-1][:160].replace("\n", " ") if ass else "(none)"))
        # user 발화 중 핵심 사실(payment/gift/id 언급)
        us = [m.get("content") for m in (s.get("messages") or [])
              if m.get("role") == "user" and isinstance(m.get("content"), str)]
        keyu = [u[:150].replace("\n", " ") for u in us if any(w in u.lower()
                for w in ("gift", "paypal", "credit", "refund", "address", "#w"))]
        for u in keyu[:2]:
            print("  user-fact: %s" % u)


if __name__ == "__main__":
    main()
