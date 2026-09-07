#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""x772 — 판정·읽기 5레버(T2_DIAG_UNAMBIGUOUS / T2_READ_PER_ENTITY / T2_VERDICT_GATE /
T2_CLAIM_VERIFY / T2_SCHEMA_ENUM)의 **발화 실측**.

캠페인 정본 집합을 로컬 회수분에서 재구성한다 — 태스크당 최신 sim(2026-09-03T14 ~ 09-05T03).
읽기만 한다. 코드·A2·go_stack 수정 0.
"""
import gzip, json, os, sys, glob, datetime, collections

SIMDIR = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"
LO = "2026-09-03T14"
HI = "2026-09-05T03"

def load(fp):
    with gzip.open(fp, "rt", encoding="utf-8") as fh:
        return json.load(fh)

def main():
    best = {}   # task_id -> (ts, file, sim)
    for fp in glob.glob(os.path.join(SIMDIR, "bank_*.results.json.gz")):
        try:
            d = load(fp)
        except Exception:
            continue
        for s in d.get("simulations", []):
            ts = s.get("timestamp") or s.get("end_time") or ""
            if not (LO <= ts <= HI + "\uffff"):
                continue
            tid = s.get("task_id")
            if tid is None:
                continue
            cur = best.get(tid)
            if cur is None or ts > cur[0]:
                best[tid] = (ts, os.path.basename(fp), s)
    rows = []
    for tid, (ts, fn, s) in sorted(best.items()):
        ri = s.get("reward_info") or {}
        rw = ri.get("reward")
        rows.append((tid, ts, fn, s.get("id"), rw))
    npass = sum(1 for r in rows if (r[4] or 0) >= 1.0)
    print("tasks=%d pass=%d fail=%d" % (len(rows), npass, len(rows) - npass))
    out = os.path.join(SIMDIR, "..", "x772_campaign_set.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump([{"task": r[0], "ts": r[1], "file": r[2], "sim": r[3], "reward": r[4]} for r in rows],
                  fh, ensure_ascii=False, indent=1)
    print("wrote", os.path.abspath(out))
    fails = [r[0] for r in rows if (r[4] or 0) < 1.0]
    print("FAIL(%d): %s" % (len(fails), " ".join(x.replace("task_", "") for x in fails)))

main()
