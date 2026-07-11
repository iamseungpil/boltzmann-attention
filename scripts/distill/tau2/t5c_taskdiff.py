#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""per-task 압축 write-diff 포렌식 — 실패 trial의 gold vs 실행 write를 필드 단위로 대조 ([[08]]).

태스크마다: gold writes → 실패 trial별 {실행 write(gold와 diff 필드: actual→gold), 에러-시도 수,
종료사유, db_match, 마지막 assistant 발화 스니펫}. 궤적 전문 정독 전 스크리닝용.

usage: t5c_taskdiff.py --results <gz> --tasks <tasks.json> --ids 20,34,36 [--full-turns <tid>]
"""
import argparse, gzip, json
from collections import Counter, defaultdict

WR = ("modify", "exchange", "return", "cancel")


def load_json(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def args_of(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {}
    return a if isinstance(a, dict) else {}


def gold_writes(task):
    return [(x.get("name"), args_of(x.get("arguments")))
            for x in ((task.get("evaluation_criteria") or {}).get("actions") or [])
            if x.get("requestor", "assistant") == "assistant"
            and any(w in (x.get("name") or "") for w in WR)]


def diff_str(actual, gold_cands):
    """actual write args vs 최소-diff gold 후보: 다른 필드만 'k: a -> g'."""
    if not gold_cands:
        return "(no gold for this tool = EXTRA)"
    best = min(gold_cands, key=lambda g: sum(1 for k in set(g) | set(actual)
                                             if str(g.get(k)) != str(actual.get(k))))
    ds = []
    for k in sorted(set(best) | set(actual)):
        av, gv = actual.get(k), best.get(k)
        if str(av) != str(gv):
            ds.append("%s: %s -> gold %s" % (k, av, gv))
    return "; ".join(ds) if ds else "(exact match)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--full-turns", default=None, help="이 tid는 전 turn 출력(정독)")
    ap.add_argument("--all-trials", action="store_true", help="pass trial도 출력(대조)")
    a = ap.parse_args()

    tasks = {str(t["id"]): t for t in load_json(a.tasks)}
    sims = load_json(a.results)["simulations"]
    ids = [x.strip() for x in a.ids.split(",")]
    by_task = defaultdict(list)
    for s in sims:
        by_task[str(s.get("task_id"))].append(s)

    for tid in ids:
        task = tasks.get(tid, {})
        gold = gold_writes(task)
        reason = str((task.get("user_scenario", {}).get("instructions", {}) or {}).get("reason_for_call", ""))[:300]
        print("\n" + "=" * 90)
        print("### t%s  |  intent: %s" % (tid, reason.replace("\n", " ")))
        for nm, ar in gold:
            print("  GOLD %s %s" % (nm, json.dumps(ar, ensure_ascii=False)))
        for s in sorted(by_task.get(tid, []), key=lambda x: x.get("trial") or 0):
            ri = s.get("reward_info") or {}
            r = ri.get("reward")
            if r is not None and r >= 1 and not a.all_trials:
                continue
            db = (ri.get("db_check") or {})
            db_ok = db.get("db_match") if isinstance(db, dict) else None
            msgs = s.get("messages") or []
            res_by_id = {m.get("id"): m for m in msgs if m.get("role") == "tool"}
            execd, errs = [], Counter()
            last_asst = ""
            for m in msgs:
                if m.get("role") == "assistant":
                    if isinstance(m.get("content"), str) and m["content"].strip():
                        last_asst = m["content"].strip()
                    for tc in (m.get("tool_calls") or []):
                        nm, ar = tc.get("name"), args_of(tc.get("arguments"))
                        if any(w in (nm or "") for w in WR):
                            tm = res_by_id.get(tc.get("id"))
                            if tm is not None and not tm.get("error"):
                                execd.append((nm, ar))
                            else:
                                errs[nm] += 1
            print("  -- trial %s: reward=%s db=%s term=%s" % (s.get("trial"), r, db_ok, s.get("termination_reason")))
            gold_by_tool = defaultdict(list)
            for nm, ar in gold:
                gold_by_tool[nm].append(ar)
            for nm, ar in execd:
                print("     WRITE %s | %s" % (nm, diff_str(ar, gold_by_tool.get(nm))))
            missing = {nm: len(c) - sum(1 for w in execd if w[0] == nm)
                       for nm, c in gold_by_tool.items()
                       if sum(1 for w in execd if w[0] == nm) < len(c)}
            if missing:
                print("     MISSING gold writes: %s" % missing)
            if errs:
                print("     errored write attempts: %s" % dict(errs))
            if last_asst:
                print("     last-asst: %s" % last_asst[:220].replace("\n", " "))
            if a.full_turns == tid:
                print("     ---- full turns ----")
                for m in msgs:
                    role, c = m.get("role"), m.get("content")
                    tcs = ["%s(%s)" % (tc.get("name"), json.dumps(args_of(tc.get("arguments")), ensure_ascii=False)[:180])
                           for tc in (m.get("tool_calls") or [])]
                    line = (c or "").strip().replace("\n", " ")[:250] if isinstance(c, str) else ""
                    err = " [ERR]" if (role == "tool" and m.get("error")) else ""
                    print("     [%s]%s %s %s" % (role, err, line, " | ".join(tcs)))


if __name__ == "__main__":
    main()
