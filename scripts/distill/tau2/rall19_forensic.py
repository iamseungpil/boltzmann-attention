"""rall19 treat-arm per-sim forensic (2026-07-23).

Usage: python rall19_forensic.py <results.json> <tasks.json> [task_filter...]
Prints per completed sim: reward/termination/db_check/action_checks, ordered
tool-call walk with env errors, last exchanges; plus gold write actions per task.
"""
import json
import sys


def args_of(tc):
    a = tc.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            pass
    return a


def short(o, n=150):
    s = json.dumps(o, ensure_ascii=False, default=str)
    return s[:n]


def main():
    res_path, tasks_path = sys.argv[1], sys.argv[2]
    flt = sys.argv[3:] or None
    data = json.load(open(res_path))
    tasks = json.load(open(tasks_path))
    gold = {}
    for t in tasks:
        tid = str(t.get("id"))
        acts = ((t.get("evaluation_criteria") or {}).get("actions")) or []
        gold[tid] = [(a.get("name"), a.get("arguments"), a.get("requestor"))
                     for a in acts]

    sims = data.get("simulations") or []
    seen_tasks = set()
    for s in sims:
        tid = str(s.get("task_id"))
        if flt and not any(f in tid for f in flt):
            continue
        seen_tasks.add(tid)
        ri = s.get("reward_info") or {}
        print("=" * 100)
        print("SIM %s trial=%s reward=%s term=%s" % (
            tid, s.get("trial"), ri.get("reward"),
            s.get("termination_reason")))
        db = ri.get("db_check") or {}
        print("  db_match=%s" % db.get("db_match"))
        for ac in (ri.get("action_checks") or []):
            a = ac.get("action") or {}
            print("  action_check met=%s req=%s %s(%s)" % (
                ac.get("action_match"), a.get("requestor"), a.get("name"),
                short(a.get("arguments"), 110)))
        msgs = s.get("messages") or []
        print("  -- walk (%d msgs) --" % len(msgs))
        for i, m in enumerate(msgs):
            r = m.get("role")
            for tc in (m.get("tool_calls") or []):
                print("  [%02d]%s> %s %s" % (
                    i, (r or "?")[:5], tc.get("name"), short(args_of(tc), 130)))
            c = m.get("content")
            if r == "tool" and c and ("Error" in str(c) or "error" in str(c)[:60]):
                print("  [%02d]tool! %s" % (i, str(c)[:130].replace("\n", " ")))
            if r == "user" and c:
                print("  [%02d]user: %s" % (i, str(c)[:110].replace("\n", " ")))
            if r == "assistant" and c and not m.get("tool_calls"):
                print("  [%02d]agent: %s" % (i, str(c)[:110].replace("\n", " ")))
    print("=" * 100)
    for tid in sorted(seen_tasks):
        print("GOLD %s:" % tid)
        for name, a, req in gold.get(tid, []):
            print("  %s %s(%s)" % (req, name, short(a, 130)))


if __name__ == "__main__":
    main()
