# -*- coding: utf-8 -*-
"""x782 — 반증 프로브: gold 정의 전수에서 «닫힘/열림» 판정의 재료를 직접 센다.
읽기 전용. sim_results/*.results.json.gz 의 tasks[] 만 본다(궤적 무관).
"""
import gzip, glob, json, os, sys, collections

ROOT = os.path.dirname(os.path.abspath(__file__))
TASKS = {}
for p in sorted(glob.glob(os.path.join(ROOT, "sim_results", "*.results.json.gz"))):
    try:
        d = json.load(gzip.open(p, "rt", encoding="utf-8"))
    except Exception:
        continue
    for t in (d.get("tasks") or []):
        tid = t.get("id")
        if tid and tid not in TASKS:
            TASKS[tid] = t

print("TASKS collected:", len(TASKS))

def acts(t):
    ec = t.get("evaluation_criteria") or {}
    return ec.get("actions") or [], (ec.get("reward_basis") or [])

def unwrap(a):
    """디스패처 래퍼(call_discoverable_agent_tool) 언랩 — 실인자는 중첩 JSON."""
    name = a.get("name")
    args = a.get("arguments") or {}
    inner = args.get("agent_tool_name")
    if inner:
        raw = args.get("arguments")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = {}
        if isinstance(raw, dict):
            return inner, raw
        return inner, {k: v for k, v in args.items() if k not in ("agent_tool_name", "arguments")}
    return name, args

# ---------- A. customer_max_liability_amount 전수 ----------
print("\n=== A. gold customer_max_liability_amount 전수 ===")
vals = collections.Counter()
same = []
rows = []
for tid in sorted(TASKS):
    A, rb = acts(TASKS[tid])
    for a in A:
        n, ar = unwrap(a)
        if "customer_max_liability_amount" in ar:
            v = ar["customer_max_liability_amount"]
            da = ar.get("disputed_amount")
            vals[repr(v)] += 1
            rows.append((tid, a.get("action_id"), n, v, da))
            if v == da:
                same.append((tid, a.get("action_id"), v, da))
print("count rows:", len(rows))
for r in rows:
    print("   ", r)
print("value census:", dict(vals))
print("SAME(max_liab == disputed):", same, " => %d/%d" % (len(same), len(rows)))

# ---------- B. account_class gold 전수 ----------
print("\n=== B. gold account_class 전수 ===")
ac = collections.Counter()
acrows = []
for tid in sorted(TASKS):
    A, rb = acts(TASKS[tid])
    for a in A:
        n, ar = unwrap(a)
        for k in ("account_class", "card_type", "referred_account_type"):
            if k in ar:
                ac[(k, repr(ar[k]))] += 1
                acrows.append((tid, a.get("action_id"), k, ar[k]))
for k, c in sorted(ac.items()):
    print("   ", k, c)

# ---------- C. 표적 태스크 gold 전량 ----------
print("\n=== C. 표적 태스크 gold 전량 ===")
for tid in ["task_026", "task_084", "task_086", "task_088", "task_091", "task_069",
            "task_080", "task_078", "task_101", "task_055", "task_056", "task_066",
            "task_071", "task_007", "task_061", "task_060", "task_097", "task_077"]:
    t = TASKS.get(tid)
    if not t:
        print(tid, "MISSING"); continue
    A, rb = acts(t)
    print("\n-- %s reward_basis=%s actions=%d" % (tid, rb, len(A)))
    for a in A:
        n, ar = unwrap(a)
        print("   [%s] %s %s" % (a.get("action_id"), n, json.dumps(ar, ensure_ascii=False)[:400]))
