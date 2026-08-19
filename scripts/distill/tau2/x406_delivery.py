# -*- coding: utf-8 -*-
r"""x406 — 무언급 표적이 **궤적에 배달됐나**: 도구 결과(KB 검색·shell 출력) 본문 축자 검사.

NEVER_MENTIONED 65건마다, 그 도구 이름이
  ⓐ 그 sim 의 **도구 결과 본문**(role=tool)에 등장했나  ← KB 가 배달한 것
  ⓑ 그 sim 의 **손님 발화**에 등장했나
  ⓒ 배달 시점이 마지막 assistant 턴보다 앞인가 (=볼 수 있었나)
"""
import collections, io, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import t2_forensic as F
import x396_saying_vs_doing as C

rows = []
for tag in C.TAGS:
    for sim in F.scored(tag, C.SUF):
        if ((sim.get("reward_info") or {}).get("reward") or 0) >= 1.0:
            continue
        msgs = sim.get("messages") or []
        body = " ".join(" ".join(str(m.get("content") or "").split())
                        for m in msgs if m.get("role") == "assistant" and m.get("content"))
        toolbody, userbody = [], []
        for i, m in enumerate(msgs):
            c = " ".join(str(m.get("content") or "").split())
            if m.get("role") == "tool":
                toolbody.append((i, c))
            elif m.get("role") == "user":
                userbody.append((i, c))
        last_a = max([i for i, m in enumerate(msgs) if m.get("role") == "assistant"] or [0])
        calls = C.called(sim)
        for g in C.gold_rows(sim):
            if g["match"] or calls.get(g["name"]):
                continue
            nm = g["name"]
            ops = C.operand_tokens(g["args"])
            if nm in body or (ops and any(o in body for o in ops)):
                continue
            dt = [i for i, c in toolbody if nm in c]
            du = [i for i, c in userbody if nm in c]
            rows.append({"task": F.task_id(sim), "trial": sim.get("trial"), "name": nm,
                         "in_tool": bool(dt), "in_user": bool(du),
                         "first_at": (min(dt + du) if (dt or du) else None), "last_a": last_a})

print("=" * 100); print("x406 · NEVER_MENTIONED %d건 — 그 도구 이름이 궤적에 배달됐나" % len(rows))
print("=" * 100)
c = collections.Counter()
for r in rows:
    c["도구결과(KB/shell)에 등장" if r["in_tool"] else ("손님 발화에만" if r["in_user"] else "궤적 어디에도 없음")] += 1
for k, v in c.most_common():
    print("  %-24s %2d  (%.0f%%)" % (k, v, 100.0 * v / len(rows)))
d = [r for r in rows if r["first_at"] is not None]
print("\n  배달된 것 중 마지막 assistant 턴보다 앞서 온 것: %d/%d" %
      (sum(1 for r in d if r["first_at"] < r["last_a"]), len(d)))

print("\n## 태스크 × 배달")
x = collections.defaultdict(collections.Counter)
for r in rows:
    x[r["task"]][("DELIVERED" if r["in_tool"] else ("USER" if r["in_user"] else "ABSENT"))] += 1
for t in sorted(x, key=lambda z: -sum(x[z].values())):
    print("  %-9s %s" % (t, dict(x[t])))

print("\n## 도구 × 배달")
y = collections.defaultdict(collections.Counter)
for r in rows:
    y[r["name"]][("DELIVERED" if r["in_tool"] else ("USER" if r["in_user"] else "ABSENT"))] += 1
for t in sorted(y, key=lambda z: -sum(y[z].values())):
    print("  %-46s %s" % (t, dict(y[t])))
json.dump(rows, io.open(os.path.join("..", "..", "..", "reports", "facet_rft_2026",
                                     "x406_delivery.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
