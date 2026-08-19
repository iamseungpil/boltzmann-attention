# -*- coding: utf-8 -*-
r"""x407 — 배달 실패 17건: **검색을 했나 / 했는데 그 문서가 안 왔나**

KB 도메인이므로 절차는 문서에 있고 회수로 꺼내야 한다(정책 §Knowledge base search tools).
ABSENT 인 sim 마다: KB_search_bm25 / KB_search_dense / shell 호출 수 · 질의 축자 ·
그 도구 이름을 담은 문서가 결과에 등장했는지.
"""
import collections, io, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import t2_forensic as F
import x396_saying_vs_doing as C

KB = ("KB_search_bm25", "KB_search_dense", "shell")
D = json.load(io.open(os.path.join("..", "..", "..", "reports", "facet_rft_2026",
                                   "x406_delivery.json"), encoding="utf-8"))
absent = collections.defaultdict(set)
for r in D:
    if not r["in_tool"] and not r["in_user"]:
        absent[(r["task"], str(r["trial"]))].add(r["name"])

print("=" * 104); print("x407 · 배달 실패 sim 의 회수 행동"); print("=" * 104)
for tag in C.TAGS:
    for sim in F.scored(tag, C.SUF):
        k = (F.task_id(sim), str(sim.get("trial")))
        if k not in absent:
            continue
        q, nkb = [], collections.Counter()
        for m, tc in F.calls(sim):
            nm = F.nameof(tc)
            if nm in KB:
                nkb[nm] += 1
                a = F.argsof(tc)
                q.append("%s(%s)" % (nm, str(a.get("query") or a.get("command") or a)[:110]))
        print("\n  ★%s t%s  놓친도구=%s" % (k[0], k[1], ", ".join(sorted(absent[k]))))
        print("    회수 호출: %s (총 %d)" % (dict(nkb), sum(nkb.values())))
        for s in q[:14]:
            print("      %s" % s)
        if len(q) > 14:
            print("      … +%d" % (len(q) - 14))
