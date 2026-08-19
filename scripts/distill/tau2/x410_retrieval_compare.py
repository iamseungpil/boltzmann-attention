# -*- coding: utf-8 -*-
r"""x410 - 회수 3방식 비교 (사용자 지시 2026-08-19): bm25 / dense / shell

x409 가 드러낸 것: R3 의 "배달"은 **28KB BM25 덤프 안에 이름이 한 번 스친 것**이었고
88%가 지시문이 아니라 단순 언급이었다. 그러면 문제는 회수다. 세 방식을 같은 축으로 잰다.

측정 (전부 궤적 축자):
  (1) 방식별 호출 수 · 결과 길이 · 반환 문서 수
  (2) 질의 축자 - 키워드가 무엇이었나
  (3) 표적 도구 이름이 결과에 들어왔나 · 들어왔다면 결과의 몇 % 지점인가(파묻힘 깊이)
  (4) 결과에 **지시문**(give them the / use the / must unlock / in this exact order)이 함께 왔나
"""
import collections
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F
import x396_saying_vs_doing as C

METH = ("KB_search_bm25", "KB_search_dense", "shell")
DIRW = ("give them the", "give the user", "provide access to", "use the", "call the",
        "must unlock", "in this exact order", "do not skip", "before completing")
DOC_RE = re.compile(r"doc_[a-z_()0-9]+|Document \d+|## ")


def main():
    print("=" * 116)
    print("x410 · 회수 3방식 비교")
    print("=" * 116)
    calls = []
    for tag in C.TAGS:
        for sim in F.scored(tag, C.SUF):
            msgs = sim.get("messages") or []
            R = {m["id"]: " ".join(str(m.get("content") or "").split())
                 for m in msgs if m.get("role") == "tool" and m.get("id")}
            rw = ((sim.get("reward_info") or {}).get("reward") or 0)
            for m, tc in F.calls(sim):
                nm = str(F.nameof(tc))
                if nm not in METH:
                    continue
                a = F.argsof(tc)
                q = str(a.get("query") or a.get("command") or a)[:140]
                body = R.get(tc.get("id"), "")
                calls.append({"task": F.task_id(sim), "trial": sim.get("trial"), "rw": rw,
                              "meth": nm, "q": q, "len": len(body),
                              "ndoc": len(DOC_RE.findall(body)),
                              "ndirect": sum(body.lower().count(d) for d in DIRW),
                              "body": body})

    print("\n## (1) 방식별 — 호출 수 · 결과 길이 · 문서 수 · 지시문 밀도")
    print("  %-16s %6s %12s %12s %10s %10s" % ("방식", "호출", "길이중앙", "길이최대", "문서수중앙", "지시문/호출"))
    for m in METH:
        r = [c for c in calls if c["meth"] == m]
        if not r:
            continue
        L = sorted(c["len"] for c in r)
        print("  %-16s %6d %12d %12d %10d %10.1f"
              % (m, len(r), L[len(L) // 2], L[-1],
                 sorted(c["ndoc"] for c in r)[len(r) // 2],
                 sum(c["ndirect"] for c in r) / float(len(r))))

    print("\n## (2) 질의 축자 (방식별 최대 12개)")
    for m in METH:
        r = [c for c in calls if c["meth"] == m]
        print("  ### %s (%d)" % (m, len(r)))
        for q, n in collections.Counter(c["q"] for c in r).most_common(12):
            print("     x%-2d %s" % (n, q[:100]))

    # (3)(4) 표적 파묻힘
    print("\n## (3)(4) 표적 도구가 결과에 들어왔을 때 — 파묻힘 깊이와 지시문 동반")
    R3 = json.load(io.open(os.path.join("..", "..", "..", "reports", "facet_rft_2026",
                                        "x409_r3_perstep.json"), encoding="utf-8"))
    tgt = sorted(set((r["task"], r["trial"], r["tool"]) for r in R3))
    print("  %-9s %-3s %-40s %-16s %8s %8s %s"
          % ("task", "tr", "tool", "방식", "결과길이", "깊이%", "직전160자 지시문"))
    depth = []
    for t in tgt:
        for c in calls:
            if (c["task"], c["trial"]) != (t[0], t[1]):
                continue
            i = c["body"].find(t[2])
            if i < 0:
                continue
            pct = 100.0 * i / max(len(c["body"]), 1)
            near = c["body"][max(0, i - 160):i + 60].lower()
            has = [d for d in DIRW if d in near]
            depth.append(pct)
            print("  %-9s %-3s %-40s %-16s %8d %7.1f%% %s"
                  % (t[0], t[1], t[2][:40], c["meth"], c["len"], pct,
                     ",".join(has) if has else "(없음)"))
            break
    if depth:
        depth.sort()
        print("\n  깊이 중앙 %.1f%% · 사분위 %.1f%% / %.1f%%"
              % (depth[len(depth) // 2], depth[len(depth) // 4], depth[3 * len(depth) // 4]))

    print("\n## (5) 성공 sim vs 실패 sim 의 회수 행동")
    for lab, pred in (("성공(reward=1)", lambda c: c["rw"] >= 1.0), ("실패", lambda c: c["rw"] < 1.0)):
        r = [c for c in calls if pred(c)]
        if not r:
            continue
        d = collections.Counter(c["meth"] for c in r)
        print("  %-14s 호출 %3d  %s  길이중앙 %d"
              % (lab, len(r), dict(d), sorted(c["len"] for c in r)[len(r) // 2]))
    return 0


sys.exit(main())
