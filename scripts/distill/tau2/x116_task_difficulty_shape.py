# -*- coding: utf-8 -*-
"""뒤 구간이 무너지는 것은 **회귀인가 난이도인가** — 태스크 번호 대 요구 행동 수.

x115 실측(2026-08-06): pass^1이 1~32에서 31.2%, 33~64에서 5.9%, 65~102에서 1.3%다.
이 기울기의 해석은 둘로 갈린다 — ⓐ우리 스택이 뒤 태스크에서 무너진다 ⓑ뒤 태스크가 원래 더 크다.
둘을 가르는 가장 싼 계기는 **태스크 정의 자체**다: gold 액션 수·요구 문서 수·손님 도구 수를
번호 구간별로 센다. 정의는 우리 실행과 무관하므로 이 수치는 스택과 독립이다.

  usage:  x116_task_difficulty_shape.py
"""

import collections
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x109_task_dossier import load_sims, load_tasks          # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BUCKETS = [(1, 32), (33, 64), (65, 102)]


def num(t):
    m = re.search(r"(\d+)", t or "")
    return int(m.group(1)) if m else -1


def main():
    tasks = load_tasks()
    sims = load_sims()
    passed = collections.defaultdict(list)
    for s in sims:
        passed[s["task_id"]].append((s.get("reward_info") or {}).get("reward") == 1.0)

    rows = []
    for tid, (_f, t) in tasks.items():
        ec = t.get("evaluation_criteria") or {}
        acts = ec.get("actions") or []
        rows.append({
            "id": tid, "n": num(tid),
            "actions": len(acts),
            "user_actions": sum(1 for a in acts if a.get("requestor") == "user"),
            "docs": len(t.get("required_documents") or []),
            "utools": len(t.get("user_tools") or []),
            "basis": ",".join(ec.get("reward_basis") or []),
            "pass": (sum(passed[tid]) / float(len(passed[tid]))) if passed.get(tid) else None,
        })
    rows = [r for r in rows if r["n"] > 0]
    print("== 태스크 정의의 크기 (우리 실행과 무관) ==")
    print("  %-10s %5s %8s %8s %7s %7s %8s" % ("구간", "태스크", "gold평균", "최대", "손님행동", "요구문서", "pass^1"))
    for lo, hi in BUCKETS:
        sub = [r for r in rows if lo <= r["n"] <= hi]
        if not sub:
            continue
        seen = [r for r in sub if r["pass"] is not None]
        p1 = (sum(r["pass"] for r in seen) / len(seen)) if seen else float("nan")
        print("  %3d~%-6d %5d %8.1f %8d %7.1f %7.1f %7.1f%%"
              % (lo, hi, len(sub),
                 sum(r["actions"] for r in sub) / float(len(sub)),
                 max(r["actions"] for r in sub),
                 sum(r["user_actions"] for r in sub) / float(len(sub)),
                 sum(r["docs"] for r in sub) / float(len(sub)),
                 100.0 * p1))

    print("\n== gold 액션 수별 통과율 (난이도-통과 관계) ==")
    bins = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 99)]
    for lo, hi in bins:
        sub = [r for r in rows if lo <= r["actions"] <= hi and r["pass"] is not None]
        if not sub:
            continue
        p1 = sum(r["pass"] for r in sub) / len(sub)
        print("  gold %2d~%-3d 태스크 %2d종 · pass^1 %5.1f%% · 번호 중앙값 %d"
              % (lo, hi, len(sub), 100 * p1, sorted(r["n"] for r in sub)[len(sub) // 2]))

    print("\n== 채점 기준별 ==")
    for basis in sorted({r["basis"] for r in rows}):
        sub = [r for r in rows if r["basis"] == basis and r["pass"] is not None]
        if not sub:
            continue
        print("  %-10s 태스크 %2d종 · pass^1 %5.1f%%"
              % (basis or "?", len(sub), 100.0 * sum(r["pass"] for r in sub) / len(sub)))

    print("\n== 뒤 구간(65~102) 큰 태스크 상위 ==")
    for r in sorted([r for r in rows if r["n"] >= 65], key=lambda x: -x["actions"])[:12]:
        print("  %-10s gold %2d (손님 %d) · 문서 %d · %s · pass %s"
              % (r["id"], r["actions"], r["user_actions"], r["docs"], r["basis"],
                 "-" if r["pass"] is None else "%.0f%%" % (100 * r["pass"])))


if __name__ == "__main__":
    main()
