# -*- coding: utf-8 -*-
r"""t7333 실패 per-step 포렌식 — **pass 를 올릴 자리가 남아 있나** (2026-08-21·오프라인·LLM 0)

사용자 물음: *"task_001 −1 · 003·047·055·063·070 변화 없음 … 실패 원인을 정밀하게 per step
포렌식하라. pass 를 올릴 방법이 없는지 확인하라."*

[[08]] 대로 집계에서 결론 직행하지 않는다 — **변이 집합**(MISSING·WRONGARG·EXTRA·DUP·BLOCKED)이
reward 의 실패 단위다([[69]]). 각 sim 을 그 단위로 열고, 거절이 있었으면 **누가** 거절했는지까지 본다.
"""
import collections
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F   # noqa: E402  정본(사본 금지·[[67]])

BASE = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")


def load(arm, tag="20260821c", prefix="bank_t7333"):
    out = []
    for part in ("hot", "rest"):
        p = os.path.abspath(os.path.join(BASE, "%s_%s_%s_%s.results.json.gz" % (prefix, arm, part, tag)))
        if os.path.exists(p):
            out.extend(json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or [])
    return out


def brief(items, n=3):
    out = []
    for x in items[:n]:
        a = x.get("args") or {}
        keys = sorted(a)[:3]
        out.append("%s(%s)" % (x.get("name"), ",".join("%s=%s" % (k, str(a[k])[:22]) for k in keys)))
    return " · ".join(out) or "-"


def main():
    tasks = sys.argv[1:] or ["task_001", "task_003", "task_047", "task_055", "task_063", "task_070"]
    arms = {"ctl": load("ctl"), "treat": load("treat")}
    mut = F.mutating_tools()
    tally = collections.Counter()
    for t in tasks:
        print("=" * 96)
        print("### %s" % t)
        for arm in ("ctl", "treat"):
            for s in arms[arm]:
                if str(s.get("task_id") or "") != t:
                    continue
                r = (s.get("reward_info") or {}).get("reward") or 0.0
                d = F.mutation_diff(s, mut)
                term = F.termination(s) if hasattr(F, "termination") else (
                    s.get("termination_reason") or "?")
                cells = []
                for k in ("missing", "wrongarg", "extra", "dup", "blocked"):
                    v = d.get(k) or []
                    if v:
                        cells.append("%s=%d" % (k.upper(), len(v)))
                        if r < 1.0:
                            tally[(t, k)] += len(v)
                print("  %-6s t%-2s reward=%.1f  %-34s term=%s"
                      % (arm, s.get("trial"), r, " ".join(cells) or "(변이 일치)", term))
                if r < 1.0:
                    for k in ("missing", "wrongarg", "extra", "dup"):
                        v = d.get(k) or []
                        if v:
                            print("        %-8s %s" % (k.upper(), brief(v)))
                    for b in (d.get("blocked") or [])[:2]:
                        print("        BLOCKED  %s ← %s"
                              % (b.get("name"), str(b.get("deny") or "")[:110]))
    print("\n" + "=" * 96)
    print("[실패 단위 집계] (reward<1 인 sim 만)")
    for (t, k), n in sorted(tally.items(), key=lambda x: (-x[1], x[0])):
        print("  %-10s %-9s %d" % (t, k.upper(), n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
