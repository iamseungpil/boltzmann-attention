#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x790 — 한 sim 의 **turn 단위 궤적 덤프** (2026-09-05, task_027 포렌식용).

사용:  python x790_027_traj.py <tag> <simid> [maxchars]
출력:  [i] ROLE | 요약 (assistant 는 tool_calls 를 name(args) 로, tool 은 결과 앞머리)
⛔ 판정하지 않는다. 찍기만 한다.
"""
import io, json, sys
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

SIMROOT = "/home/woori/scratch/tau2-bench/data/simulations"
tag, simid = sys.argv[1], sys.argv[2]
MAX = int(sys.argv[3]) if len(sys.argv) > 3 else 700

d = json.load(open("%s/%s/results.json" % (SIMROOT, tag)))
sim = next(s for s in d["simulations"] if s["id"] == simid)
print("== %s %s task=%s reward=%s term=%s nmsg=%d" % (
    tag, simid, sim.get("task_id"),
    (sim.get("reward_info") or {}).get("reward"), sim.get("termination_reason"),
    len(sim["messages"])))

for i, m in enumerate(sim["messages"]):
    role = m.get("role")
    turn = m.get("turn_idx")
    content = (m.get("content") or "")
    if isinstance(content, str):
        content = content.replace("\n", " \\n ")
    parts = []
    for tc in (m.get("tool_calls") or []):
        parts.append("%s(%s)" % (tc.get("name"), json.dumps(tc.get("arguments"), ensure_ascii=False)))
    if role == "tool":
        head = "[id=%s] %s" % (m.get("id"), content)
    elif parts:
        head = ("TXT<%s> " % content[:180]) + " ;; ".join(parts)
    else:
        head = content
    print("[%03d] t=%s %s | %s" % (i, turn, role.upper(), head[:MAX]))
