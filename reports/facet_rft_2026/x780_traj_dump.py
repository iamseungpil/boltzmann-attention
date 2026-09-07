#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x780 - per-step trajectory dump for w-value-select forensics (read-only).
usage: x780_traj_dump.py <results.json.gz> <task_id> [sim_id]
"""
import gzip, json, sys, io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

path, tid = sys.argv[1], sys.argv[2]
simid = sys.argv[3] if len(sys.argv) > 3 else None
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt", encoding="utf-8") as f:
    res = json.load(f)

sims = [s for s in res.get("simulations", []) if s.get("task_id") == tid]
if simid:
    sims = [s for s in sims if s.get("id") == simid] or sims
print("### SIMS for %s : %d" % (tid, len(sims)))
for s in sims:
    print("--- sim %s reward=%s term=%s nmsg=%d" % (
        s.get("id"), (s.get("reward_info") or {}).get("reward"),
        s.get("termination_reason"), len(s.get("messages") or [])))
s = sims[0]
ri = s.get("reward_info") or {}
print("\n### REWARD_INFO")
print(json.dumps(ri, ensure_ascii=False, indent=1)[:6000])
print("\n### MESSAGES")
for i, m in enumerate(s.get("messages") or []):
    role = m.get("role")
    turn = m.get("turn_idx")
    content = m.get("content")
    tcs = m.get("tool_calls") or []
    print("\n=== [%d] role=%s turn_idx=%s" % (i, role, turn))
    if content:
        print("CONTENT: %s" % content)
    for tc in tcs:
        print("TOOLCALL: %s(%s)" % (tc.get("name"), json.dumps(tc.get("arguments"), ensure_ascii=False)))
    if m.get("requestor"):
        print("REQUESTOR: %s" % m.get("requestor"))
