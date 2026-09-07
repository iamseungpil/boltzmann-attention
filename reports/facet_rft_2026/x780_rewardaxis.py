#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x780b - reward axis + failing action units, compact."""
import gzip, json, sys, io
try: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception: pass
path, tid = sys.argv[1], sys.argv[2]
simid = sys.argv[3] if len(sys.argv)>3 else None
op = gzip.open if path.endswith(".gz") else open
with op(path,"rt",encoding="utf-8") as f: res=json.load(f)
sims=[s for s in res.get("simulations",[]) if s.get("task_id")==tid]
if simid: sims=[s for s in sims if s.get("id")==simid] or sims
s=sims[0]
ri=s.get("reward_info") or {}
print("reward=%s term=%s nmsg=%d" % (ri.get("reward"), s.get("termination_reason"), len(s.get("messages") or [])))
print("db_check=%s" % json.dumps(ri.get("db_check"),ensure_ascii=False))
print("reward_basis=%s" % json.dumps(ri.get("reward_basis"),ensure_ascii=False))
print("info=%s" % json.dumps(ri.get("info"),ensure_ascii=False)[:1500])
acs=ri.get("action_checks") or []
print("ACTION_CHECKS n=%d  matched=%d" % (len(acs), sum(1 for a in acs if a.get("action_match"))))
for a in acs:
    act=a.get("action") or {}
    print("  %-8s match=%-5s type=%-7s %s(%s)" % (act.get("action_id"), a.get("action_match"), a.get("tool_type"), act.get("name"), json.dumps(act.get("arguments"),ensure_ascii=False)[:260]))
