#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x780 - compact per-message dump."""
import gzip, json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
path, tid = sys.argv[1], sys.argv[2]
simid = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else None
W = int(sys.argv[sys.argv.index("--w")+1]) if "--w" in sys.argv else 700
lo = int(sys.argv[sys.argv.index("--lo")+1]) if "--lo" in sys.argv else 0
hi = int(sys.argv[sys.argv.index("--hi")+1]) if "--hi" in sys.argv else 10**9
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt", encoding="utf-8") as f: res = json.load(f)
sims = [s for s in res.get("simulations", []) if s.get("task_id") == tid]
if simid: sims = [s for s in sims if s.get("id") == simid] or sims
s = sims[0]
for i, m in enumerate(s.get("messages") or []):
    if not (lo <= i <= hi): continue
    role = m.get("role"); turn = m.get("turn_idx")
    print("\n=== [%d] %s turn_idx=%s%s" % (i, role, turn, (" req=" + str(m.get("requestor"))) if m.get("requestor") else ""))
    c = m.get("content")
    if c: print(c[:W])
    for tc in (m.get("tool_calls") or []):
        print(">>> CALL %s(%s)" % (tc.get("name"), json.dumps(tc.get("arguments"), ensure_ascii=False)[:W*2]))
