#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x780 - sidecar (fb_*.jsonl.gz) rows for one task, ordered by turn."""
import gzip, json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
p, tid = sys.argv[1], sys.argv[2]
full = "--full" in sys.argv
tmin = int(sys.argv[sys.argv.index("--tmin")+1]) if "--tmin" in sys.argv else -1
tmax = int(sys.argv[sys.argv.index("--tmax")+1]) if "--tmax" in sys.argv else 10**9
rows = []
for ln in gzip.open(p, "rt", encoding="utf-8"):
    ln = ln.strip()
    if not ln: continue
    d = json.loads(ln)
    st = d.get("simtag") or ""
    if tid not in st: continue
    rows.append(d)
rows.sort(key=lambda d: (d.get("turn") if d.get("turn") is not None else -1))
print("### fb rows for %s : %d" % (tid, len(rows)))
for d in rows:
    t = d.get("turn")
    if t is None: t = -1
    if not (tmin <= t <= tmax): continue
    print("\n--- turn=%s kind=%s call=%s channel=%s target=%s outcome=%s arrived=%s folded=%s rank=%s lost_to=%s err=%s" % (
        t, d.get("kind"), d.get("call_name"), d.get("channel"), d.get("target"),
        d.get("outcome"), d.get("arrived"), d.get("folded"), d.get("rank"), d.get("lost_to"), d.get("err")))
    oh = d.get("out_head")
    if oh: print("OUT: %s" % (oh if full else oh[:600]))
    tx = d.get("text")
    if tx: print("TEXT: %s" % (tx if full else tx[:1200]))
