#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x780 - pair gold actions vs agent calls on a chosen key, per write tool."""
import gzip, json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
path, tid = sys.argv[1], sys.argv[2]
simid = sys.argv[3]
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt", encoding="utf-8") as f: res = json.load(f)
s = [x for x in res["simulations"] if x["id"] == simid][0]
ri = s.get("reward_info") or {}

def inner(args):
    a = dict(args or {})
    if isinstance(a.get("arguments"), str):
        try: a.update(json.loads(a["arguments"])); a.pop("arguments")
        except Exception: pass
    return a

print("### GOLD write actions")
gold = {}
for ac in (ri.get("action_checks") or []):
    act = ac.get("action") or {}
    a = inner(act.get("arguments"))
    key = a.get("transaction_id") or a.get("account_id") or a.get("credit_card_account_id") or json.dumps(a, sort_keys=True)[:60]
    gold[key] = a
    print("  %-8s %-45s %s" % (act.get("action_id"), act.get("name"), json.dumps(a, ensure_ascii=False, sort_keys=True)))

print("\n### AGENT write calls")
for i, m in enumerate(s["messages"]):
    for tc in (m.get("tool_calls") or []):
        a = inner(tc.get("arguments"))
        print("  [%d] %-35s %s" % (i, tc.get("name"), json.dumps(a, ensure_ascii=False, sort_keys=True)))
