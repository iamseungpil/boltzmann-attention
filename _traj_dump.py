#!/usr/bin/env python3
"""Dump two_stage trajectory(ies) by task-id substring. Usage: _traj_dump.py <results.json> <substr> [maxlines]"""
import json, re, sys

def step(c):
    if not c:
        return ""
    m = re.search(r"Plan:\s*([a-zA-Z_]+)", c)
    return m.group(1) if m else ""

def short(x, n):
    return (str(x) or "").replace("\n", " ")[:n]

def dump(s, maxlines):
    print("####", s.get("task_id"), "term=", s.get("termination_reason"),
          "reward=", (s.get("reward_info") or {}).get("reward"))
    n = 0
    for m in s.get("messages", []):
        if n > maxlines:
            print("  ...(truncated)")
            break
        role = m.get("role")
        if role == "assistant":
            tcs = m.get("tool_calls") or []
            st = step(m.get("content"))
            if tcs:
                for tc in tcs:
                    a = json.dumps(tc.get("arguments") or tc.get("args") or {}, ensure_ascii=False)
                    print("  A[%s] %s(%s)" % (st or "-", tc.get("name"), a[:90]))
                    n += 1
            else:
                print("  A.say:", short(m.get("content"), 95))
                n += 1
        elif role == "user":
            print("  U:", short(m.get("content"), 95))
            n += 1
        elif role == "tool":
            print("    t->", short(m.get("content"), 85))
            n += 1

def main():
    path, sub = sys.argv[1], sys.argv[2]
    maxlines = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    sims = {s["task_id"]: s for s in json.load(open(path))["simulations"]}
    hits = [t for t in sims if sub in t]
    if not hits:
        print("NO MATCH for:", sub); return
    for t in hits[:1]:
        dump(sims[t], maxlines)

if __name__ == "__main__":
    main()
