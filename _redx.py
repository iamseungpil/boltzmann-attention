#!/usr/bin/env python3
"""Quantify resolver failure mechanisms: read-loop signature + write-call sparsity.
Usage: _redx.py <base_dir>"""
import json, sys, os
from collections import Counter
import statistics as st

# read tool names for telecom (heuristic: anything starting get_/check_/lookup_ or 'read')
def is_read(name):
    if not name:
        return False
    n = name.lower()
    return n.startswith(("get_", "check_", "lookup_", "list_", "search_", "find_", "read"))

def calls(s):
    out = []
    for m in s.get("messages", []):
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                out.append((tc.get("name"), json.dumps(tc.get("arguments") or tc.get("args") or {}, sort_keys=True)))
    return out

def main():
    base = sys.argv[1]
    paths = {k: f"{base}/tel_{k}/{k}_telecom_base.json/results.json" for k in ["base", "resolver", "fallback"]}
    for k, p in paths.items():
        if not os.path.exists(p):
            print(k, "MISSING", p); continue
        sims = {s["task_id"]: s for s in json.load(open(p))["simulations"]}
        max_rep = []          # per-sim max repetition of identical (name,args) call
        read_frac = []        # per-sim fraction of calls that are reads
        n_writes = []         # per-sim agent write-call count
        loopers = 0           # sims with an identical call repeated >=4x
        loop_fail = 0         # of those, how many failed
        for tid, s in sims.items():
            cs = calls(s)
            if not cs:
                max_rep.append(0); read_frac.append(0.0); n_writes.append(0); continue
            c = Counter(cs)
            mr = max(c.values())
            max_rep.append(mr)
            nr = sum(1 for (nm, _) in cs if is_read(nm))
            read_frac.append(nr / len(cs))
            nw = sum(1 for (nm, _) in cs if not is_read(nm))
            n_writes.append(nw)
            r = (s.get("reward_info") or {}).get("reward", 0) or 0
            if mr >= 4:
                loopers += 1
                if r < 0.999:
                    loop_fail += 1
        n = len(sims)
        print(f"=== {k} (N={n}) ===")
        print(f"  max identical-call repetition: mean={st.mean(max_rep):.1f} median={st.median(max_rep):.0f} max={max(max_rep)}")
        print(f"  read fraction of agent calls : mean={st.mean(read_frac):.2f}")
        print(f"  agent WRITE calls per task   : mean={st.mean(n_writes):.2f} median={st.median(n_writes):.0f} (tasks w/ 0 writes={sum(1 for x in n_writes if x==0)})")
        print(f"  read-LOOP sims (>=4x same call): {loopers}  (of which failed: {loop_fail})")

if __name__ == "__main__":
    main()
