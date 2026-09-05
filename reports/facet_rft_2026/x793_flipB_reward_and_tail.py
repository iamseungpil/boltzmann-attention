#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x793 - flip 짝의 **채점 단위**(reward_info)와 궤적 꼬리를 나란히 찍는다. 2026-09-05.
사용:  python x793_flipB_reward_and_tail.py <pairs.txt> [tail_n] [maxchars]
⛔ 판정하지 않는다.
"""
import io, json, sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

SIMROOT = "/home/woori/scratch/tau2-bench/data/simulations"
_c = {}


def load(tag):
    if tag not in _c:
        _c[tag] = json.load(open("%s/%s/results.json" % (SIMROOT, tag)))
    return _c[tag]


def main():
    pairs = [l.split() for l in open(sys.argv[1]).read().splitlines() if l.strip() and not l.startswith("#")]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    MAXC = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    for task, ptag, psim, ftag, fsim in pairs:
        print("\n" + "=" * 100)
        print("### %s" % task)
        for lab, tag, sid in (("PASS", ptag, psim), ("FAIL", ftag, fsim)):
            d = load(tag)
            s = next(x for x in d["simulations"] if x["id"] == sid)
            ri = s.get("reward_info") or {}
            print("-- %s %s term=%s nmsg=%d reward=%s" % (lab, tag, s.get("termination_reason"),
                                                          len(s["messages"]), ri.get("reward")))
            print("   reward_breakdown=%s" % json.dumps(ri.get("reward_breakdown"), ensure_ascii=False)[:400])
            for k in ("db_check", "env_assertions", "action_checks", "nl_assertions", "communicate_checks"):
                v = ri.get(k)
                if v:
                    print("   %s=%s" % (k, json.dumps(v, ensure_ascii=False)[:1400]))
            ms = s["messages"]
            for i in range(max(0, len(ms) - N), len(ms)):
                m = ms[i]
                cc = (m.get("content") or "")
                if isinstance(cc, str):
                    cc = cc.replace("\n", " \n ")
                tcs = " ;; ".join("%s(%s)" % (t.get("name"), json.dumps(t.get("arguments"), ensure_ascii=False))
                                 for t in (m.get("tool_calls") or []))
                print("   [%d] %-9s %s%s" % (i, m.get("role"), cc[:MAXC], (" ;; " + tcs) if tcs else ""))


if __name__ == "__main__":
    main()
