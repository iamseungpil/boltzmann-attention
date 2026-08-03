"""Every failed B4 simulation, traced to where it broke.

Same shape as `ax33g_taskcause.py` but pointed at the live B4 run and carrying the
two arm artifacts, so each failure can be read against the question that matters:
did ① (the new toll) or ④ (the matches line) touch this trajectory before it went
wrong, or was it already going wrong for its own reasons?

Only requestor=assistant gold actions are the agent's to emit; user-requestor ones
are executed by the customer after a hand-off and their absence says nothing about
the agent (see AX33G_SPLIT_FORENSIC §7.5).
"""

import argparse
import collections
import glob
import gzip
import io
import json

SD = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results"
NOTICE = "TRANSFER NOTICE: Would you like"


def load(pattern, gz=False):
    sims = []
    for p in sorted(glob.glob(pattern)):
        try:
            d = json.load(gzip.open(p, "rt", encoding="utf-8")) if gz \
                else json.load(io.open(p, encoding="utf-8"))
        except Exception:
            continue
        sims.extend(d.get("simulations") or [])
    return sims


def norm(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {"_raw": a}
    return a if isinstance(a, dict) else {}


def calls(sim):
    out = []
    for m in sim.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            n = tc.get("name") or (tc.get("function") or {}).get("name")
            a = tc.get("arguments")
            if a is None:
                a = (tc.get("function") or {}).get("arguments")
            out.append((n, norm(a)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="/home/woori/scratch/tau2-bench/data/simulations/"
                                      "bank_b4_gpu*_20260803h/results.json")
    args = ap.parse_args()

    B = load(args.live)
    A = {(s["task_id"], s.get("trial")): s for s in
         load(f"{SD}/bank_ax33n_gpu*_20260803g.results.json.gz", gz=True)}

    fails = [s for s in B if ((s.get("reward_info") or {}).get("reward") or 0.0) != 1.0]
    print(f"B4 완주 {len(B)} · 실패 {len(fails)}\n")

    tally = collections.Counter()
    for s in sorted(fails, key=lambda x: (x["task_id"], x.get("trial") or 0)):
        ri = s.get("reward_info") or {}
        cl = calls(s)
        key = (s["task_id"], s.get("trial"))
        a = A.get(key)
        arew = (a.get("reward_info") or {}).get("reward") if a else None

        notice = [i for i, m in enumerate(s.get("messages") or [])
                  if m.get("role") == "assistant" and isinstance(m.get("content"), str)
                  and NOTICE in m["content"]]
        matches = [i for i, m in enumerate(s.get("messages") or [])
                   if m.get("role") == "tool" and isinstance(m.get("content"), str)
                   and "matches:" in m["content"]]

        agg = collections.Counter()
        for c in ri.get("action_checks") or []:
            if c.get("action_match"):
                continue
            g = c.get("action") or {}
            if g.get("requestor") != "assistant":
                continue
            n, ga = g.get("name"), norm(g.get("arguments") or {})
            tried = [e for e in cl if e[0] == n]
            if not tried:
                agg[(n, "NEVER")] += 1
            else:
                best = min(tried, key=lambda e: sum(
                    1 for k in set(ga) | set(e[1]) if ga.get(k) != e[1].get(k)))
                d = [k for k in sorted(set(ga) | set(best[1])) if ga.get(k) != best[1].get(k)]
                agg[(n, "ARG:" + ",".join(d[:3]))] += 1

        mark = "" if arew is None else ("  [A도 실패]" if arew != 1.0 else "  ★[A는 통과]")
        tally["A도 실패" if arew != 1.0 else "A는 통과"] += 1
        print(f"## {key[0]}/t{key[1]}  term={s.get('termination_reason')} calls={len(cl)} "
              f"basis={ri.get('reward_basis')} db={(ri.get('db_check') or {}).get('db_match')}{mark}")
        print(f"   ① notice@{notice or '-'}   ④ matches {len(matches)}개"
              + (f" (최초 msg{matches[0]})" if matches else ""))
        if not agg:
            print("   에이전트-측 MISS 없음  ⇒ 실패는 user-측 실행 또는 db 상태")
        for k2, v in agg.most_common():
            print(f"   {v}x {k2[0]}  [{k2[1]}]")
        tail = [m for m in (s.get("messages") or [])
                if m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"]]
        print(f"   last: {' '.join(tail[-1]['content'].split())[:110] if tail else ''}\n")

    print("=== 요약 ===", dict(tally))


if __name__ == "__main__":
    main()
