"""Where in the trajectory each failed run left the gold path, and what it did instead.

The engine does not change between tasks, so "the engine is at fault" is not a claim a
class tally can support. What it can be checked against is position: if runs on one task
set leave the gold path at the same place — before the first discoverable-tool step, say
— and runs on another set leave it deep in, then the two sets fail at different stations
and only one of them is about the levers that were built.

So this walks gold in order against the agent's own calls and reports, per failed run:

  k / N            the first gold action that was missed, and how many there were
  divergence step  the transcript step of the last gold action that *did* match — after
                   that point the run is off the path
  instead          the agent's next tool call after that point, or that it stopped calling

`k = 1` means the run never got onto the path at all. That is a different failure from a
run that completes eight gold actions and drops the ninth, and the aggregate class counts
cannot tell them apart.
"""

import argparse
import collections
import glob
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x50_says_not_does import ARMS, SIM, norm_args  # noqa: E402
from x66_effective_tool_miss import WRAPPER, effective  # noqa: E402


def indexed_actions(sim):
    """(step, wrapper, effective_name, effective_args) for each assistant tool call."""
    out = []
    for i, m in enumerate(sim.get("messages") or []):
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            n = tc.get("name") or (tc.get("function") or {}).get("name")
            a = tc.get("arguments")
            if a is None:
                a = (tc.get("function") or {}).get("arguments")
            eff, eargs = effective(n, norm_args(a))
            out.append((i, n, eff, eargs))
    return out


def load(pattern):
    out = []
    for p in sorted(glob.glob(f"{SIM}/{pattern}.results.json.gz")):
        out.extend(json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or [])
    return out


def analyse(sim):
    checks = [c for c in (sim.get("reward_info") or {}).get("action_checks") or []
              if (c.get("action") or {}).get("requestor") == "assistant"]
    if not checks:
        return None
    acts = indexed_actions(sim)

    first_miss = None
    last_ok_step = -1
    for k, c in enumerate(checks, 1):
        g = c.get("action") or {}
        geff, _ = effective(g.get("name"), norm_args(g.get("arguments")))
        if c.get("action_match"):
            hit = [i for i, _, e, _ in acts if e == geff and i > last_ok_step]
            if hit:
                last_ok_step = hit[0]
            continue
        if first_miss is None:
            first_miss = (k, geff, g.get("name"))
    if first_miss is None:
        return None

    nxt = next(((i, w, e) for i, w, e, _ in acts if i > last_ok_step), None)
    return {
        "k": first_miss[0], "n": len(checks), "tool": first_miss[1],
        "wrapper": first_miss[2], "last_ok_step": last_ok_step,
        "instead": f"{nxt[2]}" if nxt else "(더 이상 도구 호출 없음)",
        "instead_wrapper": (nxt[1] if nxt else None),
        "calls_total": len(acts),
        "unlocks": sum(1 for _, w, _, _ in acts if w == "unlock_discoverable_agent_tool"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="N97", choices=sorted(ARMS))
    ap.add_argument("--out")
    args = ap.parse_args()

    front = {s["task_id"] for s in load(ARMS["A"])} | {s["task_id"] for s in load(ARMS["B4"])}
    sims = load(ARMS[args.arm])

    lines, agg = [], collections.defaultdict(collections.Counter)
    kdist = collections.defaultdict(list)
    instead = collections.defaultdict(collections.Counter)
    for s in sorted(sims, key=lambda x: (x["task_id"], x.get("trial") or 0)):
        if (s.get("reward_info") or {}).get("reward") == 1.0:
            continue
        r = analyse(s)
        if r is None:
            lines.append(f"{s['task_id']}/t{s.get('trial')}  "
                         f"에이전트-측 gold 없음 [{s.get('termination_reason')}]")
            continue
        grp = "front32" if s["task_id"] in front else "new65"
        agg[grp]["fail"] += 1
        agg[grp]["첫 gold부터 이탈(k=1)"] += (r["k"] == 1)
        kdist[grp].append(r["k"] / r["n"])
        instead[grp][r["instead"]] += 1
        lines.append(
            f"{s['task_id']}/t{s.get('trial')}  [{grp}]  이탈 {r['k']}/{r['n']} "
            f"· 마지막 정답 step {r['last_ok_step']} · 놓친 것 {r['tool']} "
            f"· 대신 {r['instead']} · 총 호출 {r['calls_total']} · unlock {r['unlocks']}")

    print(f"=== {args.arm}: 첫 이탈 지점 ===")
    for grp in ("front32", "new65"):
        a = agg[grp]
        if not a["fail"]:
            continue
        ks = sorted(kdist[grp])
        med = ks[len(ks) // 2]
        print(f"\n  [{grp}]  실패 {a['fail']}")
        print(f"    첫 gold 행동부터 이탈(k=1) : {a['첫 gold부터 이탈(k=1)']} "
              f"({100*a['첫 gold부터 이탈(k=1)']/a['fail']:.0f}%)")
        print(f"    이탈 위치 중앙값           : gold의 {100*med:.0f}% 지점")
        print(f"    이탈 직후 한 일 (상위 6)")
        for k, v in instead[grp].most_common(6):
            print(f"      {v:3}x  {k}")

    if args.out:
        open(args.out, "w", encoding="utf-8").write("\n".join(lines))
        print(f"\n전 실패 sim 목록 → {args.out} ({len(lines)}행)")


if __name__ == "__main__":
    main()
