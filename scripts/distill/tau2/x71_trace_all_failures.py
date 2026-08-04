"""Render every failed run compactly enough that all of them can actually be read.

Class tallies kept sending the read-through to three vivid trajectories and a general
claim. This dumps all of them — one dense line per turn, tool calls with their arguments,
results clipped to the part that changes what happens next — so the reason for each task
is read off its own transcript rather than inferred from a bucket it landed in.

Gold is printed first with each action marked hit or miss, so the divergence is visible
against what was actually required.
"""

import argparse
import glob
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x50_says_not_does import ARMS, SIM, norm_args  # noqa: E402

NOISE = ("get_current_time",)


def clip(s, n):
    return " ".join((s or "").split())[:n]


def render(s):
    ri = s.get("reward_info") or {}
    out = [f"═══ {s['task_id']}/t{s.get('trial')}  reward={ri.get('reward')} "
           f"basis={ri.get('reward_basis')} db={(ri.get('db_check') or {}).get('db_match')} "
           f"term={s.get('termination_reason')} msgs={len(s.get('messages') or [])}"]
    for c in ri.get("action_checks") or []:
        g = c.get("action") or {}
        a = norm_args(g.get("arguments"))
        inner = a.get("agent_tool_name") or a.get("discoverable_tool_name") or a.get("user_tool_name")
        nm = inner or g.get("name")
        rest = a.get("arguments") if inner else a
        out.append(f"  {'OK ' if c.get('action_match') else 'MISS'} {g.get('requestor','')[:5]:5} "
                   f"{nm} {clip(json.dumps(rest, ensure_ascii=False), 90)}")
    for i, m in enumerate(s.get("messages") or []):
        r = m.get("role")
        txt = clip(m.get("content"), 150)
        if r == "user":
            if txt:
                out.append(f" {i:3} U: {txt}")
            for tc in m.get("tool_calls") or []:
                out.append(f" {i:3}   ↑user ran {tc.get('name')} "
                           f"{clip(json.dumps(tc.get('arguments'), ensure_ascii=False), 110)}")
        elif r == "assistant":
            if txt:
                out.append(f" {i:3} A: {txt}")
            for tc in m.get("tool_calls") or []:
                n = tc.get("name") or (tc.get("function") or {}).get("name")
                a = tc.get("arguments")
                if a is None:
                    a = (tc.get("function") or {}).get("arguments")
                out.append(f" {i:3}   → {n} {clip(json.dumps(a, ensure_ascii=False), 130)}")
        elif r == "tool":
            if any(k in (m.get("content") or "") for k in
                   ("NOT_VERIFIED", "VERIFIED", "Error", "error", "denied", "[T2_", "[coverage",
                    "DUPLICATE", "unlocked", "successful", "Executed", "not found", "No records")):
                out.append(f" {i:3}   ← {clip(m.get('content'), 170)}")
            else:
                out.append(f" {i:3}   ← {clip(m.get('content'), 80)}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="N97", choices=sorted(ARMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-front", action="store_true")
    args = ap.parse_args()

    front = set()
    if args.exclude_front:
        for a in ("A", "B4"):
            for p in sorted(glob.glob(f"{SIM}/{ARMS[a]}.results.json.gz")):
                for s in json.load(gzip.open(p, "rt", encoding="utf-8"))["simulations"]:
                    front.add(s["task_id"])

    sims = []
    for p in sorted(glob.glob(f"{SIM}/{ARMS[args.arm]}.results.json.gz")):
        sims += json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or []

    chosen = [s for s in sorted(sims, key=lambda x: (x["task_id"], x.get("trial") or 0))
              if (s.get("reward_info") or {}).get("reward") != 1.0
              and s["task_id"] not in front]
    body = "\n\n".join(render(s) for s in chosen)
    open(args.out, "w", encoding="utf-8").write(body)
    print(f"{len(chosen)} 실패 sim → {args.out}  ({body.count(chr(10))+1} 줄)")


if __name__ == "__main__":
    main()
