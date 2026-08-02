"""Prompt-level cause classifier for AX33 run-g.

task_041/0 gave a fully readable mechanism: the agent hands a discoverable tool to
the user, then declares in prose that it will call that tool itself, then emits a
different tool it does have — 85 times, verbatim, while the engine's duplicate note
counts up to 82. This script tests how far that signature and its neighbours
generalise across all 64 simulations.

Signatures, all decidable without domain knowledge:
  DECL_MISMATCH  assistant names a tool in backticks, then calls a different one
  GAVE_AWAY      agent gives a discoverable tool to the user, then never asks the
                 user to run it (no reference in later assistant prose)
  REPEAT_LOOP    same (tool, args) issued >= 4 times
  ZERO_CALL      simulation ends with no tool call at all
  VERIFY_ORDER   verify_identity called before any get_user_information_by_*
"""

import argparse
import collections
import glob
import gzip
import json
import re

SIM_DIR = (
    "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results"
)
BACKTICK = re.compile(r"`([a-z][a-z0-9_]{4,})`")


def load(tag):
    sims = []
    for path in sorted(glob.glob(f"{SIM_DIR}/bank_ax33n_gpu*_{tag}.results.json.gz")):
        sims.extend(json.load(gzip.open(path, "rt", encoding="utf-8")).get("simulations") or [])
    return sims


def steps(sim):
    """(kind, payload) stream: ('say', text) | ('call', (name, args)) | ('tool', text)."""
    out = []
    for m in sim.get("messages") or []:
        role = m.get("role")
        content = m.get("content") if isinstance(m.get("content"), str) else ""
        if role == "assistant":
            if content:
                out.append(("say", content))
            for tc in m.get("tool_calls") or []:
                name = tc.get("name") or (tc.get("function") or {}).get("name")
                args = tc.get("arguments")
                if args is None:
                    args = (tc.get("function") or {}).get("arguments")
                if not isinstance(args, str):
                    args = json.dumps(args, sort_keys=True, ensure_ascii=False)
                out.append(("call", (name, args)))
        elif role == "tool":
            out.append(("tool", content))
        elif role == "user":
            out.append(("user", content))
    return out


def classify(sim, tool_names):
    st = steps(sim)
    calls = [p for k, p in st if k == "call"]
    flags = collections.Counter()
    detail = {}

    # DECL_MISMATCH: prose names a known tool, the very next call is a different tool
    pending = None
    for kind, payload in st:
        if kind == "say":
            named = [t for t in BACKTICK.findall(payload) if t in tool_names]
            pending = named[-1] if named else None
        elif kind == "call":
            if pending and payload[0] != pending:
                flags["DECL_MISMATCH"] += 1
                detail.setdefault("decl", (pending, payload[0]))
            pending = None

    # GAVE_AWAY: handed a discoverable tool to the user, then pursued it alone
    given = []
    for name, args in calls:
        if name == "give_discoverable_user_tool":
            try:
                given.append(json.loads(args).get("discoverable_tool_name"))
            except Exception:
                pass
    asked = any(
        k == "say" and any(g and g in p for g in given) and "call_discoverable_user_tool" in p
        for k, p in st
    )
    if given and not asked:
        flags["GAVE_AWAY"] = len(given)
        detail["given"] = given

    counts = collections.Counter(calls)
    if counts:
        top, n = counts.most_common(1)[0]
        if n >= 4:
            flags["REPEAT_LOOP"] = n
            detail["repeat"] = top[0]
    if not calls:
        flags["ZERO_CALL"] = 1

    first_verify = next((i for i, c in enumerate(calls) if c[0] == "verify_identity"), None)
    first_fetch = next(
        (i for i, c in enumerate(calls) if c[0].startswith("get_user_information")), None
    )
    if first_verify is not None and (first_fetch is None or first_verify < first_fetch):
        flags["VERIFY_ORDER"] = 1

    return flags, detail, len(calls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="20260803g")
    args = ap.parse_args()
    sims = load(args.tag)

    tool_names = set()
    for s in sims:
        for k, p in steps(s):
            if k == "call" and p[0]:
                tool_names.add(p[0])

    rows = []
    for s in sims:
        flags, detail, ncalls = classify(s, tool_names)
        r = (s.get("reward_info") or {}).get("reward") or 0.0
        rows.append((s.get("task_id"), s.get("trial"), r, ncalls, flags, detail))
    rows.sort()

    print(f"tools seen: {len(tool_names)}   sims: {len(rows)}\n")
    print(f"{'task':10s} {'t':>2} {'rew':>4} {'calls':>5}  signatures")
    for t, tr, r, n, f, d in rows:
        sig = " ".join(f"{k}={v}" for k, v in sorted(f.items()))
        print(f"{t:10s} {tr:2d} {r:4.1f} {n:5d}  {sig}")
        if "decl" in d:
            print(f"{'':25s}   declared `{d['decl'][0]}` -> called `{d['decl'][1]}`")
        if "given" in d:
            print(f"{'':25s}   gave away {d['given']} and never asked the user to run it")

    print("\n=== signature x outcome ===")
    for key in ["DECL_MISMATCH", "GAVE_AWAY", "REPEAT_LOOP", "ZERO_CALL", "VERIFY_ORDER"]:
        hit = [r for r in rows if key in r[4]]
        miss = [r for r in rows if key not in r[4]]
        ph = sum(1 for r in hit if r[2] == 1.0) / len(hit) if hit else float("nan")
        pm = sum(1 for r in miss if r[2] == 1.0) / len(miss) if miss else float("nan")
        print(f"  {key:14s} n={len(hit):3d} pass={ph:.3f}   |  absent n={len(miss):3d} pass={pm:.3f}")


if __name__ == "__main__":
    main()
