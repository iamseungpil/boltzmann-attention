"""Per-step trajectory dump for AX33 run-g, decomposed to the argument level.

Prints every step of a simulation: what the model was holding in context (the
preceding tool response, verbatim), what it said, and every tool call with its
full arguments. Gold actions from reward_info are printed first so each proposed
call can be read against what the task required.

The point is prompt-level attribution: for a wrong call, the text that produced
it is the tool response immediately above it.

Usage:
    python ax33g_perstep.py --task task_041 --trial 0
    python ax33g_perstep.py --task task_041 --trial 0 --width 4000
    python ax33g_perstep.py --all --outdir /home/woori/scratch/ax33fx/steps
"""

import argparse
import glob
import gzip
import json
import os

SIM_DIR = (
    "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results"
)


def load(tag):
    sims = []
    for path in sorted(glob.glob(f"{SIM_DIR}/bank_ax33n_gpu*_{tag}.results.json.gz")):
        sims.extend(json.load(gzip.open(path, "rt", encoding="utf-8")).get("simulations") or [])
    return sims


def clip(text, width):
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    text = text.replace("\r", "")
    if len(text) <= width:
        return text
    head = width * 2 // 3
    tail = width - head
    return f"{text[:head]}\n   … [{len(text) - width} chars elided] …\n{text[-tail:]}"


def tool_calls(msg):
    out = []
    for tc in msg.get("tool_calls") or []:
        name = tc.get("name") or (tc.get("function") or {}).get("name")
        args = tc.get("arguments")
        if args is None:
            args = (tc.get("function") or {}).get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        out.append((name, args))
    return out


def dump(sim, width, out):
    ri = sim.get("reward_info") or {}
    p = out.append
    p("#" * 100)
    p(f"# {sim.get('task_id')}  trial {sim.get('trial')}  reward={ri.get('reward')}  "
      f"term={sim.get('termination_reason')}  dur={round(sim.get('duration') or 0)}s")
    db = ri.get("db_check") or {}
    p(f"# db_match={db.get('db_match')}  reward_basis={ri.get('reward_basis')}  "
      f"breakdown={ri.get('reward_breakdown')}")
    p("#" * 100)

    checks = ri.get("action_checks") or []
    if checks:
        p("\n=== GOLD ACTIONS (required) ===")
        for c in checks:
            a = c.get("action") or {}
            mark = "OK " if c.get("action_match") else "MISS"
            p(f"  [{mark}] {a.get('name')}  {json.dumps(a.get('arguments'), ensure_ascii=False)}")
    for ea in ri.get("env_assertions") or []:
        p(f"  [env {'OK' if ea.get('met') else 'MISS'}] {ea.get('env_assertion')}")

    p("\n=== STEPS ===")
    call_no = 0
    for i, m in enumerate(sim.get("messages") or []):
        role = m.get("role")
        content = m.get("content")
        if role == "assistant":
            p(f"\n[{i:3d}] ASSISTANT")
            if content:
                p(f"      say: {clip(content, width)}")
            for name, args in tool_calls(m):
                p(f"      CALL #{call_no}: {name}")
                pretty = json.dumps(args, ensure_ascii=False, indent=8, sort_keys=True)
                p("        args " + clip(pretty, width))
                call_no += 1
        elif role == "tool":
            p(f"[{i:3d}] TOOL RESULT ({len(content) if isinstance(content, str) else 0} chars)")
            p(f"      {clip(content, width)}")
        elif role == "user":
            p(f"[{i:3d}] USER: {clip(content, width)}")
        else:
            p(f"[{i:3d}] {role}: {clip(content, width)}")
    p("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="20260803g")
    ap.add_argument("--task")
    ap.add_argument("--trial", type=int)
    ap.add_argument("--width", type=int, default=1500)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--outdir")
    args = ap.parse_args()

    sims = load(args.tag)
    if args.task:
        sims = [s for s in sims if s.get("task_id") == args.task]
    if args.trial is not None:
        sims = [s for s in sims if (s.get("trial") or 0) == args.trial]
    sims.sort(key=lambda s: (s.get("task_id") or "", s.get("trial") or 0))

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        for s in sims:
            buf = []
            dump(s, args.width, buf)
            path = os.path.join(args.outdir, f"{s.get('task_id')}_t{s.get('trial')}.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(buf))
        print(f"wrote {len(sims)} dumps to {args.outdir}")
        return

    buf = []
    for s in sims:
        dump(s, args.width, buf)
    print("\n".join(buf))


if __name__ == "__main__":
    main()
