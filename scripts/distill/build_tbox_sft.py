#!/usr/bin/env python3
"""build_tbox_sft.py — TRUE TBox-only planner SFT data (design fix, 2026-05-30).

build_abstract_sft.py only PREPENDS 'Plan: <step>' to each fix turn while KEEPING the
original content + the concrete tool_call, so the trained model stays TBox+ABox
ENTANGLED (it still learns which concrete tool/args to emit). That is why the
"base"/abstract model already scores ~87% turn-level GT-correct on telecom: the ABox
is internalized, not delegated.

This script instead REPLACES each action turn's assistant content with exactly
'Plan: <step>'. The concrete tool_calls are KEPT in the record only so the chat
template stays valid (a following tool result needs a parent tool_call); the trainer
run with --mask-toolcalls supervises ONLY the content ('Plan: <step>') tokens and
masks the tool_call tokens to -100. The planner therefore learns the abstract step
policy (TBox) and gets ZERO gradient on concrete tool selection (ABox) — the resolver
is the sole ABox source at inference (two_stage_agent --mode tbox / resolver).

Abstract step per action turn (domain-general), via step_realization_<dom>.json:
  - tool in step map      -> that step (apply_targeted_fix/apply_policy_action/escalate_or_document)
  - all tools are reads    -> gather_account_context
  - unknown write          -> apply_targeted_fix (fallback)
Content-only assistant turns (user-facing messages, no tool_calls) are UNCHANGED so
the planner still learns WHEN to talk vs act.

Usage:
  python scripts/distill/build_tbox_sft.py --domains telecom retail airline
  # then concat telecom+retail -> telret (airline held out, LODO), like abstract.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_fix_coverage import is_read  # noqa: E402

SFT_DIR = "reports/facet_rft_2026/phase4_distill/sft_data"
INDUCED_DIR = "reports/facet_rft_2026/phase4_distill/induced"

GATHER = "gather_account_context"
WRITE_FALLBACK = "apply_targeted_fix"


def load_realization(domain, induced_dir):
    p = os.path.join(induced_dir, f"step_realization_{domain}.json")
    if not os.path.exists(p):
        raise SystemExit(f"[err] missing {p} — run induce_step_realization.py --domain {domain}")
    return json.load(open(p)).get("step_realizes_tool", {})


def turn_step(tool_names, step_map):
    mapped = []
    for tn in tool_names:
        s = step_map.get(tn)
        if s and s != "step" and s not in mapped:
            mapped.append(s)
    if mapped:
        return " ".join(mapped)
    if tool_names and all(is_read(t) for t in tool_names):
        return GATHER
    return WRITE_FALLBACK


def convert(record, step_map, stats):
    for m in record.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls") or []
        if not tcs:
            stats["msg_only"] += 1
            continue
        names = [((tc.get("function") or {}).get("name") or tc.get("name")) for tc in tcs]
        names = [n for n in names if n]
        step = turn_step(names, step_map)
        m["content"] = f"Plan: {step}"
        stats[f"step:{step}"] += 1
        stats["action_turns"] += 1
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["telecom", "retail", "airline"])
    ap.add_argument("--sft-dir", default=SFT_DIR)
    ap.add_argument("--induced-dir", default=INDUCED_DIR)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    for dom in args.domains:
        step_map = load_realization(dom, args.induced_dir)
        src = os.path.join(args.sft_dir, f"sft_plain_{args.split}_{dom}.jsonl")
        if not os.path.exists(src):
            print(f"[warn] missing {src}; skip"); continue
        stats = Counter()
        out_recs = []
        with open(src) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out_recs.append(convert(json.loads(line), step_map, stats))
        dst = os.path.join(args.sft_dir, f"sft_tbox_{args.split}_{dom}.jsonl")
        with open(dst, "w") as f:
            for r in out_recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        steps = {k.split(":", 1)[1]: v for k, v in stats.items() if k.startswith("step:")}
        print(f"[{dom}] {len(out_recs)} recs, action_turns={stats['action_turns']} "
              f"(-> Plan:step only), msg_only={stats['msg_only']} steps={steps}\n  -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
