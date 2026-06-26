#!/usr/bin/env python3
"""build_abstract_sft.py — inject abstract plan-step prefixes into SFT data (TBox-isolation B).

Reads sft_plain_train_<domain>.jsonl + induced/step_realization_<domain>.json and, for
every assistant turn that makes a tool call, sets the turn's content to
"Plan: <abstract_step>" (the domain-general PLAN_STEP_VOCAB role for that tool). The
model thus learns to emit the domain-INVARIANT plan-step as a reasoning prefix BEFORE
the concrete (domain-specific, swappable) tool call. At transfer the plan-step policy
carries; the concrete realization adapts to the target domain's tools in context.

Output: sft_abstract_train_<domain>.jsonl (+ combined _all). Train with the existing
lora_train_chat_toolcall.py (assistant-only mask covers prefix + call).

Usage:
  python scripts/distill/build_abstract_sft.py --domains telecom retail airline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SFT_DIR = "reports/facet_rft_2026/phase4_distill/sft_data"
INDUCED_DIR = "reports/facet_rft_2026/phase4_distill/induced"


def load_realization(domain):
    p = os.path.join(INDUCED_DIR, f"step_realization_{domain}.json")
    if not os.path.exists(p):
        raise SystemExit(f"[err] missing {p} — run induce_step_realization.py --domain {domain} first")
    return json.load(open(p)).get("step_realizes_tool", {})


def inject(record, realize, stats):
    """Set assistant tool-call turn content to 'Plan: <step>' (prefix the concrete call)."""
    for m in record.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls") or []
        if not tcs:
            continue
        name = (tcs[0].get("function") or {}).get("name") or tcs[0].get("name")
        step = realize.get(name)
        if not step:
            stats["unmapped_tool"] += 1
            step = "apply_policy_action"  # safe default
        prefix = f"Plan: {step}"
        cur = m.get("content") or ""
        m["content"] = prefix if not cur.strip() else f"{prefix}\n{cur}"
        stats[f"step:{step}"] += 1
        stats["turns_injected"] += 1
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["telecom", "retail", "airline"])
    ap.add_argument("--sft-dir", default=SFT_DIR)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    combined = []
    for dom in args.domains:
        realize = load_realization(dom)
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
                rec = json.loads(line)
                out_recs.append(inject(rec, realize, stats))
        dst = os.path.join(args.sft_dir, f"sft_abstract_{args.split}_{dom}.jsonl")
        with open(dst, "w") as f:
            for r in out_recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        combined += out_recs
        steps = {k.split(":", 1)[1]: v for k, v in stats.items() if k.startswith("step:")}
        print(f"[{dom}] {len(out_recs)} recs, {stats['turns_injected']} turns injected, "
              f"unmapped={stats['unmapped_tool']} steps={steps} -> {dst}")

    if len(args.domains) > 1 and combined:
        allout = os.path.join(args.sft_dir, f"sft_abstract_{args.split}_all.jsonl")
        with open(allout, "w") as f:
            for r in combined:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[all] {len(combined)} recs -> {allout}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
