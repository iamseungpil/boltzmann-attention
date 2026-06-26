#!/usr/bin/env python3
"""build_dpo_dataset.py — Group J contrastive preference pairs for offline RFT (stage b).

Plain SFT imitates teacher SUCCESSES only (positives). The measured failure mode is
WRONG-TOOL fixation + recall-miss at the fix-commitment point (§15.2). DPO needs a
NEGATIVE; Group J `distractor_for` supplies it (the plausible-but-wrong tool a goal
attracts, mined from failures). This builds (prompt, chosen, rejected) pairs at the
first GT-fix commitment point of teacher-success trajectories:

  prompt   = system(+/-policy per --system-mode) + conversation prefix up to the commit
  chosen   = teacher's correct fix tool call (the GT write)
  rejected = same turn but the tool name swapped to a `distractor_for[goal]` wrong tool

Domain-general: GT from tasks.json, distractors from induced/tbox_relations_<domain>.json,
read/write by prefix. TRAIN split only (no test leakage). Multi-domain combinable.

Usage:
  python scripts/distill/build_dpo_dataset.py --domains telecom retail airline \
      --system-mode none --out-dir reports/facet_rft_2026/phase4_distill/dpo_data
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_fix_coverage import is_read  # noqa: E402
from procedure_scorecard import gt_agent_actions  # noqa: E402
from build_sft_dataset import (  # noqa: E402
    convert_messages, build_system_and_tools, _require_tau2, _split_ids,
    iter_teacher_files,
)

DEFAULT_TAU2 = "/home/woori/workspace_common/boltzmann-attention/external/tau2-bench"
ALLOWED_VARIANTS = {"default", "base"}


def reward_of(s):
    return (s.get("reward_info") or {}).get("reward", 0) or 0


def goal_sig(gt_writes):
    return "+".join(sorted(set(gt_writes))) if gt_writes else "(none)"


def load_distractors(induced_dir, domain):
    """goal_sig -> ordered [wrong_tool]; plus a global fallback pool (domain)."""
    path = os.path.join(induced_dir, f"tbox_relations_{domain}.json")
    by_goal, glob_ctr = {}, Counter()
    if not os.path.exists(path):
        print(f"[warn] no induced map {path}; distractor pairs will use global only")
        return by_goal, []
    d = json.load(open(path))
    for entry in d.get("distractor_for", []):
        g = entry["goal"]
        tools = [x["wrong_tool"] for x in entry.get("distractors", [])]
        by_goal[g] = tools
        for x in entry.get("distractors", []):
            glob_ctr[x["wrong_tool"]] += x["count"]
    glob_pool = [t for t, _ in glob_ctr.most_common()]
    return by_goal, glob_pool


def first_fix_index(conv, gt_set):
    """Index of the first assistant turn whose tool_call name is a GT write (fix)."""
    for i, m in enumerate(conv):
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            name = (tc.get("function") or {}).get("name")
            if name and not is_read(name) and name in gt_set:
                return i, name
    return None, None


def make_rejected(chosen_turn, wrong_tool):
    """Clone the chosen assistant turn but swap the (single) tool-call name to wrong_tool."""
    rej = copy.deepcopy(chosen_turn)
    rej["content"] = ""
    tcs = rej.get("tool_calls") or []
    if not tcs:
        return None
    # keep only the first tool call, rename it (args left as-is: the lesson is tool choice)
    tc = tcs[0]
    tc.setdefault("function", {})["name"] = wrong_tool
    rej["tool_calls"] = [tc]
    return rej


def build_for_domain(domain, tau2_root, shipped_dir, induced_dir, system_mode, split):
    registry, SYSTEM_PROMPT, AGENT_INSTRUCTION = _require_tau2(tau2_root)
    system_full, tools = build_system_and_tools(registry, SYSTEM_PROMPT,
                                                AGENT_INSTRUCTION, domain)
    system_content = "" if system_mode == "none" else system_full

    tasks = json.load(open(os.path.join(tau2_root, "data", "tau2", "domains",
                                         domain, "tasks.json")))
    gt_map = {t.get("id", ""): [a["name"] for a in gt_agent_actions(t)] for t in tasks}
    by_goal, glob_pool = load_distractors(induced_dir, domain)
    train_ids = _split_ids(tau2_root, domain, split)

    pairs = []
    stats = Counter()
    for path, teacher, variant in iter_teacher_files(tau2_root, domain):
        if variant not in ALLOWED_VARIANTS:
            continue
        for s in json.load(open(path)).get("simulations", []):
            tid = s.get("task_id", "")
            if tid not in train_ids:
                continue
            if reward_of(s) < 0.999:
                continue
            gt = set(gt_map.get(tid, []))
            if not gt:
                continue
            conv = convert_messages(s.get("messages") or [])
            if conv is None:
                stats["malformed"] += 1
                continue
            k, fix_tool = first_fix_index(conv, gt)
            if k is None:
                stats["no_fix_commit"] += 1
                continue
            g = goal_sig(gt)
            # choose a distractor: goal-specific first, else global, excluding GT tools
            cand = [w for w in by_goal.get(g, []) if w not in gt and w != fix_tool]
            if not cand:
                cand = [w for w in glob_pool if w not in gt and w != fix_tool]
            if not cand:
                stats["no_distractor"] += 1
                continue
            wrong = cand[0]
            chosen = conv[k]
            rejected = make_rejected(chosen, wrong)
            if rejected is None:
                stats["no_toolcall"] += 1
                continue
            prompt = [{"role": "system", "content": system_content}] + conv[:k]
            pairs.append({
                "prompt": prompt, "tools": tools,
                "chosen": chosen, "rejected": rejected,
                "meta": {"domain": domain, "task_id": tid, "teacher": teacher,
                         "goal": g, "chosen_tool": fix_tool, "rejected_tool": wrong,
                         "system_mode": system_mode},
            })
            stats["pairs"] += 1
    return pairs, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["telecom", "retail", "airline"])
    ap.add_argument("--tau2-root", default=DEFAULT_TAU2)
    ap.add_argument("--shipped-dir", default=DEFAULT_TAU2 + "/data/tau2/results/final")
    ap.add_argument("--induced-dir", default="reports/facet_rft_2026/phase4_distill/induced")
    ap.add_argument("--system-mode", default="none", choices=["full", "none"],
                    help="match the arm the DPO refines (none=internalization)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-dir", default="reports/facet_rft_2026/phase4_distill/dpo_data")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    combined = []
    for dom in args.domains:
        pairs, stats = build_for_domain(dom, args.tau2_root, args.shipped_dir,
                                        args.induced_dir, args.system_mode, args.split)
        out = os.path.join(args.out_dir, f"dpo_{args.system_mode}_{dom}_{args.split}.jsonl")
        with open(out, "w") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        combined += pairs
        print(f"[{dom}] pairs={stats['pairs']} "
              f"(no_fix_commit={stats['no_fix_commit']} no_distractor={stats['no_distractor']} "
              f"malformed={stats['malformed']}) -> {out}")

    if len(args.domains) > 1:
        allout = os.path.join(args.out_dir, f"dpo_{args.system_mode}_all_{args.split}.jsonl")
        with open(allout, "w") as f:
            for p in combined:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        rej = Counter(p["meta"]["rejected_tool"] for p in combined)
        print(f"[all] {len(combined)} pairs -> {allout}")
        print(f"  top rejected(distractor) tools: {rej.most_common(8)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
