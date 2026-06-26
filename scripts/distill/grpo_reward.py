#!/usr/bin/env python3
"""grpo_reward.py — verifiable dense reward for facet GRPO (step ③, coworker B3).

Motivation: the 7B student is at pass^1~0.18 -> SPARSE cold-start: most rollouts get
pass=0, so a GRPO group of G rollouts often has all-zero reward -> zero advantage ->
no learning signal. A DENSE, GT-grounded process reward (seq_F1 of the rollout's
agent tool sequence vs the task's ground-truth actions) spreads even all-fail groups,
giving gradient. It is VERIFIABLE (deterministic, computed from tasks.json GT actions;
no LLM judge) — unlike most agent-PRMs.

reward(rollout) = w_pass * pass              # terminal env reward (0/1), dominates
                + w_proc * seq_F1            # dense goal->tool match (recall&order&minimal)
                - w_extra * extra_norm       # over-diagnosis penalty (minimality)
                + w_arg  * arg_bind          # correct entity-arg binding (student-quality)

Anti-hacking: seq_F1 (not recall) -> "call everything" is penalized by precision;
extra_norm penalizes spurious calls; w_pass dominates so true success > any partial.
GRPO group-normalizes, so absolute scale is secondary; the RANKING of rollouts drives
the gradient — and seq_F1/extra/arg_bind differentiate rollouts that all have pass=0.

Use: import compute_reward into the trl GRPO loop; pass the rollout's messages +
the task's GT action names + env pass.  Policy is initialized from the SFT adapter
(full or none mode); rollouts run in tau2 env with gpt-4.1 user_sim.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from procedure_scorecard import (  # noqa: E402
    gt_agent_actions, agent_calls, arg_binding, _lcs_len, _f1, is_read, DEFAULT_TAU2,
)

DEFAULT_WEIGHTS = {"pass": 1.0, "proc": 0.5, "extra": 0.3, "arg": 0.1}


def compute_reward(messages, gt_names, entity_keys, action_params, env_pass,
                   weights=None):
    """Return (reward, breakdown). gt_names = ordered GT agent write-action names."""
    w = weights or DEFAULT_WEIGHTS
    gt_set = set(gt_names)
    actions = [n for n, _ in agent_calls(messages) if not is_read(n)]
    aset = set(actions)
    if gt_names:
        matched = gt_set & aset
        recall = len(matched) / len(gt_set)
        precision = (len(matched) / len(aset)) if aset else 0.0
        lcs = _lcs_len(actions, gt_names)
        seq_match = lcs / len(gt_names)
        seq_prec = (lcs / len(actions)) if actions else 0.0
        seq_f1 = _f1(seq_prec, seq_match)
        extra = len(aset - gt_set)
        extra_norm = extra / (len(gt_set) + 1)            # ~[0,1+]
    else:
        recall = precision = seq_f1 = 0.0
        extra_norm = len(aset)                            # any action when none required = over-action
    ab = arg_binding(messages, entity_keys, action_params)
    arg_bonus = ab if ab is not None else 0.0
    r = (w["pass"] * float(env_pass)
         + w["proc"] * seq_f1
         - w["extra"] * extra_norm
         + w["arg"] * arg_bonus)
    return r, {"pass": float(env_pass), "seq_f1": round(seq_f1, 3),
               "extra_norm": round(extra_norm, 3), "arg_bind": round(arg_bonus, 3),
               "reward": round(r, 3)}


# --- sanity: dense reward should (a) rank success>fail, (b) SPREAD all-fail group ---
def _sanity(domain, tau2_root, shipped_dir, induced_dir):
    import statistics as st
    tasks = json.load(open(os.path.join(tau2_root, "data", "tau2", "domains", domain, "tasks.json")))
    gt_map = {t.get("id", ""): [a["name"] for a in gt_agent_actions(t)] for t in tasks}
    ek, ap = set(), {}
    p = os.path.join(induced_dir, f"param_dataflow_{domain}.json")
    if os.path.exists(p):
        df = json.load(open(p)); ek = set(df.get("entity_keys", [])); ap = df.get("action_params", {})
    succ_r, fail_r, fail_proc = [], [], []
    for f in (glob.glob(os.path.join(shipped_dir, f"*_{domain}_default_*4trials.json")) +
              glob.glob(os.path.join(shipped_dir, f"*_{domain}_base_*4trials.json"))):
        for s in json.load(open(f)).get("simulations", []):
            tid = s.get("task_id", "")
            if tid not in gt_map or not gt_map[tid]:
                continue
            ep = 1.0 if (s.get("reward_info") or {}).get("reward", 0) >= 0.999 else 0.0
            r, bd = compute_reward(s.get("messages") or [], gt_map[tid], ek, ap, ep)
            (succ_r if ep else fail_r).append(r)
            if not ep:
                fail_proc.append(bd["seq_f1"])
    print(f"[{domain}] reward: success mean={st.mean(succ_r):.3f} (n={len(succ_r)}) "
          f"vs failure mean={st.mean(fail_r):.3f} (n={len(fail_r)})")
    print(f"  ALL-FAIL group signal: seq_f1 among failures mean={st.mean(fail_proc):.3f} "
          f"std={st.pstdev(fail_proc):.3f} (std>0 => dense reward gives GRPO gradient on all-fail groups)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="telecom")
    ap.add_argument("--tau2-root", default=DEFAULT_TAU2)
    ap.add_argument("--shipped-dir", default=DEFAULT_TAU2 + "/data/tau2/results/final")
    ap.add_argument("--induced-dir", default="reports/facet_rft_2026/phase4_distill/induced")
    args = ap.parse_args()
    _sanity(args.domain, args.tau2_root, args.shipped_dir, args.induced_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
