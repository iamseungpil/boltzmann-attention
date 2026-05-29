#!/usr/bin/env python3
"""procedure_scorecard.py — multi-axis goal->tool procedure quality (extends fix-coverage).

fix-coverage measures only RECALL (did the agent call the required tools). The
goal->tool procedure is richer; this scores 5 axes against the benchmark ground
truth (tasks.json evaluation_criteria.actions, which is an ORDERED list with
requestor / arguments / compare_args):

  recall       |GT_req ∩ called| / |GT_req|                  (what to call)   = fix-coverage
  precision    |GT_req ∩ called_actions| / |called_actions|  (minimality / no over-action)
  call_eff     |GT_req| / (#agent action calls incl. repeats) (call efficiency)
  order        pairwise agreement of called-required tools vs GT order (Kendall-ish)
  arg_match    matched required tools whose agent args match GT args (compare_args keys)
  repeat       #non-idempotent, non-loop-capable tools called >1x (waste)  [telecom ontology]

★FIX vs the earlier GT-action coverage: only requestor=='assistant' GT actions are
required of the AGENT (telecom is dual-control: toggle_airplane_mode etc. are USER
actions). Earlier task_required_tools counted user actions too -> understated recall.

Usage:
  python scripts/distill/procedure_scorecard.py --domain telecom            # shipped
  python scripts/distill/procedure_scorecard.py --domain telecom --results '<glob>'
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_fix_coverage import is_read, _shipped_files  # noqa: E402
from ontology_filter import _load_ont  # noqa: E402

DEFAULT_TAU2 = "/home/woori/workspace_common/boltzmann-attention/external/tau2-bench"
ONT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ontology"))


def gt_agent_actions(task):
    """Ordered list of AGENT-required actions: [{name, args, compare_args}] (requestor==assistant)."""
    ec = task.get("evaluation_criteria") or {}
    out = []
    for a in ec.get("actions") or []:
        # exclude only explicit USER actions (telecom dual-control). assistant or
        # requestor=None (single-control retail/airline) are the agent's.
        if a.get("requestor") == "user":
            continue
        name = a.get("name") or a.get("func_name")
        if not name or is_read(name):   # only state-changing (write) actions are the goal->tool target
            continue
        out.append({"name": name, "args": a.get("arguments") or {},
                    "compare_args": a.get("compare_args")})
    return out


def agent_calls(messages):
    """Ordered agent tool calls: [(name, args_dict)] for requestor==assistant."""
    out = []
    for m in messages or []:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if tc.get("requestor", "assistant") == "assistant" and tc.get("name"):
                    out.append((tc["name"], tc.get("arguments") or {}))
    return out


def _lcs_len(a, b):
    """Length of longest common subsequence (order-preserving) of two name sequences."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        ai = a[i - 1]
        for j in range(1, n + 1):
            cur[j] = prev[j - 1] + 1 if ai == b[j - 1] else (prev[j] if prev[j] >= cur[j - 1] else cur[j - 1])
        prev = cur
    return prev[n]


def _scalars(obj, out, depth=0):
    if depth > 4:
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _scalars(v, out, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            _scalars(v, out, depth + 1)
    elif isinstance(obj, (str, int)) and obj != "":
        out.add(str(obj))


def arg_binding(messages, entity_keys, action_params):
    """Data-flow arg-correctness: of the entity-key params an agent action consumes,
    fraction whose value was RETRIEVED from a PRIOR read response (provenance) and
    non-empty. Replaces literal arg match (not reward-aligned)."""
    if not entity_keys or not action_params:
        return None
    id2name = {}
    for m in messages or []:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                id2name[tc.get("id")] = tc.get("name")
    known, total, bound = set(), 0, 0
    for m in messages or []:
        r = m.get("role")
        if r == "assistant":
            for tc in m.get("tool_calls") or []:
                if tc.get("requestor", "assistant") != "assistant":
                    continue
                name = tc.get("name")
                if not name or is_read(name):
                    continue
                args = tc.get("arguments") or {}
                for p in action_params.get(name, []):
                    if p not in entity_keys:
                        continue
                    v = args.get(p)
                    if v in (None, "", [], {}):
                        continue
                    total += 1
                    if str(v) in known:
                        bound += 1
        elif r == "tool":
            caller = id2name.get(m.get("id"))
            if caller and is_read(caller):
                c = m.get("content")
                try:
                    d = json.loads(c) if isinstance(c, str) else c
                except Exception:
                    d = None
                _scalars(d, known)
    return (bound / total) if total else None


def score_trajectory(sim, gt_map, idem, loopcap, entity_keys=None, action_params=None):
    tid = sim.get("task_id", "")
    gt = gt_map.get(tid, [])
    if not gt:
        return None  # no agent-required actions (e.g. airline info tasks) -> skip
    gt_names = [a["name"] for a in gt]
    gt_set = set(gt_names)
    calls = agent_calls(sim.get("messages") or [])
    action_calls = [(n, a) for n, a in calls if not is_read(n)]   # state-changing only
    called_actions = [n for n, _ in action_calls]
    called_set = set(called_actions)
    first_idx = {}
    for i, n in enumerate(called_actions):
        first_idx.setdefault(n, i)

    matched = gt_set & called_set
    recall = len(matched) / len(gt_set)
    precision = (len(matched) / len(called_set)) if called_set else None
    call_eff = (len(gt_set) / len(action_calls)) if action_calls else None  # <=1 ideal=1

    # order: pairwise agreement among called required tools vs GT order
    order_list = [n for n in gt_names if n in first_idx]
    pairs = ok = 0
    for i in range(len(order_list)):
        for j in range(i + 1, len(order_list)):
            pairs += 1
            if first_idx[order_list[i]] < first_idx[order_list[j]]:
                ok += 1
    order = (ok / pairs) if pairs else None

    # arg binding (data-flow provenance): entity params bound from prior reads
    arg_bind = arg_binding(sim.get("messages") or [], entity_keys or set(), action_params or {})

    # sequence match (order-aware): LCS of agent vs GT write-action order
    lcs = _lcs_len(called_actions, gt_names)
    seq_match = lcs / len(gt_names)                      # order-preserving recall
    seq_prec = (lcs / len(called_actions)) if called_actions else None

    # repeat misuse (telecom ontology)
    cnt = Counter(called_actions)
    repeat = sum(1 for t, c in cnt.items()
                 if c > 1 and idem.get(t) is False and loopcap.get(t) is False)

    return {"recall": recall, "precision": precision, "seq_match": seq_match,
            "seq_prec": seq_prec, "call_eff": call_eff, "order": order,
            "arg_bind": arg_bind, "repeat": float(repeat)}


def _agg(vals):
    vals = [v for v in vals if v is not None]
    return round(st.mean(vals), 3) if vals else None


def run(sims, gt_map, idem, loopcap, label, entity_keys=None, action_params=None):
    buckets = {"succ": [], "fail": []}
    for s in sims:
        sc = score_trajectory(s, gt_map, idem, loopcap, entity_keys, action_params)
        if sc is None:
            continue
        cls = "succ" if (s.get("reward_info") or {}).get("reward", 0) >= 0.999 else "fail"
        buckets[cls].append(sc)
    axes = ["recall", "precision", "seq_match", "seq_prec", "call_eff", "order", "arg_bind", "repeat"]
    print(f"\n[{label}] n_scored: succ={len(buckets['succ'])} fail={len(buckets['fail'])}")
    print(f"  {'axis':10} {'success':>9} {'failure':>9} {'disc(s-f)':>10}")
    for ax in axes:
        s = _agg([d[ax] for d in buckets["succ"]])
        f = _agg([d[ax] for d in buckets["fail"]])
        disc = round(s - f, 3) if (s is not None and f is not None) else None
        print(f"  {ax:10} {str(s):>9} {str(f):>9} {str(disc):>10}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["telecom", "retail", "airline"])
    ap.add_argument("--tau2-root", default=DEFAULT_TAU2)
    ap.add_argument("--shipped-dir", default=DEFAULT_TAU2 + "/data/tau2/results/final")
    ap.add_argument("--results", nargs="*", default=None)
    ap.add_argument("--induced-dir", default="reports/facet_rft_2026/phase4_distill/induced",
                    help="dir with param_dataflow_<domain>.json (for arg_bind)")
    args = ap.parse_args()

    tasks = json.load(open(os.path.join(args.tau2_root, "data", "tau2", "domains",
                                        args.domain, "tasks.json")))
    gt_map = {t.get("id", ""): gt_agent_actions(t) for t in tasks}
    n_with = sum(1 for v in gt_map.values() if v)
    print(f"[gt] {args.domain}: {len(gt_map)} tasks, {n_with} with agent-required actions "
          f"(mean {sum(len(v) for v in gt_map.values())/max(n_with,1):.2f} actions)")

    try:
        ont = _load_ont(args.domain, ONT_DIR)
        idem = getattr(ont, "IDEMPOTENT", {}); loopcap = getattr(ont, "LOOP_CAPABLE", {})
    except Exception:
        idem = loopcap = {}

    entity_keys, action_params = set(), {}
    df_path = os.path.join(args.induced_dir, f"param_dataflow_{args.domain}.json")
    if os.path.exists(df_path):
        df = json.load(open(df_path))
        entity_keys = set(df.get("entity_keys", []))
        action_params = df.get("action_params", {})
        print(f"[dataflow] entity_keys={sorted(entity_keys)} (arg_bind enabled)")
    else:
        print(f"[dataflow] {df_path} not found -> arg_bind disabled")

    if not args.results:
        sims = []
        for f in _shipped_files(args.domain, args.shipped_dir):
            sims += json.load(open(f)).get("simulations", [])
        run(sims, gt_map, idem, loopcap, f"shipped {args.domain}", entity_keys, action_params)
        return 0
    for r in args.results:
        for p in glob.glob(r):
            try:
                sims = json.load(open(p)).get("simulations", [])
            except Exception as e:
                print(f"[skip] {p}: {e}"); continue
            run(sims, gt_map, idem, loopcap, os.path.basename(os.path.dirname(p)) or p,
                entity_keys, action_params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
