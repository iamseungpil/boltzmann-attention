"""B0 deep-dive: error timing, max_steps category breakdown, too_many_errors patterns,
DB match vs reward anomaly, message-length growth.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

TASK_ID_RE = re.compile(r"^\[([^\]]+)\]([^\[]*)\[PERSONA:([^\]]+)\]$")


def parse_task_id(task_id: str):
    m = TASK_ID_RE.match(task_id)
    if not m:
        return task_id, "", "Unknown"
    return m.group(1), m.group(2), m.group(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--run-log", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"loading {args.results}")
    with open(args.results) as f:
        data = json.load(f)
    sims = data["simulations"]

    # 1. max_steps category x persona breakdown
    max_steps_cat = defaultdict(Counter)
    too_many_errors_cat = defaultdict(Counter)
    user_stop_pass_cat = defaultdict(lambda: {"pass": 0, "n": 0})
    for s in sims:
        cat, _, persona = parse_task_id(s["task_id"])
        term = s["termination_reason"]
        if term == "max_steps":
            max_steps_cat[cat][persona] += 1
        if term == "too_many_errors":
            too_many_errors_cat[cat][persona] += 1
        if term == "user_stop":
            user_stop_pass_cat[cat]["n"] += 1
            ri = s.get("reward_info") or {}
            if ri.get("reward", 0.0) >= 1.0:
                user_stop_pass_cat[cat]["pass"] += 1

    # 2. Message count distribution by termination
    msg_by_term = defaultdict(list)
    duration_by_term = defaultdict(list)
    for s in sims:
        msg_by_term[s["termination_reason"]].append(len(s.get("messages") or []))
        duration_by_term[s["termination_reason"]].append(s.get("duration", 0))

    msg_term_stats = {}
    for term, vals in msg_by_term.items():
        vals_sorted = sorted(vals)
        msg_term_stats[term] = {
            "mean": sum(vals) / len(vals) if vals else 0,
            "median": statistics.median(vals) if vals else 0,
            "p95": vals_sorted[int(len(vals_sorted) * 0.95)] if vals_sorted else 0,
            "max": max(vals) if vals else 0,
            "min": min(vals) if vals else 0,
        }

    duration_term_stats = {}
    for term, vals in duration_by_term.items():
        vals_sorted = sorted(vals)
        duration_term_stats[term] = {
            "mean": sum(vals) / len(vals) if vals else 0,
            "median": statistics.median(vals) if vals else 0,
            "p95": vals_sorted[int(len(vals_sorted) * 0.95)] if vals_sorted else 0,
        }

    # 3. DB match vs reward
    db_x_reward = {"match_and_pass": 0, "match_and_fail": 0, "nomatch_and_pass": 0, "nomatch_and_fail": 0,
                   "unchecked_and_pass": 0, "unchecked_and_fail": 0}
    for s in sims:
        ri = s.get("reward_info")
        if not ri:
            continue
        r = ri.get("reward", 0.0)
        passed = r >= 1.0
        db = ri.get("db_check") or {}
        if "db_match" not in db:
            key = "unchecked_and_pass" if passed else "unchecked_and_fail"
        elif db["db_match"]:
            key = "match_and_pass" if passed else "match_and_fail"
        else:
            key = "nomatch_and_pass" if passed else "nomatch_and_fail"
        db_x_reward[key] += 1

    # 4. Action checks: how many actions per task, how often satisfied
    action_stats = []
    env_assertion_stats = []
    for s in sims:
        ri = s.get("reward_info")
        if not ri:
            continue
        acts = ri.get("action_checks") or []
        action_stats.append(len(acts))
        evs = ri.get("env_assertions") or []
        env_assertion_stats.append(len(evs))

    # 5. Trial-level pattern per task: who passes which trial?
    trial_x_task = defaultdict(dict)  # task_id -> trial -> reward
    for s in sims:
        ri = s.get("reward_info") or {}
        trial_x_task[s["task_id"]][s["trial"]] = (ri.get("reward", 0.0), s["termination_reason"])

    # Tasks where some trials pass but not others (interesting variance)
    inconsistent_tasks = []
    consistent_pass_tasks = []
    consistent_fail_tasks = 0
    for tid, trials in trial_x_task.items():
        rewards = [r for r, _ in trials.values()]
        if all(r >= 1.0 for r in rewards):
            consistent_pass_tasks.append(tid)
        elif all(r < 1.0 for r in rewards):
            consistent_fail_tasks += 1
        else:
            inconsistent_tasks.append((tid, rewards, [t for _, t in trials.values()]))

    # 6. Error-line analysis from run.log (within B0 region = up to line 184875)
    print(f"parsing {args.run_log} (B0 region)")
    ctx_window_lines = []
    conn_err_lines = []
    too_many_err_lines = []
    unknown_tool_lines = []
    B0_MAX = 184875
    with open(args.run_log, "r", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            if i > B0_MAX:
                break
            if "ContextWindowExceededError" in line and "failed after 4 attempts" in line:
                ctx_window_lines.append((i, line.strip()[:300]))
            elif "InternalServerError" in line and "Connection error" in line and "failed after 4 attempts" in line:
                conn_err_lines.append((i, line.strip()[:300]))
            elif "Unknown tool" in line and "failed after" in line:
                unknown_tool_lines.append((i, line.strip()[:300]))

    # Earliest/latest timestamps for each error type
    def first_last(lines):
        if not lines:
            return None, None
        return lines[0], lines[-1]

    out = {
        "max_steps_by_category_x_persona": {cat: dict(c) for cat, c in max_steps_cat.items()},
        "too_many_errors_by_category_x_persona": {cat: dict(c) for cat, c in too_many_errors_cat.items()},
        "user_stop_pass_rate_by_category": {cat: {**v, "pass_rate": v["pass"] / v["n"] if v["n"] else 0}
                                            for cat, v in user_stop_pass_cat.items()},
        "msg_count_by_termination": msg_term_stats,
        "duration_by_termination": duration_term_stats,
        "db_check_x_reward": db_x_reward,
        "action_checks": {
            "mean_per_sim": sum(action_stats) / len(action_stats) if action_stats else 0,
            "max": max(action_stats) if action_stats else 0,
            "zero_actions_n": sum(1 for a in action_stats if a == 0),
        },
        "env_assertions": {
            "mean_per_sim": sum(env_assertion_stats) / len(env_assertion_stats) if env_assertion_stats else 0,
            "max": max(env_assertion_stats) if env_assertion_stats else 0,
            "zero_n": sum(1 for a in env_assertion_stats if a == 0),
        },
        "trial_consistency": {
            "consistent_pass_4_of_4": len(consistent_pass_tasks),
            "consistent_fail_0_of_4": consistent_fail_tasks,
            "inconsistent": len(inconsistent_tasks),
            "consistent_pass_task_ids": consistent_pass_tasks[:10],
            "inconsistent_sample": [
                {"task_id": tid, "rewards": rs, "terminations": ts}
                for tid, rs, ts in inconsistent_tasks[:15]
            ],
        },
        "error_log": {
            "ctx_window_count": len(ctx_window_lines),
            "ctx_window_first": ctx_window_lines[0] if ctx_window_lines else None,
            "ctx_window_last": ctx_window_lines[-1] if ctx_window_lines else None,
            "connection_count": len(conn_err_lines),
            "connection_first": conn_err_lines[0] if conn_err_lines else None,
            "connection_last": conn_err_lines[-1] if conn_err_lines else None,
            "unknown_tool_count": len(unknown_tool_lines),
            "unknown_tool_first": unknown_tool_lines[0] if unknown_tool_lines else None,
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
