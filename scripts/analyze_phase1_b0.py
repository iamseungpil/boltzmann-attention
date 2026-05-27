"""B0 Vanilla baseline thorough analysis (Phase 1 baseline N=114 trials=4).

Reads B0 results.json, produces:
  1) JSON metrics (analysis.json) — machine-readable
  2) Markdown report (analysis.md) — human-readable

pass^k formula (tau2-bench convention):
  pass^k(task) = 1 if at least one of k trials achieves reward >= threshold
  default threshold = 1.0 (strict; tau2-bench standard)

Wilson 95% CI is computed for pass^1 over the evaluated set.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


TASK_ID_RE = re.compile(r"^\[([^\]]+)\]([^\[]*)\[PERSONA:([^\]]+)\]$")


def parse_task_id(task_id: str):
    """Return (category, body, persona) or (raw, '', 'Unknown') on parse failure."""
    m = TASK_ID_RE.match(task_id)
    if not m:
        return task_id, "", "Unknown"
    return m.group(1), m.group(2), m.group(3)


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for binomial proportion (k successes out of n)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="path to results.json")
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--label", default="B0", help="label used in report headings")
    ap.add_argument("--pass-threshold", type=float, default=1.0,
                    help="reward threshold for pass^k (default 1.0 = fully correct)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.label}] loading {args.results} ...")
    with open(args.results) as f:
        data = json.load(f)

    info = data["info"]
    sims = data["simulations"]
    tasks = {t["id"]: t for t in data["tasks"]}
    print(f"  tasks={len(tasks)} sims={len(sims)} trials={info.get('num_trials')} max_steps={info.get('max_steps')}")

    # --- 1. Termination distribution ---
    term_counter = Counter(s.get("termination_reason") for s in sims)

    # --- 2. Reward stats ---
    rewards_all = [s["reward_info"]["reward"] for s in sims]
    evaluated = [s for s in sims if s.get("termination_reason") != "infrastructure_error"]
    rewards_eval = [s["reward_info"]["reward"] for s in evaluated]
    full_credit_eval = sum(1 for r in rewards_eval if r >= args.pass_threshold)
    p_eval, lo_eval, hi_eval = wilson_ci(full_credit_eval, len(rewards_eval))

    # Average reward (continuous, tau2 standard)
    avg_reward_eval = mean(rewards_eval)
    avg_reward_all = mean(rewards_all)

    # --- 3. pass^k task-level (at least 1 trial reaches threshold) ---
    task_rewards = defaultdict(list)  # task_id -> list of (trial, reward, terminated)
    for s in sims:
        task_rewards[s["task_id"]].append((s["trial"], s["reward_info"]["reward"], s["termination_reason"]))

    pass_k = {}
    pass_k_counts = {}
    K = info.get("num_trials", 4)
    for k in range(1, K + 1):
        succ_tasks = 0
        for tid, trials in task_rewards.items():
            # Sort by trial id, take first k
            sorted_trials = sorted(trials, key=lambda x: x[0])[:k]
            if any(r >= args.pass_threshold and term != "infrastructure_error" for _, r, term in sorted_trials):
                succ_tasks += 1
        pass_k[k] = succ_tasks / len(task_rewards)
        pass_k_counts[k] = succ_tasks

    # --- 4. Per-task category breakdown ---
    cat_rewards = defaultdict(list)
    cat_terms = defaultdict(Counter)
    persona_rewards = defaultdict(list)
    for s in sims:
        cat, _body, persona = parse_task_id(s["task_id"])
        cat_rewards[cat].append(s["reward_info"]["reward"])
        cat_terms[cat][s["termination_reason"]] += 1
        persona_rewards[persona].append((s["reward_info"]["reward"], s["termination_reason"]))

    cat_summary = {}
    for cat, rs in cat_rewards.items():
        cat_summary[cat] = {
            "n_sims": len(rs),
            "avg_reward": mean(rs),
            "termination": dict(cat_terms[cat]),
        }

    persona_summary = {}
    for persona, items in persona_rewards.items():
        evals = [r for r, term in items if term != "infrastructure_error"]
        persona_summary[persona] = {
            "n_total": len(items),
            "n_evaluated": len(evals),
            "avg_reward_evaluated": mean(evals),
        }

    # --- 5. Trial variance per task ---
    trial_stds = []
    trial_means = []
    for tid, trials in task_rewards.items():
        rs = [r for _, r, term in trials if term != "infrastructure_error"]
        if len(rs) >= 2:
            trial_stds.append(statistics.stdev(rs))
            trial_means.append(mean(rs))
    avg_trial_std = mean(trial_stds)

    # --- 6. Duration distribution ---
    durations = [s["duration"] for s in sims]
    eval_durations = [s["duration"] for s in evaluated]
    duration_stats = {
        "mean_all": mean(durations),
        "median_all": statistics.median(durations) if durations else 0,
        "p95_all": sorted(durations)[int(len(durations) * 0.95)] if durations else 0,
        "min_all": min(durations) if durations else 0,
        "max_all": max(durations) if durations else 0,
        "mean_evaluated": mean(eval_durations),
    }

    # --- 7. Termination vs reward cross-tab ---
    term_reward_xtab = defaultdict(lambda: {"n": 0, "sum_reward": 0.0, "n_pass": 0})
    for s in sims:
        term = s.get("termination_reason")
        r = s["reward_info"]["reward"]
        term_reward_xtab[term]["n"] += 1
        term_reward_xtab[term]["sum_reward"] += r
        if r >= args.pass_threshold:
            term_reward_xtab[term]["n_pass"] += 1
    for term, stats in term_reward_xtab.items():
        stats["avg_reward"] = stats["sum_reward"] / stats["n"] if stats["n"] else 0
        stats["pass_rate"] = stats["n_pass"] / stats["n"] if stats["n"] else 0
    term_reward_xtab = dict(term_reward_xtab)

    # --- 8. Reward basis breakdown ---
    rb_counter = Counter()
    db_match_evaluated = 0
    db_total_evaluated = 0
    for s in evaluated:
        ri = s["reward_info"]
        for rb in ri.get("reward_basis") or []:
            rb_counter[rb] += 1
        db = ri.get("db_check") or {}
        if "db_match" in db:
            db_total_evaluated += 1
            if db["db_match"]:
                db_match_evaluated += 1

    # --- 9. Top-N failing tasks (lowest avg reward over 4 trials) ---
    task_avg = {tid: mean([r for _, r, _ in trials]) for tid, trials in task_rewards.items()}
    task_avg_sorted = sorted(task_avg.items(), key=lambda x: x[1])
    worst_tasks = task_avg_sorted[:10]
    best_tasks = [(tid, avg) for tid, avg in task_avg_sorted if avg > 0][-10:][::-1]

    # --- 10. Hallucination / retry stats ---
    halluc = [s.get("hallucination_retries_used", 0) for s in sims]
    halluc_total = sum(halluc)
    halluc_nonzero = sum(1 for h in halluc if h > 0)

    # --- 11. Message length stats ---
    msg_lens = [len(s.get("messages") or []) for s in sims]
    msg_stats = {
        "mean": mean(msg_lens),
        "median": statistics.median(msg_lens) if msg_lens else 0,
        "p95": sorted(msg_lens)[int(len(msg_lens) * 0.95)] if msg_lens else 0,
        "min": min(msg_lens) if msg_lens else 0,
        "max": max(msg_lens) if msg_lens else 0,
    }

    # --- assemble metrics ---
    metrics = {
        "label": args.label,
        "results_path": args.results,
        "info": {
            "num_trials": info.get("num_trials"),
            "max_steps": info.get("max_steps"),
            "max_errors": info.get("max_errors"),
            "seed": info.get("seed"),
            "git_commit": info.get("git_commit"),
            "agent_info": info.get("agent_info"),
            "user_info": info.get("user_info"),
        },
        "counts": {
            "tasks": len(tasks),
            "simulations": len(sims),
            "evaluated": len(evaluated),
            "infrastructure_errors": term_counter.get("infrastructure_error", 0),
            "too_many_errors": term_counter.get("too_many_errors", 0),
        },
        "termination": dict(term_counter),
        "reward": {
            "avg_reward_evaluated": avg_reward_eval,
            "avg_reward_all": avg_reward_all,
            "full_credit_evaluated_count": full_credit_eval,
            "full_credit_evaluated_rate": p_eval,
            "wilson_95_ci_low": lo_eval,
            "wilson_95_ci_high": hi_eval,
            "pass_threshold": args.pass_threshold,
        },
        "pass_k": pass_k,
        "pass_k_counts": pass_k_counts,
        "trial_variance": {
            "mean_per_task_std": avg_trial_std,
            "n_tasks_with_variance": len(trial_stds),
        },
        "duration_seconds": duration_stats,
        "termination_x_reward": term_reward_xtab,
        "reward_basis_counts": dict(rb_counter),
        "db_match_evaluated": {
            "matched": db_match_evaluated,
            "total": db_total_evaluated,
            "rate": db_match_evaluated / db_total_evaluated if db_total_evaluated else 0,
        },
        "categories": cat_summary,
        "personas": persona_summary,
        "worst_tasks_by_avg_reward": [
            {"task_id": tid, "avg_reward": r} for tid, r in worst_tasks
        ],
        "best_tasks_by_avg_reward": [
            {"task_id": tid, "avg_reward": r} for tid, r in best_tasks
        ],
        "hallucination": {
            "total_retries": halluc_total,
            "sims_with_retries": halluc_nonzero,
        },
        "message_count_per_sim": msg_stats,
    }

    json_path = out_dir / f"{args.label}_analysis.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  wrote {json_path}")

    # --- markdown render ---
    md = render_markdown(metrics)
    md_path = out_dir / f"{args.label}_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  wrote {md_path}")


def render_markdown(m: dict) -> str:
    L = m["label"]
    c = m["counts"]
    r = m["reward"]
    t = m["termination"]
    pk = m["pass_k"]
    pkc = m["pass_k_counts"]
    dur = m["duration_seconds"]
    info = m["info"]

    lines = []
    lines.append(f"# {L} Vanilla Baseline — Phase 1 N={c['tasks']} Trials={info['num_trials']} Analysis")
    lines.append("")
    lines.append(f"- results file: `{m['results_path']}`")
    lines.append(f"- git_commit: `{info.get('git_commit')}`")
    lines.append(f"- agent: `{(info.get('agent_info') or {}).get('llm', '?')}` · max_steps={info['max_steps']} · seed={info['seed']}")
    lines.append("")

    lines.append("## 1. Headline metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Total simulations | {c['simulations']} |")
    lines.append(f"| Evaluated (exclude infra errors) | {c['evaluated']} |")
    lines.append(f"| Infrastructure errors | {c['infrastructure_errors']} |")
    lines.append(f"| **Avg reward (evaluated)** | **{r['avg_reward_evaluated']:.4f}** |")
    lines.append(f"| Avg reward (all, infra=0) | {r['avg_reward_all']:.4f} |")
    lines.append(f"| Full-credit rate (reward≥{r['pass_threshold']}) | {r['full_credit_evaluated_count']}/{c['evaluated']} = {r['full_credit_evaluated_rate']:.4f} |")
    lines.append(f"| Wilson 95% CI (full-credit) | [{r['wilson_95_ci_low']:.4f}, {r['wilson_95_ci_high']:.4f}] |")
    lines.append("")

    lines.append("## 2. pass^k (task-level)")
    lines.append("")
    lines.append("\"Pass\" = at least 1 of k trials achieves reward ≥ threshold and is not an infrastructure error.")
    lines.append("")
    lines.append("| k | passed tasks | pass^k |")
    lines.append("|---|---|---|")
    for k in sorted(pk.keys()):
        lines.append(f"| {k} | {pkc[k]}/{c['tasks']} | {pk[k]:.4f} |")
    lines.append("")

    lines.append("## 3. Termination breakdown")
    lines.append("")
    lines.append("| Termination | Count | Share |")
    lines.append("|---|---|---|")
    total = c['simulations']
    for term, n in sorted(t.items(), key=lambda x: -x[1]):
        lines.append(f"| {term} | {n} | {n/total:.3f} |")
    lines.append("")

    lines.append("## 4. Termination × Reward cross-tab")
    lines.append("")
    lines.append("| Termination | n | avg_reward | pass_rate (reward≥1) |")
    lines.append("|---|---|---|---|")
    for term, st in sorted(m["termination_x_reward"].items(), key=lambda x: -x[1]["n"]):
        lines.append(f"| {term} | {st['n']} | {st['avg_reward']:.4f} | {st['pass_rate']:.4f} |")
    lines.append("")

    lines.append("## 5. Persona breakdown")
    lines.append("")
    lines.append("| Persona | n_total | n_evaluated | avg_reward (evaluated) |")
    lines.append("|---|---|---|---|")
    for persona, st in sorted(m["personas"].items()):
        lines.append(f"| {persona} | {st['n_total']} | {st['n_evaluated']} | {st['avg_reward_evaluated']:.4f} |")
    lines.append("")

    lines.append("## 6. Per-category breakdown (task_id prefix)")
    lines.append("")
    lines.append("| Category | n_sims | avg_reward | top termination |")
    lines.append("|---|---|---|---|")
    cats = sorted(m["categories"].items(), key=lambda x: -x[1]["avg_reward"])
    for cat, st in cats:
        top_term = max(st["termination"].items(), key=lambda x: x[1])[0] if st["termination"] else "?"
        lines.append(f"| {cat} | {st['n_sims']} | {st['avg_reward']:.4f} | {top_term} |")
    lines.append("")

    lines.append("## 7. Trial variance (within-task across 4 trials)")
    lines.append("")
    tv = m["trial_variance"]
    lines.append(f"- Mean per-task reward std: **{tv['mean_per_task_std']:.4f}**")
    lines.append(f"- Tasks with ≥2 evaluated trials: {tv['n_tasks_with_variance']}/{c['tasks']}")
    lines.append("")

    lines.append("## 8. Duration (seconds per simulation)")
    lines.append("")
    lines.append(f"- mean={dur['mean_all']:.1f}, median={dur['median_all']:.1f}, p95={dur['p95_all']:.1f}")
    lines.append(f"- min={dur['min_all']:.1f}, max={dur['max_all']:.1f}")
    lines.append(f"- evaluated only mean={dur['mean_evaluated']:.1f}")
    lines.append("")

    lines.append("## 9. Message count per simulation")
    lines.append("")
    ms = m["message_count_per_sim"]
    lines.append(f"- mean={ms['mean']:.1f}, median={ms['median']:.1f}, p95={ms['p95']:.1f}, max={ms['max']}")
    lines.append("")

    lines.append("## 10. Reward basis distribution")
    lines.append("")
    rb = m["reward_basis_counts"]
    total_rb = sum(rb.values())
    for k, v in sorted(rb.items(), key=lambda x: -x[1]):
        lines.append(f"- {k}: {v} ({v/total_rb:.3f})" if total_rb else f"- {k}: {v}")
    lines.append("")
    db = m["db_match_evaluated"]
    lines.append(f"- DB match (evaluated): {db['matched']}/{db['total']} = {db['rate']:.4f}")
    lines.append("")

    lines.append("## 11. Worst 10 tasks by avg reward (across 4 trials)")
    lines.append("")
    for x in m["worst_tasks_by_avg_reward"]:
        tid = x["task_id"][:120] + ("..." if len(x["task_id"]) > 120 else "")
        lines.append(f"- `{tid}` → avg_reward={x['avg_reward']:.4f}")
    lines.append("")

    lines.append("## 12. Best 10 tasks (avg reward > 0)")
    lines.append("")
    if m["best_tasks_by_avg_reward"]:
        for x in m["best_tasks_by_avg_reward"]:
            tid = x["task_id"][:120] + ("..." if len(x["task_id"]) > 120 else "")
            lines.append(f"- `{tid}` → avg_reward={x['avg_reward']:.4f}")
    else:
        lines.append("- none (all tasks have avg_reward = 0)")
    lines.append("")

    lines.append("## 13. Hallucination retries")
    lines.append("")
    h = m["hallucination"]
    lines.append(f"- Total retries used: {h['total_retries']}")
    lines.append(f"- Sims with ≥1 retry: {h['sims_with_retries']}/{c['simulations']}")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
