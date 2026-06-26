"""Phase 2a — alpha grid-search smoke test.

For each alpha in the grid, run a tau2-bench smoke (N=20, trials=1, max_steps=200)
against the steering server. Steering is encoded in the served model name:

    <base>:<relation>@<alpha>[/L1,L2,L3]
    qwen7b-steer:validates@0.5/12,13,14

alpha=0 => baseline (server skips the hook).

Outputs reports/facet_rft_2026/phase2_steering/alpha_grid_<tag>_<ts>.json
with rows {alpha, pass1, productive_rate, n_completed, wallclock_s, out_dir}.

Usage:
  python _phase2a_alpha_grid.py \
    --steer-url http://127.0.0.1:8200/v1 \
    --steer-base-model qwen7b-steer \
    --user-llm openrouter/openai/gpt-4o-mini \
    --user-base-url https://openrouter.ai/api/v1 \
    --user-api-key $OPENROUTER_KEY \
    --relation validates --layers 12,13,14 \
    --alphas 0.0,0.1,0.3,0.5,1.0,2.0 \
    --n 20 --trials 1 --max-steps 200 \
    --concurrency 3 --per-sim-timeout 600 \
    --tag qwen7b
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


REPO = Path(os.environ.get("BAP_REPO",
    "/home/woori/workspace_common/boltzmann-attention-pi"))
PHASE2 = REPO / "reports/facet_rft_2026/phase2_steering"


def build_model_name(base: str, relation: str, alpha: float, layers: str) -> str:
    if alpha == 0.0:
        return base
    return f"{base}:{relation}@{alpha:g}/{layers}"


def run_one(args, alpha: float) -> dict:
    PHASE2.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"{args.tag}_a{alpha:g}"
    out_dir = PHASE2 / f"alpha_grid_{tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = build_model_name(args.steer_base_model, args.relation, alpha, args.layers)
    (out_dir / "steering.json").write_text(json.dumps(
        {"relation": args.relation, "alpha": alpha,
         "layers": [int(x) for x in args.layers.split(",")],
         "served_model_name": model_name}, indent=2))

    cmd = [
        "python", "scripts/phase1_runner.py",
        "--variants", "B0",
        "--task-split", "base",
        "--base-url", args.steer_url,
        "--agent-llm", f"openai/{model_name}",
        "--user-llm", args.user_llm,
        "--user-base-url", args.user_base_url,
        "--user-api-key", args.user_api_key,
        "--domain", "telecom",
        "--num-trials", str(args.trials),
        "--max-steps", str(args.max_steps),
        "--max-concurrency", str(args.concurrency),
        "--timeout", str(args.per_sim_timeout),
        "--auto-resume",
        "--out-dir", str(out_dir),
    ]
    if args.task_ids_file:
        cmd += ["--task-ids-file", args.task_ids_file]
    else:
        cmd += ["--num-tasks", str(args.n)]
    print(f"[alpha-grid] alpha={alpha} model={model_name}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    dt = time.time() - t0

    # phase1_runner.py writes B0_telecom_base.json/results.json (directory + file)
    results_p = out_dir / "B0_telecom_base.json" / "results.json"
    s = {"pass1": None, "productive_rate": None, "n_completed": 0}
    if results_p.exists():
        try:
            d = json.loads(results_p.read_text())
            sims = d.get("simulations", [])
            completed = [x for x in sims if x.get("end_time")]
            rewards = []
            term_counts = {}
            for x in completed:
                ri = x.get("reward_info") or {}
                r = ri.get("reward") if isinstance(ri, dict) else None
                if r is not None:
                    rewards.append(r)
                tr = x.get("termination_reason")
                term_counts[tr] = term_counts.get(tr, 0) + 1
            n_completed = len(rewards)
            n_pass = sum(1 for r in rewards if r >= 1.0)
            s["n_completed"] = n_completed
            s["pass1"] = (n_pass / n_completed) if n_completed > 0 else None
            s["productive_rate"] = (n_pass / max(args.n * args.trials, 1))
            s["n_pass"] = n_pass
            s["term_counts"] = term_counts
        except Exception as e:
            s["parse_error"] = str(e)
    if proc.returncode != 0:
        s["stderr_tail"] = proc.stderr[-2000:]
    s.update({"alpha": alpha, "model_name": model_name,
              "wallclock_s": dt, "out_dir": str(out_dir),
              "returncode": proc.returncode})
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steer-url", required=True)
    ap.add_argument("--steer-base-model", required=True,
                    help="e.g. qwen7b-steer or hermes3-steer (the --served-model-name)")
    ap.add_argument("--user-llm", required=True)
    ap.add_argument("--user-base-url", required=True)
    ap.add_argument("--user-api-key", required=True)
    ap.add_argument("--relation", default="validates")
    ap.add_argument("--layers", default="12,13,14")
    ap.add_argument("--alphas", default="0.0,0.1,0.3,0.5,1.0,2.0")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--per-sim-timeout", type=int, default=600)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--task-ids-file", default=None,
                    help="JSON file with explicit task IDs (overrides --n)")
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    rows = []
    for a in alphas:
        r = run_one(args, a)
        rows.append(r)
        p1 = r.get("pass1"); pr = r.get("productive_rate")
        print(f"  -> alpha={a} pass1={p1} prod={pr} n={r.get('n_completed')} "
              f"rc={r['returncode']} dt={r['wallclock_s']:.0f}s")

    PHASE2.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = PHASE2 / f"alpha_grid_{args.tag}_{ts}.json"
    out.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
    print(f"\n[alpha-grid] saved {out}")
    print("\nalpha   pass^1  prod%   n  dt(s)")
    for r in rows:
        p1 = r.get("pass1"); pr = r.get("productive_rate")
        print(f"{r['alpha']:5.2f}  "
              f"{(f'{p1:.3f}' if p1 is not None else '----')}  "
              f"{(f'{pr*100:5.1f}' if pr is not None else '----')}  "
              f"{r.get('n_completed', '-'):>3}  {r['wallclock_s']:5.0f}")


if __name__ == "__main__":
    main()
