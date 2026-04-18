"""Canonical AdaSEKA routing diagnostic.

For each of N smoke tasks, capture:
  - steer_mask.sum() (# active mask tokens) and prompt length
  - per-(layer, head) 4-expert coefs (post max-norm)
  - entropy H per (layer, head) query: H = -sum_m p_m log p_m, p_m = |coef_m|/sum|coef|
  - dominant expert argmax frequency

Writes JSON + prints summary.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from canonical_adaseka_engine import (
    CanonicalAdaSEKAEngine,
    build_marker_steer_mask,
)
# reuse chat template + prompt builder + tools from eval_tau2_bench
from eval_tau2_bench import (
    build_chat_prompt_marked,
    build_tools_json,
    extract_domain_tools,
)


def coefs_entropy(coefs: torch.Tensor) -> float:
    abs_c = coefs.abs()
    s = abs_c.sum().item()
    if s <= 1e-12:
        return 0.0
    p = (abs_c / s).cpu().tolist()
    h = 0.0
    for pi in p:
        if pi > 0:
            h -= pi * math.log(pi)
    return h


def diagnose_one_task(
    model,
    tokenizer,
    engine: CanonicalAdaSEKAEngine,
    task: dict,
    tools_json,
    expert_names,
):
    """Return a dict of diagnostics for a single task."""
    prompt_marked = build_chat_prompt_marked(tokenizer, task, tools_json)
    ids, steer_mask = build_marker_steer_mask(prompt_marked, tokenizer)
    ids = ids.to(next(model.parameters()).device)
    steer_mask = steer_mask.to(next(model.parameters()).device)

    prompt_len = int(ids.shape[1])
    mask_sum = int(steer_mask.sum().item())

    # precompute projections — this computes routing coefs per (layer, head)
    # Intercept at _routing_coefficients level
    original_rc = engine._routing_coefficients
    captured = []  # list of dicts: {layer_idx, head_idx, coefs: list[float]}
    L_sel = len(engine.sel_layers)
    n_kv = engine.n_kv

    def spy_rc(q_head, i_layer, h):
        coefs = original_rc(q_head, i_layer, h)
        captured.append({
            "i_layer": int(i_layer),
            "layer": int(engine.sel_layers[i_layer]),
            "head": int(h),
            "coefs": coefs.detach().cpu().tolist(),
        })
        return coefs

    engine._routing_coefficients = spy_rc
    try:
        _ = engine.precompute_projections(ids)
    finally:
        engine._routing_coefficients = original_rc

    # aggregate
    expert_names = list(expert_names)
    n_exp = len(expert_names)
    # per-entry entropy + argmax
    entries = []
    argmax_counts = {n: 0 for n in expert_names}
    for rec in captured:
        coefs_t = torch.tensor(rec["coefs"])
        ent = coefs_entropy(coefs_t)
        am = int(coefs_t.abs().argmax().item())
        argmax_counts[expert_names[am]] += 1
        entries.append({**rec, "entropy": ent, "argmax": expert_names[am]})

    # per-expert aggregate stats (mean, std of |coef|)
    per_expert = {}
    for m, name in enumerate(expert_names):
        vals = [abs(e["coefs"][m]) for e in entries]
        per_expert[name] = {
            "mean_abs": sum(vals) / max(1, len(vals)),
            "max_abs": max(vals) if vals else 0.0,
            "min_abs": min(vals) if vals else 0.0,
            "argmax_freq": argmax_counts[name] / max(1, len(entries)),
        }

    entropies = [e["entropy"] for e in entries]
    max_entropy = math.log(n_exp)
    return {
        "task_id": task.get("id", "?"),
        "prompt_len": prompt_len,
        "mask_sum": mask_sum,
        "mask_ratio": mask_sum / max(1, prompt_len),
        "n_routing_entries": len(entries),
        "entropy_mean": sum(entropies) / max(1, len(entries)),
        "entropy_max": max(entropies) if entropies else 0.0,
        "entropy_min": min(entropies) if entropies else 0.0,
        "max_possible_entropy_log4": max_entropy,
        "argmax_distribution": {k: v / max(1, len(entries)) for k, v in argmax_counts.items()},
        "per_expert_stats": per_expert,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--domain", default="telecom")
    p.add_argument("--dataset", default="external/tau2-bench/data/tau2/domains/telecom/tasks.json")
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--b-ont", default="external/SEKA/seka_projections/ontology-qwen25-7b-tau2-telecom/B_ont.pt")
    p.add_argument("--adaseka-expert-paths",
                   default="external/SEKA/seka_projections/adaseka-qwen25-7b-tau2-telecom/expert_paths.json")
    p.add_argument("--adaseka-layers", default="last10")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--out", default="reports/tau2_2026_04_18/canonical_adaseka_routing_diag.json")
    args = p.parse_args()

    print(f"[load] {args.model} on {args.device}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(args.device)
    model.eval()

    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // n_q
    print(f"[cfg] L={cfg.num_hidden_layers} n_q={n_q} n_kv={n_kv} head_dim={head_dim}")

    engine = CanonicalAdaSEKAEngine(
        model=model,
        expert_paths_json=args.adaseka_expert_paths,
        layers=args.adaseka_layers,
        topk=args.topk,
        temperature=args.T,
        n_kv=n_kv, n_q=n_q, head_dim=head_dim,
    )
    print(f"[engine] experts={engine.expert_names} layers={engine.sel_layers}")

    tasks = json.load(open(args.dataset))[: args.n_samples]
    domain_tools = extract_domain_tools(tasks)
    tools_json = build_tools_json(domain_tools)
    print(f"[data] N={len(tasks)} tasks | {len(domain_tools)} GT tools")

    diags = []
    for i, task in enumerate(tasks):
        d = diagnose_one_task(model, tok, engine, task, tools_json, engine.expert_names)
        diags.append(d)
        print(f"\n[task {i+1}/{len(tasks)}] {d['task_id'][:60]}")
        print(f"  prompt_len={d['prompt_len']}  mask_sum={d['mask_sum']} ({100*d['mask_ratio']:.1f}%)")
        print(f"  entropy mean={d['entropy_mean']:.3f}  max_possible={d['max_possible_entropy_log4']:.3f}")
        print(f"  argmax_distribution: {d['argmax_distribution']}")
        for name, st in d['per_expert_stats'].items():
            print(f"    {name:17s}: mean|c|={st['mean_abs']:.3f}  argmax_freq={st['argmax_freq']:.3f}")

    # aggregate across tasks
    agg_argmax = {n: 0.0 for n in engine.expert_names}
    agg_perexp = {n: {"mean_abs": 0.0, "argmax_freq": 0.0} for n in engine.expert_names}
    agg_ent = 0.0
    agg_mask_ratio = 0.0
    for d in diags:
        for k, v in d["argmax_distribution"].items():
            agg_argmax[k] += v
        for k, st in d["per_expert_stats"].items():
            agg_perexp[k]["mean_abs"] += st["mean_abs"]
            agg_perexp[k]["argmax_freq"] += st["argmax_freq"]
        agg_ent += d["entropy_mean"]
        agg_mask_ratio += d["mask_ratio"]
    N = len(diags)
    for k in agg_argmax: agg_argmax[k] /= N
    for k in agg_perexp:
        agg_perexp[k]["mean_abs"] /= N
        agg_perexp[k]["argmax_freq"] /= N
    agg_ent /= N
    agg_mask_ratio /= N

    print("\n==== AGGREGATE OVER {} TASKS ====".format(N))
    print(f"  avg mask_ratio: {100*agg_mask_ratio:.2f}%  (mean mask tokens / prompt tokens)")
    print(f"  avg entropy per (layer,head) routing: {agg_ent:.3f} (max={math.log(len(engine.expert_names)):.3f})")
    print(f"  avg argmax distribution over experts:")
    for k, v in sorted(agg_argmax.items(), key=lambda x: -x[1]):
        print(f"    {k:17s}: {100*v:5.1f}%")
    print(f"  avg per-expert |coef| magnitude:")
    for k in agg_perexp:
        print(f"    {k:17s}: mean|c|={agg_perexp[k]['mean_abs']:.3f}  argmax_freq={agg_perexp[k]['argmax_freq']:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "model": args.model,
            "n_tasks": N,
            "expert_names": engine.expert_names,
            "layers": engine.sel_layers,
            "n_kv": n_kv,
            "topk": args.topk,
            "T": args.T,
            "aggregate": {
                "mask_ratio_mean": agg_mask_ratio,
                "entropy_mean": agg_ent,
                "max_possible_entropy": math.log(len(engine.expert_names)),
                "argmax_distribution": agg_argmax,
                "per_expert": agg_perexp,
            },
            "per_task": diags,
        }, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
