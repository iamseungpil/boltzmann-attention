#!/usr/bin/env python3
"""B2 — Layer-resolved KL: is non-monotonic tool-name KL an FFN/LM-head artifact?

For each Haar-orthonormal random perturbation rank r, we project every
layer's last-token residual through the model's unembed (lm_head + final
RMS norm) to obtain a "layer-local logit distribution".  We then compute
KL(p_perturbed || p_baseline) at every layer 0..L-1 and at the final
LM-head output.

Hypothesis H-D: attention-layer KL (early residual) is monotonic in r
(matching Lemma 2's √(r/d) scaling), while non-monotonic shape emerges
only after FFN + LM-head composition.

Output:
    reports/new_theorem_test/phase_b2_layer_kl.json

Per-layer view: {rank: {layer: {kl_mean, kl_std}}, "final": {rank: kl}}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from canonical_adaseka_engine import build_marker_steer_mask
from eval_tau2_bench import (
    build_chat_prompt_marked,
    build_tools_json,
    extract_domain_tools,
)
from measure_lemma_empirical import (
    parse_layers_spec,
    build_random_projection,
    make_kbias_hook,
)


@torch.no_grad()
def forward_layerwise_logits(model, ids):
    """Forward with output_hidden_states, then project each layer's last-token
    hidden through the model's final norm + lm_head to obtain "layer-local
    logits"."""
    out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
    hidden_states = out.hidden_states  # tuple length L_total+1 (incl. embed)
    final_norm = model.model.norm
    lm_head = model.lm_head
    layer_logits = []
    for h in hidden_states:
        last = h[0, -1, :]                # (d_model,)
        normed = final_norm(last.unsqueeze(0).unsqueeze(0))[0, 0, :]
        logits = lm_head(normed.unsqueeze(0))[0]   # (V,)
        layer_logits.append(logits.float())
    final_logits = out.logits[0, -1, :].float()
    return layer_logits, final_logits


def kl_over_vocab(logit_base, logit_pert):
    logp_base = F.log_softmax(logit_base, dim=-1)
    logp_p = F.log_softmax(logit_pert, dim=-1)
    p_p = logp_p.exp()
    return float((p_p * (logp_p - logp_base)).sum().item())


@torch.no_grad()
def run_one_task(model, tok, task, tools_json, ranks, sel_layers,
                 n_kv, head_dim, amp, base_seed, task_idx):
    d = head_dim
    prompt_marked = build_chat_prompt_marked(tok, task, tools_json)
    ids, steer_mask = build_marker_steer_mask(prompt_marked, tok)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    ids = ids.to(device)
    mask_b = steer_mask[0].to(device)
    if mask_b.sum().item() == 0:
        return None

    layer_logits_base, final_logits_base = forward_layerwise_logits(model, ids)
    L_total_plus_1 = len(layer_logits_base)   # L+1 (incl embedding)

    per_rank = {}
    for r in ranks:
        seed_r = base_seed + 1000 * r + (task_idx + 1) * 7919
        proj = build_random_projection(len(sel_layers), n_kv, d, r, seed_r, device, dtype)

        handles = []
        for i, L in enumerate(sel_layers):
            attn = model.model.layers[L].self_attn
            module = attn.k_norm if hasattr(attn, "k_norm") else attn.k_proj
            handles.append(module.register_forward_hook(
                make_kbias_hook(i, proj, n_kv, head_dim, mask_b, amp)
            ))
        try:
            layer_logits_p, final_logits_p = forward_layerwise_logits(model, ids)
        finally:
            for h in handles:
                h.remove()

        per_layer_kl = []
        for lb, lp in zip(layer_logits_base, layer_logits_p):
            per_layer_kl.append(kl_over_vocab(lb, lp))
        final_kl = kl_over_vocab(final_logits_base, final_logits_p)
        per_rank[str(r)] = {
            "per_layer_kl": per_layer_kl,  # length L+1
            "final_kl": final_kl,
        }
        del proj

    return {
        "prompt_len": int(ids.shape[1]),
        "mask_sum": int(mask_b.sum().item()),
        "per_rank": per_rank,
        "L_total_plus_1": L_total_plus_1,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dataset", default="external/tau2-bench/data/tau2/domains/telecom/tasks.json")
    p.add_argument("--layers-spec", default="last10")
    p.add_argument("--ranks", nargs="+", type=int, default=[1, 6, 12, 48, 96])
    p.add_argument("--amp", type=float, default=0.3)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="reports/new_theorem_test/phase_b2_layer_kl.json")
    args = p.parse_args()

    print(f"[load] {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
    ).to(args.device)
    model.eval()

    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // n_q
    L_total = len(model.model.layers)
    sel_layers = parse_layers_spec(args.layers_spec, L_total)
    print(f"[config] n_q={n_q} n_kv={n_kv} d={head_dim} L_total={L_total}")

    tasks_all = json.load(open(args.dataset))
    tasks = tasks_all[: args.n]
    domain_tools = extract_domain_tools(tasks_all)
    tools_json = build_tools_json(domain_tools)
    print(f"[tasks] N={len(tasks)}")
    print(f"[sweep] ranks={args.ranks} amp={args.amp}")

    per_task = []
    t0 = time.time()
    for idx, task in enumerate(tasks):
        try:
            out = run_one_task(model, tok, task, tools_json, args.ranks, sel_layers,
                               n_kv, head_dim, args.amp, args.seed, idx)
        except Exception as exc:
            print(f"  [{idx}] ERR {type(exc).__name__}: {exc}", flush=True)
            continue
        if out is None:
            continue
        per_task.append({"idx": idx, **out})
        if (idx + 1) % 5 == 0 or idx == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (len(tasks) - idx - 1)
            summary = "  ".join(
                f"r={r}:final={statistics.mean(pt['per_rank'][str(r)]['final_kl'] for pt in per_task):.4f}"
                for r in args.ranks
            )
            print(f"  [{idx+1}/{len(tasks)}] t={elapsed:.1f}s eta={eta:.1f}s  {summary}", flush=True)

    if not per_task:
        raise RuntimeError("no tasks produced results")

    # Aggregate per-layer mean KL across tasks
    L_plus_1 = per_task[0]["L_total_plus_1"]
    agg = {}
    for r in args.ranks:
        per_layer = [[] for _ in range(L_plus_1)]
        finals = []
        for pt in per_task:
            pr = pt["per_rank"][str(r)]
            for li, v in enumerate(pr["per_layer_kl"]):
                per_layer[li].append(v)
            finals.append(pr["final_kl"])
        agg[str(r)] = {
            "per_layer_mean": [statistics.mean(v) for v in per_layer],
            "per_layer_std": [statistics.pstdev(v) for v in per_layer],
            "final_mean": statistics.mean(finals),
            "final_std": statistics.pstdev(finals),
            "n": len(per_task),
        }

    # Diagnostic: monotonicity of per-layer means across ranks
    print(f"\n========== Per-layer mean KL across ranks ==========")
    rank_list = args.ranks
    print(f"Layer  " + "  ".join(f"r={r:>3}" for r in rank_list))
    for li in range(L_plus_1):
        vals = [agg[str(r)]["per_layer_mean"][li] for r in rank_list]
        print(f"L{li:>3}   " + "  ".join(f"{v:.4f}" for v in vals))
    print(f"final  " + "  ".join(f"{agg[str(r)]['final_mean']:.4f}" for r in rank_list))

    # Check monotonicity of each layer: is the KL vector across ranks monotonic?
    print(f"\n========== Monotonicity check (KL vs rank, per layer) ==========")
    monotonic_count = 0
    non_monotonic_layers = []
    for li in range(L_plus_1):
        vals = [agg[str(r)]["per_layer_mean"][li] for r in rank_list]
        diffs = [b - a for a, b in zip(vals[:-1], vals[1:])]
        all_nonneg = all(d >= -1e-6 for d in diffs)
        if all_nonneg:
            monotonic_count += 1
        else:
            non_monotonic_layers.append(li)
    print(f"Monotonic layers: {monotonic_count}/{L_plus_1}")
    print(f"Non-monotonic layers: {non_monotonic_layers}")
    final_vals = [agg[str(r)]["final_mean"] for r in rank_list]
    final_diffs = [b - a for a, b in zip(final_vals[:-1], final_vals[1:])]
    final_monotonic = all(d >= -1e-6 for d in final_diffs)
    print(f"Final LM-head logits monotonic: {final_monotonic}  (values: {final_vals})")

    out = {
        "args": vars(args),
        "sel_layers": sel_layers,
        "L_total": L_total,
        "n_tasks_used": len(per_task),
        "aggregate": agg,
        "monotonic_count": monotonic_count,
        "non_monotonic_layers": non_monotonic_layers,
        "final_monotonic": final_monotonic,
        "per_task": per_task,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
