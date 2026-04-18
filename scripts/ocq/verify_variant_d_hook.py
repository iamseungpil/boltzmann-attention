"""Phase 0 verification for variant D (random orthonormal experts).

Purpose: verify that variant D's K-side hook is actually firing with non-trivial
delta magnitude. If delta is near-zero everywhere, variant D's F1=no_steer result
is a no-op artifact, not evidence that random basis doesn't work.

Checks:
  1. Mask.sum() > 0 (localization active)
  2. Per-expert coef distribution non-trivial (routing active)
  3. Delta ||amp * P_dyn @ K|| relative to ||K|| (actual perturbation magnitude)
  4. Compare with variant A (our B_ont) on same tasks

Usage:
  python verify_variant_d_hook.py --n-samples 5
"""
from __future__ import annotations

import argparse
import json
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
from eval_tau2_bench import (
    build_chat_prompt_marked,
    build_tools_json,
    extract_domain_tools,
)


def per_task_delta_magnitude(model, tok, engine, task, tools_json, amp=0.3):
    """For a single task: run engine.precompute + install hook with monitoring,
    do a single forward pass, capture delta magnitudes per (layer, head)."""
    prompt_marked = build_chat_prompt_marked(tok, task, tools_json)
    ids, steer_mask = build_marker_steer_mask(prompt_marked, tok)
    ids = ids.to(next(model.parameters()).device)
    steer_mask = steer_mask.to(next(model.parameters()).device)

    # Capture routing coefs + delta magnitudes
    proj_table = engine.precompute_projections(ids)
    # Analyze per-(layer, head) projection magnitude (Frobenius)
    proj_info = {}
    for (i_layer, h), P in proj_table.items():
        frob = P.float().norm().item()
        proj_info[(i_layer, h)] = {
            "P_frob_norm": frob,
        }

    # Forward pass with hook to measure delta
    deltas = {}  # (layer, head) -> max ||delta||
    def make_spy_hook(i_layer, L):
        def _hook(mod, inputs, output):
            k_in = output
            if k_in.dim() == 3:
                B, T, D = k_in.shape
                n_kv = engine.n_kv
                head_dim = engine.head_dim
                if D != n_kv * head_dim:
                    return k_in
                k_view = k_in.view(B, T, n_kv, head_dim)
            elif k_in.dim() == 4:
                B, T, n_kv, head_dim = k_in.shape
                k_view = k_in
            else:
                return k_in

            if steer_mask is None or steer_mask.sum() == 0:
                return k_in
            mask_b = steer_mask[0].to(k_view.device)

            for h in range(n_kv):
                dyn = proj_table.get((i_layer, h))
                if dyn is None:
                    continue
                dyn = dyn.to(k_view.device)
                k_sel = k_view[0, mask_b, h, :]
                delta = torch.matmul(dyn.to(k_sel.dtype), k_sel.T).T
                delta_norm = delta.float().norm().item()
                k_norm = k_sel.float().norm().item()
                deltas[(i_layer, h)] = {
                    "L": L,
                    "delta_frob": delta_norm,
                    "k_frob": k_norm,
                    "ratio": delta_norm / (k_norm + 1e-8),
                    "mask_tokens": int(mask_b.sum().item()),
                }
            return k_in  # don't modify, just measure
        return _hook

    # attach monitoring hooks (not modifying)
    handles = []
    for i, L in enumerate(engine.sel_layers):
        attn = model.model.layers[L].self_attn
        module = attn.k_norm if hasattr(attn, "k_norm") else attn.k_proj
        handles.append(module.register_forward_hook(make_spy_hook(i, L)))

    try:
        with torch.no_grad():
            _ = model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return {
        "task_id": task.get("id", "?"),
        "prompt_len": int(ids.shape[1]),
        "mask_sum": int(steer_mask.sum().item()),
        "n_routing_entries": len(proj_table),
        "proj_info": {f"{k[0]}_{k[1]}": v for k, v in proj_info.items()},
        "deltas": {f"{k[0]}_{k[1]}": v for k, v in deltas.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--domain", default="telecom")
    p.add_argument("--dataset", default="external/tau2-bench/data/tau2/domains/telecom/tasks.json")
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--b-ont", default="external/SEKA/seka_projections/ontology-qwen25-7b-tau2-telecom/B_ont.pt")
    p.add_argument("--out", default="reports/tau2_2026_04_18/variantD_phase0_verification.json")
    args = p.parse_args()

    print(f"[load] {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(args.device)
    model.eval()
    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // n_q

    tasks = json.load(open(args.dataset))[: args.n_samples]
    domain_tools = extract_domain_tools(tasks)
    tools_json = build_tools_json(domain_tools)

    results = {}
    for variant, paths in [
        ("A", "external/SEKA/seka_projections/adaseka-qwen25-7b-tau2-telecom/expert_paths.json"),
        ("D", "external/SEKA/seka_projections/adaseka-qwen25-7b-tau2-telecom-variantD/expert_paths.json"),
    ]:
        print(f"\n=== Variant {variant} ===")
        engine = CanonicalAdaSEKAEngine(
            model=model, expert_paths_json=paths, layers="last10",
            topk=3, temperature=1.0,
            n_kv=n_kv, n_q=n_q, head_dim=head_dim,
        )
        print(f"  experts={engine.expert_names}")
        variant_results = []
        for i, task in enumerate(tasks):
            info = per_task_delta_magnitude(model, tok, engine, task, tools_json)
            variant_results.append(info)
            # Summary of deltas
            if info["deltas"]:
                ratios = [v["ratio"] for v in info["deltas"].values()]
                proj_frobs = [v["P_frob_norm"] for v in info["proj_info"].values()]
                print(f"  task {i+1}/{len(tasks)} ({info['task_id'][:50]})")
                print(f"    mask_sum={info['mask_sum']}  routing_entries={info['n_routing_entries']}")
                print(f"    P_frob: mean={sum(proj_frobs)/len(proj_frobs):.4f}  max={max(proj_frobs):.4f}  min={min(proj_frobs):.4f}")
                print(f"    ||delta||/||K|| ratio: mean={sum(ratios)/len(ratios):.4f}  max={max(ratios):.4f}  min={min(ratios):.4f}")
        results[variant] = variant_results
        del engine
        torch.cuda.empty_cache()

    # Compare A vs D side by side
    print("\n=== A vs D COMPARISON (per task, mean delta/K ratio) ===")
    for i in range(len(tasks)):
        a = results["A"][i]
        d = results["D"][i]
        a_ratios = [v["ratio"] for v in a["deltas"].values()]
        d_ratios = [v["ratio"] for v in d["deltas"].values()]
        a_proj = [v["P_frob_norm"] for v in a["proj_info"].values()]
        d_proj = [v["P_frob_norm"] for v in d["proj_info"].values()]
        print(f"  task {i+1} ({a['task_id'][:45]})")
        print(f"    A: P_frob mean={sum(a_proj)/len(a_proj):.4f}  delta/K mean={sum(a_ratios)/len(a_ratios):.5f}")
        print(f"    D: P_frob mean={sum(d_proj)/len(d_proj):.4f}  delta/K mean={sum(d_ratios)/len(d_ratios):.5f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
