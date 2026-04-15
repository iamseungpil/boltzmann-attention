#!/usr/bin/env python3
"""
H-cat falsifiability test for Mistral-Inst specificity reversal.

Hypothesis 1 (paper §5.5.1 reframing): Mistral-Inst-v0.3 violates (H-cat) —
key channels do NOT exhibit bimodal facet structure. The catalog-built B_ont
projects onto a model-irrelevant direction, hence real B_ont causes systematic
damage while random noise averages out.

Measurement: Var(B_ont^T K) / Var(K) per layer, per head — the facet-projection
energy ratio. Compare Mistral-Inst (reversed) vs Qwen-Inst (specificity intact)
vs Llama-Inst (large positive lift).

If H-cat holds (Qwen, Llama), facet-projection captures a substantial fraction
of K-variance (>= ~rank/d_head ratio, e.g. 24/128 ~= 0.19, but with concentration
typically 2-5x). If H-cat violated (Mistral-Inst hypothesis), ratio collapses
toward random projection floor (~rank/d_head with no concentration gain).

Also reports per-head min-rank histogram of B_ont (skipL0+padmax artifact check).

Runtime: ~10 min per model on A6000.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_data(path: str, n: int):
    with open(path) as f:
        d = json.load(f)
    return d[:n]


def get_layer_keys(model, tokenizer, prompts, device, layer: int, n_kv: int, head_dim: int):
    """Run forward pass, hook layer-l k_proj output, return [N_tokens, n_kv, head_dim]."""
    keys = []

    def hook(module, inp, out):
        # out: [B, T, n_kv * head_dim]
        B, T, _ = out.shape
        k = out.view(B, T, n_kv, head_dim)
        keys.append(k.detach().to(torch.float32).cpu())

    # Locate layer-l k_proj
    if hasattr(model, "model"):
        layer_module = model.model.layers[layer]
    else:
        layer_module = model.layers[layer]
    h = layer_module.self_attn.k_proj.register_forward_hook(hook)

    try:
        for prompt in prompts:
            inputs = tokenizer(prompt["query"] if isinstance(prompt, dict) else prompt,
                              return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                model(**inputs)
    finally:
        h.remove()

    # Concatenate over batch dim, keeping per-head structure
    K = torch.cat([k.reshape(-1, n_kv, head_dim) for k in keys], dim=0)  # [N_tokens, n_kv, head_dim]
    return K


def measure_hcat_ratio(K: torch.Tensor, B_ont_layer: torch.Tensor):
    """
    K: [N_tokens, n_kv, head_dim]
    B_ont_layer: [n_kv, head_dim, R]  per-head facet basis at this layer

    Returns per-head: Var(B_ont^T K) / Var(K), rank, ||B_ont||_F.
    """
    N, n_kv, d = K.shape
    R = B_ont_layer.shape[-1]

    # Total variance per head: trace of cov = sum of per-channel variances
    K_centered = K - K.mean(dim=0, keepdim=True)  # [N, n_kv, d]
    var_K = (K_centered ** 2).sum(dim=(0, 2)) / N  # [n_kv]

    # Project to facet subspace: B_ont^T K
    # K_h: [N, d], B_h: [d, R] => proj: [N, R]
    proj_var = torch.zeros(n_kv)
    for h in range(n_kv):
        K_h = K_centered[:, h, :]  # [N, d]
        B_h = B_ont_layer[h]  # [d, R]
        proj = K_h @ B_h  # [N, R]
        proj_var[h] = (proj ** 2).sum() / N

    ratio = proj_var / var_K.clamp_min(1e-10)

    # Random baseline: expected ratio for random R-dim subspace = R/d
    random_baseline = R / d

    # Concentration gain over random
    gain = ratio / random_baseline

    return {
        "ratio": ratio.tolist(),
        "ratio_mean": float(ratio.mean()),
        "ratio_median": float(ratio.median()),
        "random_baseline": random_baseline,
        "gain_mean": float(gain.mean()),
        "gain_median": float(gain.median()),
        "var_K_mean": float(var_K.mean()),
        "rank": R,
        "head_dim": d,
        "n_kv": n_kv,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--b-ont", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--layers", type=int, nargs="+", default=[5, 13, 21],
                   help="Layers to measure (low/mid/high)")
    p.add_argument("--dataset", default="/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json")
    p.add_argument("--n-prompts", type=int, default=64)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print(f"[load] {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="eager", low_cpu_mem_usage=True,
    )
    model.eval()
    n_kv = model.config.num_key_value_heads
    head_dim = (model.config.hidden_size // model.config.num_attention_heads
                if not hasattr(model.config, "head_dim") else model.config.head_dim)

    print(f"[load] B_ont {args.b_ont}")
    B_ont = torch.load(args.b_ont, map_location="cpu", weights_only=False)
    # B_ont shape: [L, n_kv, head_dim, R]
    print(f"[B_ont] shape={tuple(B_ont.shape)}")
    L_ont, n_kv_ont, d_ont, R = B_ont.shape

    # Per-head rank histogram (min over heads per layer)
    rank_per_layer = []
    for ell in range(L_ont):
        ranks_h = []
        for h in range(n_kv_ont):
            B_lh = B_ont[ell, h]  # [d, R]
            r = int(torch.linalg.matrix_rank(B_lh.float(), tol=1e-5).item())
            ranks_h.append(r)
        rank_per_layer.append({
            "layer": ell,
            "min_rank": min(ranks_h),
            "max_rank": max(ranks_h),
            "median_rank": sorted(ranks_h)[len(ranks_h)//2],
        })
    pathological_layers = [r["layer"] for r in rank_per_layer
                           if r["min_rank"] < 0.5 * r["median_rank"]]

    print(f"[data] loading {args.n_prompts} prompts")
    data = load_data(args.dataset, args.n_prompts)
    prompts = [d["query"] for d in data]

    results = {
        "model": args.model,
        "b_ont_path": args.b_ont,
        "rank_per_layer": rank_per_layer,
        "pathological_layers_count": len(pathological_layers),
        "pathological_layers": pathological_layers,
        "per_layer_hcat": [],
    }

    for ell in args.layers:
        print(f"[measure] layer {ell}")
        K = get_layer_keys(model, tok, prompts, args.device, ell, n_kv, head_dim)
        print(f"  K shape: {tuple(K.shape)}")
        m = measure_hcat_ratio(K, B_ont[ell])
        m["layer"] = ell
        results["per_layer_hcat"].append(m)
        print(f"  ratio_mean={m['ratio_mean']:.4f} random_baseline={m['random_baseline']:.4f} "
              f"gain={m['gain_mean']:.2f}x")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[wrote] {args.out}")

    print("\n=== Summary ===")
    print(f"Model: {args.model}")
    print(f"Pathological layers (min_rank < 0.5*median): {len(pathological_layers)}")
    for m in results["per_layer_hcat"]:
        verdict = "H-cat HOLDS" if m["gain_mean"] > 2.0 else "H-cat WEAK" if m["gain_mean"] > 1.2 else "H-cat VIOLATED"
        print(f"  L={m['layer']:2d} gain={m['gain_mean']:.2f}x ({verdict})")


if __name__ == "__main__":
    main()
