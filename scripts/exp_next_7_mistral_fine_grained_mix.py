#!/usr/bin/env python3
"""
Next-7: Fine-grained mixed-precision configs on Mistral-7B
=============================================================

Next-5 showed top15@3b gives Pareto-optimal at avg 2.47 bits.
This experiment explores more fine-grained mix configurations to
find the true Pareto frontier between top5@3b and all-4b.

Configs (tested at various bit budgets):
  - top-K @ 3b vs top-K @ 4b at same K
  - top5@4b + top5-10@3b (mixed tiers)
  - top3@4b + top3-10@3b
  - top5@4b + top5-15@3b + top15-20@3b (3-tier)
  - Descending budget: top1@8b, top2@6b, top3@5b, top5@4b, top10@3b
"""
import json
import time
import gc
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

MODEL_NAME = 'mistralai/Mistral-7B-v0.3'
SHORT = 'mistral-7b'
DEVICE = 'cuda:0'
DTYPE = torch.bfloat16
N_CALIB_TOKENS = 1024
N_EVAL_TOKENS = 2048

OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')


def lloyd_max_1d_fit(col, bits, n_iter=30):
    n_levels = 2 ** bits
    pcts = np.linspace(0, 100, n_levels + 2)[1:-1]
    centroids = np.percentile(col, pcts)
    centroids = np.sort(centroids)
    for _ in range(n_iter):
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        idx = np.searchsorted(boundaries, col)
        new_c = centroids.copy()
        for k in range(n_levels):
            m = idx == k
            if m.sum() > 0:
                new_c[k] = col[m].mean()
        if np.max(np.abs(new_c - centroids)) < 1e-6:
            break
        centroids = new_c
    return centroids


class MixedBitLloydHook:
    def __init__(self, centroids_per_head, n_kv, head_dim):
        self.centroids = centroids_per_head
        self.n_kv = n_kv
        self.head_dim = head_dim

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x = output.view(B, T, self.n_kv, self.head_dim)
        x_np = x.float().cpu().numpy()
        x_q = np.zeros_like(x_np)
        for hk in range(self.n_kv):
            data = x_np[:, :, hk, :]
            shape = data.shape
            data_flat = data.reshape(-1, self.head_dim)
            c = self.centroids[hk]
            for j in range(self.head_dim):
                boundaries = (c[j, :-1] + c[j, 1:]) / 2
                idx = np.searchsorted(boundaries, data_flat[:, j])
                data_flat[:, j] = c[j, idx]
            x_q[:, :, hk, :] = data_flat.reshape(shape)
        return torch.from_numpy(x_q).view(B, T, self.n_kv * self.head_dim).to(output.device).to(output.dtype)


def fit_lloyd_for_layer_bits(model, tok, calib_text, layer_idx, bits):
    enc = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB_TOKENS)
    input_ids = enc['input_ids'].to(DEVICE)
    captured = {}
    def hook(m, i, o):
        captured['k'] = o.detach().cpu().float().numpy()
    attn = model.model.layers[layer_idx].self_attn
    h = attn.k_proj.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(input_ids, use_cache=False)
    h.remove()

    K = captured['k']
    n_kv = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    K = K.reshape(1, -1, n_kv, head_dim)[0]
    centroids_per_head = []
    for hk in range(n_kv):
        K_h = K[:, hk, :].astype(np.float32)
        centroids = np.zeros((head_dim, 2 ** bits))
        for j in range(head_dim):
            centroids[j] = lloyd_max_1d_fit(K_h[:, j], bits, n_iter=20)
        centroids_per_head.append(centroids)
    return centroids_per_head


def compute_ppl(model, input_ids):
    with torch.no_grad():
        out = model(input_ids, use_cache=False)
        logits = out.logits[:, :-1, :].contiguous()
        targets = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            reduction='mean'
        )
        return float(torch.exp(loss).item()), float(loss.item())


def get_texts(tok):
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        texts = [t for t in ds['text'] if len(t.strip()) > 100]
        return '\n\n'.join(texts[:300]), '\n\n'.join(texts[300:600])
    except Exception:
        return " ".join(["Calib."] * 5000), " ".join(["Eval."] * 5000)


def main():
    print("=" * 60)
    print(f"Next-7: Fine-grained mix on {MODEL_NAME}")
    print("=" * 60, flush=True)
    t_start = time.time()

    # Load Mistral sensitivity ranking from Exp4
    sens_path = OUT_DIR / 'exp4_per_layer_lloyd_breakdown.json'
    with open(sens_path) as f:
        sens_data = json.load(f)
    first_key = [k for k in sens_data.keys() if not k.startswith('_')][0]
    sorted_layers = [r['layer'] for r in sens_data[first_key]['sorted_by_delta']]
    print(f"  Loaded ranking. Top-15 sensitive: {sorted_layers[:15]}", flush=True)

    print("\nLoading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device_map=DEVICE,
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t_start:.1f}s", flush=True)

    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    calib_text, eval_text = get_texts(tok)
    eval_enc = tok(eval_text, return_tensors='pt', truncation=True, max_length=N_EVAL_TOKENS)
    eval_ids = eval_enc['input_ids'].to(DEVICE)

    ppl_fp16, _ = compute_ppl(model, eval_ids)
    print(f"  FP16 PPL: {ppl_fp16:.4f}", flush=True)

    # Helper to make bit assignments
    def tiers(*segments):
        """segments: list of (start, end, bits). Rest is 2-bit."""
        bits = [2] * n_layers
        for (s, e, b) in segments:
            for li in sorted_layers[s:e]:
                bits[li] = b
        return bits

    # Fine-grained mix configs
    configs = {
        # Baselines (also from Next-5, reproduced here for consistency)
        'baseline_all_2b':       [2] * n_layers,
        'top15_3b':              tiers((0, 15, 3)),    # avg 2.47 (Next-5 Pareto sweet)

        # 2-tier variants
        'top3_4b':               tiers((0, 3, 4)),       # avg 2.19
        'top5_4b':               tiers((0, 5, 4)),       # avg 2.31
        'top3_4b_top10_3b':      tiers((0, 3, 4), (3, 10, 3)),  # avg 2.41
        'top5_4b_top10_3b':      tiers((0, 5, 4), (5, 10, 3)),  # avg 2.47
        'top5_4b_top15_3b':      tiers((0, 5, 4), (5, 15, 3)),  # avg 2.63
        'top5_4b_top20_3b':      tiers((0, 5, 4), (5, 20, 3)),  # avg 2.78

        # 3-tier variants
        'top3_5b_top10_3b':      tiers((0, 3, 5), (3, 10, 3)),  # avg 2.50
        'top3_5b_top15_3b':      tiers((0, 3, 5), (3, 15, 3)),  # avg 2.66

        # Descending budget (more bits for most sensitive)
        'descending_8_6_5_4_3':  tiers((0, 1, 8), (1, 2, 6), (2, 3, 5), (3, 5, 4), (5, 10, 3)),  # avg ~2.53
        'descending_6_5_4_3':    tiers((0, 1, 6), (1, 3, 5), (3, 5, 4), (5, 15, 3)),             # avg ~2.69

        # Aggressive top
        'top1_8b':               tiers((0, 1, 8)),             # avg 2.19
        'top1_6b_top10_3b':      tiers((0, 1, 6), (1, 10, 3)), # avg 2.41

        # Control: bottom-K instead of top-K (should be worse)
        'bottom15_3b':           tiers((n_layers - 15, n_layers, 3)),  # avg 2.47 (wrong layers)

        # Upper bound
        'all_3b':                [3] * n_layers,  # avg 3.00
        'all_4b':                [4] * n_layers,  # avg 4.00
    }

    # Fit unique pairs
    unique_pairs = set()
    for bl in configs.values():
        for li, b in enumerate(bl):
            unique_pairs.add((li, b))
    print(f"\n  Fitting {len(unique_pairs)} unique (layer, bits) pairs...", flush=True)

    centroid_cache = {}
    t_fit = time.time()
    for idx, (li, b) in enumerate(sorted(unique_pairs)):
        centroid_cache[(li, b)] = fit_lloyd_for_layer_bits(model, tok, calib_text, li, b)
        if (idx + 1) % 20 == 0:
            print(f"  Fit progress: {idx+1}/{len(unique_pairs)}", flush=True)
    print(f"  Fit done in {time.time()-t_fit:.1f}s", flush=True)

    # Run configs
    results = {}
    print("\n  Running configs:", flush=True)
    for name, bits_list in configs.items():
        avg_b = sum(bits_list) / len(bits_list)
        per_layer_cent = [centroid_cache[(li, bits_list[li])] for li in range(n_layers)]
        handles = []
        for li, cent in enumerate(per_layer_cent):
            hook = MixedBitLloydHook(cent, n_kv, head_dim)
            h = model.model.layers[li].self_attn.k_proj.register_forward_hook(hook)
            handles.append(h)
        t0 = time.time()
        ppl, loss = compute_ppl(model, eval_ids)
        elapsed = time.time() - t0
        for h in handles:
            h.remove()
        results[name] = {
            'avg_bits': avg_b,
            'ppl': ppl,
            'loss': loss,
            'delta_vs_fp16_pct': (ppl - ppl_fp16) / ppl_fp16 * 100,
        }
        print(f"    {name:<28} avg={avg_b:.3f} PPL={ppl:.4f} Δ={((ppl-ppl_fp16)/ppl_fp16*100):+.2f}% ({elapsed:.1f}s)", flush=True)

    # Sort by PPL
    print(f"\n  Sorted by PPL (best first):")
    for name, r in sorted(results.items(), key=lambda x: x[1]['ppl']):
        print(f"    {name:<28} avg={r['avg_bits']:.3f} PPL={r['ppl']:.4f}")

    # Pareto frontier
    by_budget = sorted(results.items(), key=lambda x: (x[1]['avg_bits'], x[1]['ppl']))
    pareto = []
    best_so_far = float('inf')
    for name, r in by_budget:
        if r['ppl'] < best_so_far:
            pareto.append((name, r['avg_bits'], r['ppl']))
            best_so_far = r['ppl']
    print(f"\n  Pareto frontier:")
    for name, ab, ppl in pareto:
        print(f"    {name:<28} avg={ab:.3f} PPL={ppl:.4f}")

    # Save
    out = {
        'model': MODEL_NAME,
        'ppl_fp16': ppl_fp16,
        'sensitivity_ranking_top20': sorted_layers[:20],
        'configs': results,
        'pareto_frontier': pareto,
        'runtime_sec': time.time() - t_start,
    }
    out_file = OUT_DIR / 'exp_next7_fine_grained_mix.json'
    with open(out_file, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_file}")
    print(f"Total runtime: {out['runtime_sec']:.1f}s")


if __name__ == '__main__':
    main()
