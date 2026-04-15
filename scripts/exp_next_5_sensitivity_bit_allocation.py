#!/usr/bin/env python3
"""
Next-5: Sensitivity-based bit allocation (Mistral + Qwen)
============================================================

Uses PRE-MEASURED per-layer ΔPPL (from Exp4/Next-3) to rank layers by
Lloyd-Max sensitivity, then allocates extra bits to top-K most sensitive layers.

Tests various average bit budgets to find the Pareto frontier.

For each model (Mistral, Qwen):
  1. Rank layers by ΔPPL (from exp4 / exp_next3 JSON)
  2. Test configurations:
     - top-1 @ 4-bit (rest 2-bit): avg ~2.06
     - top-3 @ 3-bit: avg ~2.09
     - top-5 @ 3-bit: avg 2.16
     - top-5 @ 4-bit: avg 2.31
     - top-10 @ 3-bit: avg 2.31
     - top-10 @ 4-bit: avg 2.625
     - top-5 @ 4-bit + top 6-10 @ 3-bit: avg 2.47  (mixed)
  3. Measure PPL for each
"""
import json
import time
import gc
import sys
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

DEVICE = 'cuda:0'
DTYPE = torch.bfloat16
N_CALIB_TOKENS = 1024
N_EVAL_TOKENS = 2048

OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
        return " ".join(["Calibration."] * 5000), " ".join(["Evaluation."] * 5000)


def install_layer_hooks(model, per_layer_centroids, n_kv, head_dim):
    handles = []
    for li, centroids in enumerate(per_layer_centroids):
        if centroids is None:
            continue
        hook = MixedBitLloydHook(centroids, n_kv, head_dim)
        h = model.model.layers[li].self_attn.k_proj.register_forward_hook(hook)
        handles.append(h)
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def load_sensitivity_ranking(json_path, n_layers):
    """Load per-layer ΔPPL from JSON and return layers sorted by sensitivity desc."""
    with open(json_path) as f:
        data = json.load(f)

    # Handle both exp4 (chain container) and exp_next3 (direct) formats
    if 'per_layer' in data:
        per_layer = data['per_layer']
    else:
        # exp4 has a nested structure
        first_key = [k for k in data.keys() if not k.startswith('_')][0]
        per_layer = data[first_key]['per_layer']

    # Sort by delta_ppl descending
    sorted_layers = sorted(per_layer, key=lambda x: -x['delta_ppl'])
    ranking = [(r['layer'], r['delta_ppl']) for r in sorted_layers]
    return ranking


def build_configs(n_layers, ranking):
    """Build bit allocation configs based on sensitivity ranking."""
    sorted_layers = [li for li, _ in ranking]

    def make_config(top_k, extra_bits_per_layer):
        """Allocate (2 + extra_bits) bits to top_k layers, 2 to rest."""
        bits = [2] * n_layers
        for li in sorted_layers[:top_k]:
            bits[li] = 2 + extra_bits_per_layer
        return bits

    def make_mixed(top_a, top_b, extra_a, extra_b):
        """top_a gets +extra_a, next (top_b-top_a) gets +extra_b."""
        bits = [2] * n_layers
        for li in sorted_layers[:top_a]:
            bits[li] = 2 + extra_a
        for li in sorted_layers[top_a:top_b]:
            bits[li] = 2 + extra_b
        return bits

    configs = {
        'baseline_all_2b':        [2] * n_layers,
        'top1_4b':                make_config(1, 2),
        'top3_3b':                make_config(3, 1),
        'top5_3b':                make_config(5, 1),
        'top5_4b':                make_config(5, 2),
        'top10_3b':               make_config(10, 1),
        'top10_4b':               make_config(10, 2),
        'mixed_top5_4b_top10_3b': make_mixed(5, 10, 2, 1),
        'top15_3b':               make_config(15, 1),
        'all_4b':                 [4] * n_layers,
    }
    return configs


def run_model(model_name, short_name, sensitivity_json):
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Sensitivity source: {sensitivity_json}")
    print('='*60, flush=True)
    t_start = time.time()

    # Load sensitivity ranking
    print("Loading sensitivity ranking...", flush=True)
    tok_load = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Peek at n_layers from config
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_name)
    n_layers = cfg.num_hidden_layers
    ranking = load_sensitivity_ranking(sensitivity_json, n_layers)
    print(f"  Loaded ranking for {n_layers} layers")
    print(f"  Top-10 sensitive layers: {[(li, f'{d:+.3f}') for li, d in ranking[:10]]}", flush=True)

    # Build configs
    configs = build_configs(n_layers, ranking)

    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=DTYPE, device_map=DEVICE,
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t_start:.1f}s", flush=True)

    n_kv = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    calib_text, eval_text = get_texts(tok_load)
    eval_enc = tok_load(eval_text, return_tensors='pt', truncation=True, max_length=N_EVAL_TOKENS)
    eval_ids = eval_enc['input_ids'].to(DEVICE)

    # FP16 baseline
    ppl_fp16, _ = compute_ppl(model, eval_ids)
    print(f"  FP16 PPL: {ppl_fp16:.4f}", flush=True)

    # Unique pairs to fit
    unique_pairs = set()
    for bits_list in configs.values():
        for li, b in enumerate(bits_list):
            unique_pairs.add((li, b))
    print(f"\n  Fitting {len(unique_pairs)} (layer, bits) pairs...", flush=True)

    centroid_cache = {}
    t_fit = time.time()
    for idx, (li, b) in enumerate(sorted(unique_pairs)):
        centroid_cache[(li, b)] = fit_lloyd_for_layer_bits(model, tok_load, calib_text, li, b)
        if (idx + 1) % 20 == 0:
            print(f"  Fit progress: {idx+1}/{len(unique_pairs)}", flush=True)
    print(f"  Fit done in {time.time()-t_fit:.1f}s", flush=True)

    # Run each config
    results = {}
    print("\n  Running configs:", flush=True)
    for name, bits_list in configs.items():
        avg_b = sum(bits_list) / len(bits_list)
        per_layer = [centroid_cache[(li, bits_list[li])] for li in range(n_layers)]
        handles = install_layer_hooks(model, per_layer, n_kv, head_dim)
        t0 = time.time()
        ppl, loss = compute_ppl(model, eval_ids)
        elapsed = time.time() - t0
        remove_hooks(handles)

        results[name] = {
            'avg_bits': avg_b,
            'ppl': ppl,
            'loss': loss,
            'delta_vs_fp16_pct': (ppl - ppl_fp16) / ppl_fp16 * 100,
            'bits_per_layer': bits_list,
        }
        print(f"    {name:<28} avg={avg_b:.3f} PPL={ppl:.4f} Δ={((ppl-ppl_fp16)/ppl_fp16*100):+.2f}% ({elapsed:.1f}s)",
              flush=True)

    # Summary
    print(f"\n  {'Config':<30} | {'avg_bits':>8} | {'PPL':>10} | {'Δ vs FP16':>12} | {'Δ vs 2b':>10}")
    print('  ' + '-' * 82)
    base_ppl = results['baseline_all_2b']['ppl']
    for name, r in sorted(results.items(), key=lambda x: x[1]['avg_bits']):
        vs_base = (r['ppl'] - base_ppl) / base_ppl * 100
        print(f"  {name:<30} | {r['avg_bits']:>8.3f} | {r['ppl']:>10.4f} | {r['delta_vs_fp16_pct']:>+11.2f}% | {vs_base:>+9.2f}%")

    # Find Pareto frontier (best PPL for each bit budget)
    by_budget = sorted(results.items(), key=lambda x: (x[1]['avg_bits'], x[1]['ppl']))
    pareto = []
    best_so_far = float('inf')
    for name, r in by_budget:
        if r['ppl'] < best_so_far:
            pareto.append((name, r['avg_bits'], r['ppl']))
            best_so_far = r['ppl']

    print(f"\n  Pareto frontier (bit budget vs min PPL):")
    for name, ab, ppl in pareto:
        print(f"    {name:<30} avg={ab:.3f} PPL={ppl:.4f}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'model': model_name,
        'short_name': short_name,
        'n_layers': n_layers,
        'ppl_fp16': ppl_fp16,
        'sensitivity_ranking': ranking,
        'configs': results,
        'pareto_frontier': pareto,
        'runtime_sec': time.time() - t_start,
    }


def main():
    models = [
        ('mistralai/Mistral-7B-v0.3', 'mistral',
         OUT_DIR / 'exp4_per_layer_lloyd_breakdown.json'),
        ('Qwen/Qwen2.5-7B', 'qwen',
         OUT_DIR / 'exp_next3_qwen_per_layer_lloyd.json'),
    ]

    all_results = {}
    t_total = time.time()

    for model_name, short_name, sensitivity_json in models:
        try:
            r = run_model(model_name, short_name, sensitivity_json)
            all_results[short_name] = r
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"!!! FAILED {model_name}: {e}")
            all_results[short_name] = {'error': str(e)}

    all_results['_meta'] = {
        'total_runtime_sec': time.time() - t_total,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    out_file = OUT_DIR / 'exp_next5_sensitivity_allocation.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nSaved: {out_file}")
    print(f"Total runtime: {(time.time()-t_total)/60:.1f} min")


if __name__ == '__main__':
    main()
