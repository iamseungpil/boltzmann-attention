#!/usr/bin/env python3
"""
V2g: LayerNorm Weight Comparator — Architectural Predictor
===========================================================

Hypothesis: Mistral's pre-attention RMSNorm has large weights at the massive
channel positions, failing to damp the outlier before it reaches k_proj.
Qwen's RMSNorm may damp the same channels, so k_proj never sees the outlier.

This is a no-forward-pass static weight comparison.

Steps:
  1. Load Mistral-7B, Mistral-Nemo-12B, Qwen2.5-7B (weights only, CPU)
  2. For each layer, extract input_layernorm.weight (pre-attention RMSNorm)
  3. Identify top-k positions in each model (the massive channels)
  4. Check if those positions have anomalously large RMSNorm weights

Also: does k_proj norm concentrate on the same positions as RMSNorm weight?
      i.e. is the k_proj "learned" to match the massive channels, or does
      LN fail to damp them?

Runtime: ~30s (CPU-only, no forward pass)
"""
import json, os, time
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import numpy as np
from transformers import AutoModelForCausalLM
from pathlib import Path

OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')

MODELS = [
    ('mistralai/Mistral-7B-v0.3', 'mistral-7b'),
    ('mistralai/Mistral-Nemo-Base-2407', 'mistral-nemo-12b'),
    ('Qwen/Qwen2.5-7B', 'qwen2.5-7b'),
]


def analyze_model(model_id, short_name):
    print(f"\n{'='*70}\n  {short_name}: {model_id}\n{'='*70}", flush=True)
    t0 = time.time()
    import torch
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, device_map='cpu', low_cpu_mem_usage=True,
    )
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    head_dim = hidden // n_q
    print(f"  n_layers={n_layers}, hidden={hidden}, n_kv={n_kv}, head_dim={head_dim}", flush=True)

    per_layer = []
    # Across layers, collect:
    #   1. input_layernorm.weight: (hidden,)
    #   2. k_proj weight per-column L2 norm: (hidden,)  (= how much each input channel contributes to K)
    for li in range(n_layers):
        layer = model.model.layers[li]
        ln_w = layer.input_layernorm.weight.detach().cpu().numpy()  # (hidden,)
        k_w = layer.self_attn.k_proj.weight.detach().cpu().numpy()  # (n_kv*head_dim, hidden)
        k_col_norm = np.linalg.norm(k_w, axis=0)  # (hidden,) - L2 norm per input channel across all K output dims

        # Top-5 channels by LN weight and by k_proj col norm
        top_ln = np.argsort(-np.abs(ln_w))[:5]
        top_k = np.argsort(-k_col_norm)[:5]

        # How much does k_proj col-norm correlate with ln_weight?
        # Normalized Spearman
        from scipy.stats import spearmanr
        rho, _ = spearmanr(np.abs(ln_w), k_col_norm)

        per_layer.append({
            'layer': li,
            'ln_weight_mean': float(np.mean(np.abs(ln_w))),
            'ln_weight_max': float(np.max(np.abs(ln_w))),
            'ln_weight_max_over_mean': float(np.max(np.abs(ln_w)) / max(np.mean(np.abs(ln_w)), 1e-12)),
            'top5_ln_weight_channels': [int(x) for x in top_ln],
            'top5_ln_weight_values': [float(ln_w[x]) for x in top_ln],
            'k_col_norm_mean': float(np.mean(k_col_norm)),
            'k_col_norm_max': float(np.max(k_col_norm)),
            'k_col_norm_max_over_mean': float(np.max(k_col_norm) / max(np.mean(k_col_norm), 1e-12)),
            'top5_k_col_channels': [int(x) for x in top_k],
            'top5_k_col_values': [float(k_col_norm[x]) for x in top_k],
            'spearman_ln_kcol': float(rho),
        })

    # Summary
    mean_ln_max = float(np.mean([d['ln_weight_max_over_mean'] for d in per_layer]))
    max_ln_max = float(np.max([d['ln_weight_max_over_mean'] for d in per_layer]))
    mean_k_max = float(np.mean([d['k_col_norm_max_over_mean'] for d in per_layer]))
    mean_rho = float(np.mean([d['spearman_ln_kcol'] for d in per_layer]))

    print(f"\n  LayerNorm weight anomaly:", flush=True)
    print(f"    Mean max/mean across layers: {mean_ln_max:.2f}×")
    print(f"    Peak max/mean:               {max_ln_max:.2f}×")
    print(f"\n  k_proj input-channel-norm anomaly:", flush=True)
    print(f"    Mean max/mean across layers: {mean_k_max:.2f}×")
    print(f"\n  Spearman(|LN weight|, k_proj col norm) across layers:", flush=True)
    print(f"    Mean ρ: {mean_rho:+.3f}")

    # Look at specific Mistral massive channel (2070)
    if 'mistral' in short_name or 'nemo' in short_name or 'qwen' in short_name:
        print(f"\n  Layer 5 snapshot (first 3 layers after residual ramp):", flush=True)
        for li in [0, 1, 2, 3, 5, 10, 20, n_layers//2, n_layers-1]:
            if li < n_layers:
                d = per_layer[li]
                print(f"    L{li}: top-5 LN ch {d['top5_ln_weight_channels']}, "
                      f"top-5 k_col ch {d['top5_k_col_channels']}", flush=True)

    # Check which channels appear in BOTH top-5 LN and top-5 k_col across layers
    overlap_freq = {}
    for d in per_layer:
        both = set(d['top5_ln_weight_channels']) & set(d['top5_k_col_channels'])
        for c in both:
            overlap_freq[c] = overlap_freq.get(c, 0) + 1
    top_overlap = sorted(overlap_freq.items(), key=lambda x: -x[1])[:10]
    print(f"\n  Channels in both top-5 LN and top-5 k_col norm across layers:", flush=True)
    for c, f in top_overlap:
        pct = f / n_layers * 100
        print(f"    ch{c}: {f}/{n_layers} layers ({pct:.0f}%)", flush=True)

    # Free memory
    del model
    import gc; gc.collect()

    return {
        'model': model_id,
        'short_name': short_name,
        'n_layers': n_layers,
        'hidden': hidden,
        'mean_ln_max_over_mean': mean_ln_max,
        'peak_ln_max_over_mean': max_ln_max,
        'mean_k_col_max_over_mean': mean_k_max,
        'mean_spearman_ln_kcol': mean_rho,
        'top_overlap_channels': top_overlap,
        'per_layer': per_layer,
    }


def main():
    print("="*70)
    print("V2g: LayerNorm & k_proj Weight Comparator")
    print("="*70, flush=True)
    t_start = time.time()

    results = {}
    for mid, sn in MODELS:
        try:
            results[sn] = analyze_model(mid, sn)
        except Exception as e:
            print(f"  ERROR on {sn}: {e}", flush=True)
            import traceback; traceback.print_exc()
            results[sn] = {'error': str(e)}

    # Comparison table
    print(f"\n{'='*70}")
    print("  COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"  {'model':<20}|{'LN max/mean':>14}|{'k_col max/mean':>17}|{'ρ(LN,kcol)':>14}", flush=True)
    print(f"  {'-'*65}", flush=True)
    for sn, r in results.items():
        if 'error' in r:
            print(f"  {sn:<20}|  ERROR", flush=True)
            continue
        print(f"  {sn:<20}|{r['mean_ln_max_over_mean']:>13.2f}×|"
              f"{r['mean_k_col_max_over_mean']:>16.2f}×|"
              f"{r['mean_spearman_ln_kcol']:>+13.3f}", flush=True)

    out = OUT_DIR / 'exp_v2g_ln_weight_comparator.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Runtime: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f}m)")


if __name__ == '__main__':
    main()
