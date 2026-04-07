#!/usr/bin/env python3
"""
Experiment 2: Spherical quantizer prototype on Mistral-7B
==========================================================

For each (layer, head) of Mistral-7B:
  1. Extract pre-RoPE keys (T, d) via k_proj hook
  2. Apply 3 quantizers at 2-bit effective rate:
     (a) Uniform 2-bit (per-dim, post-PCA rotation)
     (b) L² Lloyd 2-bit (per-dim, post-PCA rotation, optimal scalar)
     (c) Spherical 2-bit: 2D polar (3 bits angle + 1 bit magnitude per block = 4 bits / 2 dims)
  3. Measure reconstruction MSE per quantizer
  4. Measure attention-weighted MSE (proxy for PPL-relevant distortion)

Output: per-head MSE comparison + outlier head deep-dive
"""
import json
import time
import gc
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

MODEL_NAME = 'mistralai/Mistral-7B-v0.3'
SHORT = 'mistral-7b'
DEVICE = 'cuda:0'
DTYPE = torch.bfloat16
N_CALIB_TOKENS = 2048
N_LAYERS = 8

OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Quantizers
# ----------------------------------------------------------------------

def uniform_quantize_nd(X, bits, clip_std=3.0):
    """Uniform quantization per dimension."""
    n_levels = 2 ** bits
    sigma = X.std(axis=0, keepdims=True) + 1e-8
    clip = clip_std * sigma
    X_clipped = np.clip(X, -clip, clip)
    delta = 2 * clip / n_levels
    idx = np.floor((X_clipped + clip) / delta).astype(int)
    idx = np.clip(idx, 0, n_levels - 1)
    recon = -clip + (idx + 0.5) * delta
    return recon


def lloyd_max_quantize_nd(X, bits, n_iter=50):
    """L² Lloyd-Max quantization per dimension (optimal scalar)."""
    n_levels = 2 ** bits
    d = X.shape[1]
    X_q = np.zeros_like(X)
    for j in range(d):
        col = X[:, j]
        # Init with percentiles
        pcts = np.linspace(0, 100, n_levels + 2)[1:-1]
        centroids = np.percentile(col, pcts)
        centroids = np.sort(centroids)
        for _ in range(n_iter):
            boundaries = (centroids[:-1] + centroids[1:]) / 2
            idx = np.searchsorted(boundaries, col)
            new_centroids = centroids.copy()
            for k in range(n_levels):
                mask = idx == k
                if mask.sum() > 0:
                    new_centroids[k] = col[mask].mean()
            if np.max(np.abs(new_centroids - centroids)) < 1e-6:
                break
            centroids = new_centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        idx = np.searchsorted(boundaries, col)
        X_q[:, j] = centroids[idx]
    return X_q


def spherical_quantize_polar_2d(X, n_angle_bits=3, n_mag_bits=1):
    """
    Spherical quantization: decompose X into 2D blocks, each block → (angle, magnitude) polar.
    Total bits per block = n_angle_bits + n_mag_bits (default 3+1=4, = 2 bits/dim).
    """
    T, d = X.shape
    assert d % 2 == 0, f"d={d} must be even for 2D block decomposition"
    n_blocks = d // 2

    X_q = np.zeros_like(X)
    n_angle = 2 ** n_angle_bits
    n_mag = 2 ** n_mag_bits

    for b in range(n_blocks):
        xy = X[:, 2*b:2*b+2]    # (T, 2)
        r = np.linalg.norm(xy, axis=1)    # (T,)
        theta = np.arctan2(xy[:, 1], xy[:, 0])  # (T,) in [-π, π]

        # Angle quantization: uniform on [-π, π]
        angle_step = 2 * np.pi / n_angle
        theta_idx = np.round((theta + np.pi) / angle_step).astype(int) % n_angle
        theta_q = -np.pi + (theta_idx + 0.5) * angle_step

        # Magnitude quantization: Lloyd-Max on r (median split for n_mag=2)
        if n_mag == 1:
            r_q = np.full_like(r, r.mean())
        elif n_mag == 2:
            r_sorted = np.sort(r)
            # simple 2-level Lloyd
            median_r = np.median(r)
            mask_lo = r <= median_r
            r_lo = r[mask_lo].mean() if mask_lo.any() else median_r * 0.5
            r_hi = r[~mask_lo].mean() if (~mask_lo).any() else median_r * 1.5
            r_q = np.where(mask_lo, r_lo, r_hi)
        else:
            # General Lloyd-Max on magnitude
            pcts = np.linspace(0, 100, n_mag + 2)[1:-1]
            centroids_r = np.percentile(r, pcts)
            for _ in range(20):
                boundaries = (centroids_r[:-1] + centroids_r[1:]) / 2
                idx = np.searchsorted(boundaries, r)
                new_c = centroids_r.copy()
                for k in range(n_mag):
                    m = idx == k
                    if m.sum() > 0:
                        new_c[k] = r[m].mean()
                if np.max(np.abs(new_c - centroids_r)) < 1e-6:
                    break
                centroids_r = new_c
            boundaries = (centroids_r[:-1] + centroids_r[1:]) / 2
            idx = np.searchsorted(boundaries, r)
            r_q = centroids_r[idx]

        X_q[:, 2*b] = r_q * np.cos(theta_q)
        X_q[:, 2*b + 1] = r_q * np.sin(theta_q)

    return X_q


# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------

def main():
    print(f"=== Experiment 2: Spherical Quantizer on {MODEL_NAME} ===")
    t_start = time.time()

    print("Loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device_map=DEVICE,
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t_start:.1f}s", flush=True)

    # Calibration text
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        text = '\n\n'.join([t for t in ds['text'] if len(t.strip()) > 100][:500])
    except Exception as e:
        text = " ".join(["KV-cache quantization with spherical geometry."] * 5000)

    enc = tok(text, return_tensors='pt', truncation=True, max_length=N_CALIB_TOKENS)
    input_ids = enc['input_ids'].to(DEVICE)
    T = input_ids.shape[1]
    print(f"  T={T} calibration tokens", flush=True)

    # Sample layers
    n_total = model.config.num_hidden_layers
    layer_idx_list = np.linspace(2, n_total - 2, N_LAYERS, dtype=int).tolist()
    print(f"  Sampling layers: {layer_idx_list}", flush=True)

    # Hooks to capture pre-RoPE keys and queries
    captured = {}
    def make_khook(li):
        def h(m, i, o):
            captured.setdefault(li, {})['k'] = o.detach().cpu().float().numpy()
        return h
    def make_qhook(li):
        def h(m, i, o):
            captured.setdefault(li, {})['q'] = o.detach().cpu().float().numpy()
        return h
    def make_attn_hook(li):
        def h(m, i, o):
            if isinstance(o, tuple) and len(o) >= 2 and o[1] is not None:
                captured.setdefault(li, {})['attn'] = o[1].detach().cpu().float().numpy()
        return h

    handles = []
    layers = model.model.layers
    for li in layer_idx_list:
        attn_mod = layers[li].self_attn
        handles.append(attn_mod.k_proj.register_forward_hook(make_khook(li)))
        handles.append(attn_mod.q_proj.register_forward_hook(make_qhook(li)))
        handles.append(attn_mod.register_forward_hook(make_attn_hook(li)))

    print("  Forward pass...", flush=True)
    t_fwd = time.time()
    with torch.no_grad():
        _ = model(input_ids, output_attentions=True, use_cache=False)
    print(f"  Forward done in {time.time()-t_fwd:.1f}s", flush=True)

    for h in handles:
        h.remove()

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    print(f"  head_dim={head_dim}, n_q={n_q}, n_kv={n_kv}", flush=True)

    # Delete model to free GPU
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Analysis per (layer, kv_head)
    results = {
        'model': MODEL_NAME,
        'T': T,
        'layers_sampled': layer_idx_list,
        'per_head': [],
    }

    print("\nQuantizer comparison per (layer, kv_head):", flush=True)
    print(f"  (L2 MSE: lower=better; ratio vs Uniform)", flush=True)

    for li in layer_idx_list:
        data = captured.get(li)
        if data is None or 'k' not in data or 'q' not in data or 'attn' not in data:
            continue

        K_all = data['k']    # (1, T, n_kv * d_head)
        Q_all = data['q']    # (1, T, n_q * d_head)
        attn = data['attn']  # (1, n_q, T, T)

        K_all = K_all.reshape(1, T, n_kv, head_dim)[0]  # (T, n_kv, d)
        Q_all = Q_all.reshape(1, T, n_q, head_dim)[0]    # (T, n_q, d)

        for hk in range(n_kv):
            K = K_all[:, hk, :].astype(np.float32)  # (T, d)
            # Center (for PCA)
            K_mean = K.mean(axis=0, keepdims=True)
            K_centered = K - K_mean
            # PCA
            cov = K_centered.T @ K_centered / max(T-1, 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, order]
            K_pca = K_centered @ eigvecs  # (T, d) in PCA basis

            # Three quantizers at 2-bit
            K_unif = uniform_quantize_nd(K_pca, bits=2)
            K_lloyd = lloyd_max_quantize_nd(K_pca, bits=2, n_iter=30)
            K_sph = spherical_quantize_polar_2d(K_pca, n_angle_bits=3, n_mag_bits=1)

            # Rotate back to original basis
            K_unif_full = K_unif @ eigvecs.T + K_mean
            K_lloyd_full = K_lloyd @ eigvecs.T + K_mean
            K_sph_full = K_sph @ eigvecs.T + K_mean

            # Reconstruction MSE
            mse_unif = float(((K - K_unif_full)**2).mean())
            mse_lloyd = float(((K - K_lloyd_full)**2).mean())
            mse_sph = float(((K - K_sph_full)**2).mean())

            # Attention-weighted MSE: weight errors by attention sensitivity
            # For each key position j, weight = Σ_t p_tj * |q_t|^2 ? simpler proxy
            # Use attention probs averaged across query heads associated with this kv head
            n_q_per_kv = n_q // n_kv
            q_heads = list(range(hk * n_q_per_kv, (hk+1) * n_q_per_kv))
            # Average attention over those q heads: (T, T)
            attn_avg = attn[0, q_heads, :, :].mean(axis=0)  # (T_q, T_k)
            # Per-key importance: Σ_t p_t,k  (how much is key k attended to)
            key_importance = attn_avg.sum(axis=0)  # (T,)
            # Attention-weighted MSE
            err_unif = ((K - K_unif_full)**2).sum(axis=1)  # (T,)
            err_lloyd = ((K - K_lloyd_full)**2).sum(axis=1)
            err_sph = ((K - K_sph_full)**2).sum(axis=1)
            awmse_unif = float((key_importance * err_unif).sum() / key_importance.sum())
            awmse_lloyd = float((key_importance * err_lloyd).sum() / key_importance.sum())
            awmse_sph = float((key_importance * err_sph).sum() / key_importance.sum())

            head_result = {
                'layer': int(li),
                'kv_head': int(hk),
                'mse_uniform': mse_unif,
                'mse_lloyd': mse_lloyd,
                'mse_spherical': mse_sph,
                'awmse_uniform': awmse_unif,
                'awmse_lloyd': awmse_lloyd,
                'awmse_spherical': awmse_sph,
                'ratio_lloyd_vs_unif': mse_lloyd / mse_unif,
                'ratio_sph_vs_unif': mse_sph / mse_unif,
                'ratio_sph_vs_lloyd': mse_sph / mse_lloyd,
                'awratio_sph_vs_unif': awmse_sph / awmse_unif,
                'awratio_sph_vs_lloyd': awmse_sph / awmse_lloyd,
            }
            results['per_head'].append(head_result)

            print(f"  L{li:3d}/H{hk}: MSE unif={mse_unif:.4f} lloyd={mse_lloyd:.4f} sph={mse_sph:.4f} "
                  f"(lloyd/unif={mse_lloyd/mse_unif:.3f}, sph/unif={mse_sph/mse_unif:.3f})", flush=True)

    # Summary
    all_lloyd_ratios = [r['ratio_lloyd_vs_unif'] for r in results['per_head']]
    all_sph_ratios = [r['ratio_sph_vs_unif'] for r in results['per_head']]
    all_awsph_ratios = [r['awratio_sph_vs_unif'] for r in results['per_head']]
    all_awsph_vs_lloyd = [r['awratio_sph_vs_lloyd'] for r in results['per_head']]

    print("\n=== Summary ===")
    print(f"n_heads: {len(results['per_head'])}")
    print(f"Lloyd/Unif MSE ratio:  median={np.median(all_lloyd_ratios):.3f}, "
          f"max={max(all_lloyd_ratios):.3f}  (lower better, <1 = Lloyd wins)")
    print(f"Sph/Unif MSE ratio:    median={np.median(all_sph_ratios):.3f}, "
          f"max={max(all_sph_ratios):.3f}  (lower better, <1 = Spherical wins)")
    print(f"Sph/Unif AttnW ratio:  median={np.median(all_awsph_ratios):.3f}, "
          f"max={max(all_awsph_ratios):.3f}  (PPL-relevant)")
    print(f"Sph/Lloyd AttnW ratio: median={np.median(all_awsph_vs_lloyd):.3f}  "
          f"(Spherical vs Lloyd in attention metric)")

    # Count wins
    sph_wins_unif = sum(1 for r in all_sph_ratios if r < 1)
    lloyd_wins_unif = sum(1 for r in all_lloyd_ratios if r < 1)
    print(f"\nSph beats Uniform in MSE: {sph_wins_unif}/{len(all_sph_ratios)}")
    print(f"Lloyd beats Uniform in MSE: {lloyd_wins_unif}/{len(all_lloyd_ratios)}")

    results['summary'] = {
        'n_heads': len(results['per_head']),
        'lloyd_vs_unif_median': float(np.median(all_lloyd_ratios)),
        'sph_vs_unif_median': float(np.median(all_sph_ratios)),
        'awsph_vs_unif_median': float(np.median(all_awsph_ratios)),
        'awsph_vs_lloyd_median': float(np.median(all_awsph_vs_lloyd)),
        'sph_beats_unif_count': sph_wins_unif,
        'lloyd_beats_unif_count': lloyd_wins_unif,
    }

    results['runtime_sec'] = time.time() - t_start
    out_file = OUT_DIR / 'exp2_spherical_quantizer_mistral.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")
    print(f"Total runtime: {results['runtime_sec']:.1f}s")


if __name__ == '__main__':
    main()
