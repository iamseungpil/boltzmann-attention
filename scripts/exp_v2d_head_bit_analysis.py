#!/usr/bin/env python3
"""
V2d: Per-Head WF Bit Allocation Analysis (Mechanistic)
========================================================

Given that exp_v2c showed per-dim WF reduces Mistral 2-bit PPL by 28.8%, this
script answers:
  Q1: Does WF allocate MORE bits to the known-high-κ heads (L1 H6, L2 H3, ...)?
  Q2: How many bits go to dim 0 vs the rest for high-κ heads?
  Q3: Which heads contribute most to the residual +1.696 PPL gap?
       (estimated via per-head mean-squared reconstruction error)

No forward pass beyond one calibration. Fast (~30s).
"""
import json, os, time
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

MODEL_NAME = 'mistralai/Mistral-7B-v0.3'
DTYPE = torch.bfloat16
N_CALIB_TOKENS = 2048
OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')


def lloyd_1d(col, bits, n_iter=20):
    n_levels = 2 ** bits
    if n_levels <= 1:
        return np.array([float(col.mean())], dtype=np.float32)
    pcts = np.linspace(0, 100, n_levels + 2)[1:-1]
    centroids = np.sort(np.percentile(col, pcts)).astype(np.float64)
    for _ in range(n_iter):
        b = (centroids[:-1] + centroids[1:]) / 2
        idx = np.searchsorted(b, col)
        new_c = centroids.copy()
        for k in range(n_levels):
            m = idx == k
            if m.sum() > 0:
                new_c[k] = col[m].mean()
        if np.max(np.abs(new_c - centroids)) < 1e-6:
            break
        centroids = new_c
    return centroids.astype(np.float32)


def wf_alloc(sigma2, total_budget, b_floor=1, b_max=8):
    n = len(sigma2)
    s = np.maximum(sigma2, 1e-12)
    bits = np.zeros(n, dtype=int)
    spent = 0
    while spent < total_budget:
        best_g = -np.inf
        best = None
        for j in range(n):
            if bits[j] == 0:
                if spent + b_floor > total_budget:
                    continue
                g = s[j] * (1.0 - 4.0 ** (-b_floor)) / b_floor
                if g > best_g:
                    best_g, best = g, ('act', j)
            elif bits[j] < b_max:
                g = s[j] * (4.0 ** (-bits[j]) - 4.0 ** (-(bits[j] + 1)))
                if g > best_g:
                    best_g, best = g, ('add', j)
        if best is None:
            break
        op, j = best
        if op == 'act':
            bits[j] = b_floor
            spent += b_floor
        else:
            bits[j] += 1
            spent += 1
    return bits


def quantize_mse(K_pca, bits):
    """Estimate reconstruction MSE given per-dim bit allocation."""
    total_mse = 0.0
    head_dim = K_pca.shape[1]
    for j in range(head_dim):
        b = int(bits[j])
        if b == 0:
            q = np.zeros_like(K_pca[:, j])
        else:
            c = lloyd_1d(K_pca[:, j], b, n_iter=15)
            bnd = (c[:-1] + c[1:]) / 2
            idx = np.searchsorted(bnd, K_pca[:, j])
            q = c[idx]
        total_mse += float(np.mean((K_pca[:, j] - q) ** 2))
    return total_mse / head_dim


def main():
    print("="*70)
    print("V2d: Per-Head WF Bit Allocation Analysis")
    print("="*70, flush=True)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    # Calibrate (wikitext)
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    calib = '\n\n'.join([t for t in ds['text'] if len(t.strip()) > 100][:300])
    ids = tok(calib, return_tensors='pt', truncation=True, max_length=N_CALIB_TOKENS)['input_ids'].to('cuda:0')

    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    # Capture K per layer
    K_per_layer = {}
    def mk(li):
        def h(m, i, o):
            K_per_layer[li] = o.detach().cpu().float().numpy()
        return h
    handles = [model.model.layers[li].self_attn.k_proj.register_forward_hook(mk(li)) for li in range(n_layers)]
    with torch.no_grad():
        _ = model(ids, use_cache=False)
    for h in handles:
        h.remove()

    # Load Mistral alignment data from v2
    with open(OUT_DIR / 'exp_v2_massive_activation_test.json') as f:
        v2 = json.load(f)
    alignment_map = {(h['layer'], h['kv_head']): h['enrichment'] for h in v2['test2_kproj_alignment']}

    # Analyze every (L, H)
    head_stats = []
    for li in range(n_layers):
        K_all = K_per_layer[li].reshape(-1, n_kv, head_dim).astype(np.float32)
        for hk in range(n_kv):
            K = K_all[:, hk, :]
            mean = K.mean(axis=0)
            Kc = K - mean
            cov = (Kc.T @ Kc) / max(K.shape[0] - 1, 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            ev = eigvals[order]
            V = eigvecs[:, order].astype(np.float32)
            K_pca = Kc @ V

            kappa = float(ev[0] / max(ev[-1], 1e-12))
            lam_top_ratio = float(ev[0] / max(np.median(ev), 1e-12))

            bits_uniform = np.full(head_dim, 2, dtype=int)
            bits_wf = wf_alloc(ev, total_budget=2 * head_dim, b_floor=1, b_max=8)

            mse_u = quantize_mse(K_pca, bits_uniform)
            mse_wf = quantize_mse(K_pca, bits_wf)

            head_stats.append({
                'layer': li,
                'kv_head': hk,
                'kappa': kappa,
                'lam_top_over_median': lam_top_ratio,
                'eigvals_top5': ev[:5].tolist(),
                'enrichment': alignment_map.get((li, hk), None),
                'wf_bits_dim0': int(bits_wf[0]),
                'wf_bits_top3_sum': int(bits_wf[:3].sum()),
                'wf_bits_max': int(bits_wf.max()),
                'wf_n_zero_dims': int((bits_wf == 0).sum()),
                'wf_avg_bits': float(bits_wf.mean()),
                'mse_uniform_2b': mse_u,
                'mse_wf_2b': mse_wf,
                'mse_reduction_pct': float((mse_u - mse_wf) / max(mse_u, 1e-12) * 100),
            })

    # Sort by kappa descending
    head_stats.sort(key=lambda x: -x['kappa'])

    # Q1/Q2: Top-15 high-κ heads
    print(f"\n  Top-15 high-κ heads — WF bit allocation:", flush=True)
    print(f"  {'rank':<4}|{'L':<3}|{'H':<3}|{'κ':>10}|{'λ1/med':>9}|{'enrich':>8}|"
          f"{'bits_d0':>8}|{'top3_sum':>9}|{'zero_dims':>10}|{'MSE_red%':>10}", flush=True)
    for i, h in enumerate(head_stats[:15]):
        enr = f"{h['enrichment']:.2f}" if h['enrichment'] is not None else "  —  "
        print(f"  {i+1:<4}|{h['layer']:<3}|{h['kv_head']:<3}|{h['kappa']:>10.1e}|"
              f"{h['lam_top_over_median']:>9.1f}|{enr:>8}|"
              f"{h['wf_bits_dim0']:>8}|{h['wf_bits_top3_sum']:>9}|{h['wf_n_zero_dims']:>10}|"
              f"{h['mse_reduction_pct']:>9.1f}%", flush=True)

    # Q3: Worst residual MSE after WF (these are the heads limiting further gains)
    head_stats_by_wf_mse = sorted(head_stats, key=lambda x: -x['mse_wf_2b'])
    print(f"\n  Top-15 heads with worst residual WF 2-bit MSE (bottleneck):", flush=True)
    print(f"  {'rank':<4}|{'L':<3}|{'H':<3}|{'κ':>10}|{'MSE_u':>11}|{'MSE_wf':>11}|"
          f"{'MSE_red%':>10}", flush=True)
    for i, h in enumerate(head_stats_by_wf_mse[:15]):
        print(f"  {i+1:<4}|{h['layer']:<3}|{h['kv_head']:<3}|{h['kappa']:>10.1e}|"
              f"{h['mse_uniform_2b']:>11.3e}|{h['mse_wf_2b']:>11.3e}|"
              f"{h['mse_reduction_pct']:>9.1f}%", flush=True)

    # Summary stats
    total_u = sum(h['mse_uniform_2b'] for h in head_stats)
    total_wf = sum(h['mse_wf_2b'] for h in head_stats)
    n_heads = len(head_stats)
    n_kappa_big = sum(1 for h in head_stats if h['kappa'] > 1e4)
    kappa_big_bits_d0 = [h['wf_bits_dim0'] for h in head_stats if h['kappa'] > 1e4]
    kappa_small_bits_d0 = [h['wf_bits_dim0'] for h in head_stats if h['kappa'] <= 1e4]

    print(f"\n  === Summary ({n_heads} heads) ===", flush=True)
    print(f"  Total reconstruction MSE (summed across heads):", flush=True)
    print(f"    Uniform 2-bit: {total_u:.3e}", flush=True)
    print(f"    WF 2-bit:      {total_wf:.3e}", flush=True)
    print(f"    Reduction:     {(1 - total_wf/total_u)*100:.1f}%", flush=True)
    print(f"\n  WF bits at dim 0 (the PCA top eigendirection):", flush=True)
    print(f"    high-κ heads (κ>1e4, n={n_kappa_big}):  mean={np.mean(kappa_big_bits_d0):.2f}, "
          f"max={max(kappa_big_bits_d0) if kappa_big_bits_d0 else 0}", flush=True)
    print(f"    normal heads (κ≤1e4, n={n_heads-n_kappa_big}): "
          f"mean={np.mean(kappa_small_bits_d0):.2f}, "
          f"max={max(kappa_small_bits_d0) if kappa_small_bits_d0 else 0}", flush=True)

    # Save
    out = {
        'model': MODEL_NAME,
        'n_layers': n_layers,
        'n_kv': n_kv,
        'head_dim': head_dim,
        'head_stats': head_stats,
        'summary': {
            'total_mse_uniform': total_u,
            'total_mse_wf': total_wf,
            'mse_reduction_pct': float((1 - total_wf/total_u) * 100),
            'n_heads_kappa_gt_1e4': n_kappa_big,
            'mean_bits_d0_high_kappa': float(np.mean(kappa_big_bits_d0)) if kappa_big_bits_d0 else 0,
            'mean_bits_d0_normal': float(np.mean(kappa_small_bits_d0)) if kappa_small_bits_d0 else 0,
        },
    }
    out_file = OUT_DIR / 'exp_v2d_head_bit_analysis.json'
    with open(out_file, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved: {out_file}")
    print(f"Total runtime: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
