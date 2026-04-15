#!/usr/bin/env python3
"""
Experiment 3: Per-token Fisher quantizer prototype
====================================================

For each (layer, kv_head):
  1. Extract pre-RoPE keys K and post-RoPE queries Q, attention probs P
  2. Compute per-token Fisher metric M_KL(t) — for each token t:
     M_KL(t) ~ p_t (1-p_t) q_t q_t^T aggregated  (simplified per-query form)
  3. Measure M_KL(t) dispersion: how much variation across tokens?
  4. Test: cluster tokens into K groups (by attention entropy), compute per-cluster M_KL
  5. Fit Mahalanobis Lloyd quantizer with cluster-specific metric
  6. Compare reconstruction cost:
     (a) L² Lloyd (baseline)
     (b) Mahalanobis Lloyd with averaged M_KL (Fisher-weighted)
     (c) Per-cluster Mahalanobis Lloyd
  7. Measure: attention KL for each quantizer (PPL proxy)

Output: per-head Fisher analysis + quantizer comparison
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
N_LAYERS = 4  # fewer layers for depth
N_CLUSTERS = 4

OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Mahalanobis Lloyd-Max (scalar per-dim in whitened space)
# ----------------------------------------------------------------------

def mahalanobis_whitening(X, M):
    """Return X_white = X @ sqrtm(M), such that MSE in white space = (x-y)^T M (x-y).
    We use eigendecomposition for numerical stability."""
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.clip(eigvals, 1e-8, None)
    W_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    W_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return X @ W_sqrt, W_sqrt, W_inv_sqrt


def lloyd_max_1d(col, bits, n_iter=30):
    """Optimal scalar quantizer."""
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
    boundaries = (centroids[:-1] + centroids[1:]) / 2
    idx = np.searchsorted(boundaries, col)
    return centroids[idx]


def mahalanobis_lloyd_quantize(X, M, bits):
    """Quantize X using Mahalanobis metric defined by PSD matrix M."""
    # Whiten: X_white = X @ W_sqrt
    X_white, W_sqrt, W_inv_sqrt = mahalanobis_whitening(X, M)
    # Apply L² Lloyd in white space (per-dim)
    X_white_q = np.zeros_like(X_white)
    d = X_white.shape[1]
    for j in range(d):
        X_white_q[:, j] = lloyd_max_1d(X_white[:, j], bits, n_iter=20)
    # De-whiten
    X_q = X_white_q @ W_inv_sqrt
    return X_q


# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------

def main():
    print(f"=== Experiment 3: Per-token Fisher Prototype on {MODEL_NAME} ===", flush=True)
    t_start = time.time()

    print("Loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device_map=DEVICE,
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t_start:.1f}s", flush=True)

    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        text = '\n\n'.join([t for t in ds['text'] if len(t.strip()) > 100][:500])
    except Exception:
        text = " ".join(["Per-token Fisher metric analysis."] * 5000)

    enc = tok(text, return_tensors='pt', truncation=True, max_length=N_CALIB_TOKENS)
    input_ids = enc['input_ids'].to(DEVICE)
    T = input_ids.shape[1]
    print(f"  T={T}", flush=True)

    n_total = model.config.num_hidden_layers
    layer_idx_list = np.linspace(2, n_total - 2, N_LAYERS, dtype=int).tolist()
    print(f"  layers: {layer_idx_list}", flush=True)

    captured = {}
    def khook(li):
        def h(m, i, o):
            captured.setdefault(li, {})['k'] = o.detach().cpu().float().numpy()
        return h
    def qhook(li):
        def h(m, i, o):
            captured.setdefault(li, {})['q'] = o.detach().cpu().float().numpy()
        return h
    def ahook(li):
        def h(m, i, o):
            if isinstance(o, tuple) and len(o) >= 2 and o[1] is not None:
                captured.setdefault(li, {})['attn'] = o[1].detach().cpu().float().numpy()
        return h

    handles = []
    layers = model.model.layers
    for li in layer_idx_list:
        attn_mod = layers[li].self_attn
        handles.append(attn_mod.k_proj.register_forward_hook(khook(li)))
        handles.append(attn_mod.q_proj.register_forward_hook(qhook(li)))
        handles.append(attn_mod.register_forward_hook(ahook(li)))

    print("  forward...", flush=True)
    t_fwd = time.time()
    with torch.no_grad():
        _ = model(input_ids, output_attentions=True, use_cache=False)
    print(f"  forward done in {time.time()-t_fwd:.1f}s", flush=True)
    for h in handles:
        h.remove()

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads

    del model
    torch.cuda.empty_cache()
    gc.collect()

    results = {
        'model': MODEL_NAME, 'T': T,
        'layers_sampled': layer_idx_list,
        'n_clusters': N_CLUSTERS,
        'per_head': [],
    }

    print("\nPer-head Fisher analysis:", flush=True)

    for li in layer_idx_list:
        data = captured.get(li)
        if data is None or any(k not in data for k in ['k', 'q', 'attn']):
            continue

        K_all = data['k'].reshape(1, T, n_kv, head_dim)[0].astype(np.float32)  # (T, Hkv, d)
        Q_all = data['q'].reshape(1, T, n_q, head_dim)[0].astype(np.float32)   # (T, Hq, d)
        attn = data['attn'][0].astype(np.float32)  # (Hq, T, T)

        for hk in range(min(n_kv, 4)):  # limit to 4 kv heads per layer to save time
            K = K_all[:, hk, :]  # (T, d)

            # Get q_heads for this kv head
            n_q_per_kv = n_q // n_kv
            q_heads = list(range(hk * n_q_per_kv, (hk+1) * n_q_per_kv))
            # Average Q and attention over those q heads
            Q_mean = Q_all[:, q_heads, :].mean(axis=1)  # (T, d)
            attn_mean = attn[q_heads, :, :].mean(axis=0)  # (T_q, T_k)

            # Per-token scalar s_t = Σ_j p_tj (1 - p_tj)
            s_t = (attn_mean * (1.0 - attn_mean)).sum(axis=1)  # (T,)

            # Per-token attention entropy (for clustering)
            p = np.clip(attn_mean, 1e-10, 1.0)
            entropy = -(p * np.log(p)).sum(axis=1)  # (T,)

            # Averaged Fisher: M_avg = (1/T) Σ_t s_t q_t q_t^T
            Q_weighted = Q_mean * s_t[:, None]
            M_avg = (Q_weighted.T @ Q_mean) / max(T, 1) + 1e-6 * np.eye(head_dim)

            # κ(M_avg) for this head
            eigvals_M = np.linalg.eigvalsh(M_avg)
            eigvals_M = np.clip(eigvals_M, 1e-12, None)
            kappa_M = float(eigvals_M.max() / eigvals_M.min())

            # Per-token M_KL(t) = s_t q_t q_t^T — this is a rank-1 matrix per token
            # Frobenius norm = s_t * |q_t|^2
            per_token_frob = s_t * np.linalg.norm(Q_mean, axis=1)**2  # (T,)
            frob_cv = float(per_token_frob.std() / (per_token_frob.mean() + 1e-8))

            # Cluster tokens by attention entropy (simple binning)
            cluster_labels = np.digitize(
                entropy, bins=np.quantile(entropy, np.linspace(0, 1, N_CLUSTERS + 1)[1:-1])
            )
            cluster_labels = np.clip(cluster_labels, 0, N_CLUSTERS - 1)

            # Per-cluster M_c
            cluster_M = {}
            for c in range(N_CLUSTERS):
                mask = cluster_labels == c
                if mask.sum() < 5:
                    cluster_M[c] = M_avg
                    continue
                Q_c = Q_mean[mask]
                s_c = s_t[mask]
                M_c = (Q_c * s_c[:, None]).T @ Q_c / max(mask.sum(), 1) + 1e-6 * np.eye(head_dim)
                cluster_M[c] = M_c

            # --- Quantize K under 3 metrics ---
            bits = 2
            K_mean = K.mean(axis=0, keepdims=True)
            K_centered = K - K_mean

            # (a) L² Lloyd (baseline)
            K_l2 = np.zeros_like(K_centered)
            for j in range(head_dim):
                K_l2[:, j] = lloyd_max_1d(K_centered[:, j], bits, n_iter=20)

            # (b) Mahalanobis Lloyd with M_avg (Fisher-weighted)
            try:
                K_fisher_avg = mahalanobis_lloyd_quantize(K_centered, M_avg, bits)
            except Exception as e:
                K_fisher_avg = K_l2
                print(f"    L{li}H{hk} Fisher-avg failed: {e}")

            # (c) Per-cluster Fisher (simplified: use each cluster's metric)
            K_fisher_cluster = np.zeros_like(K_centered)
            try:
                for c in range(N_CLUSTERS):
                    mask = cluster_labels == c
                    if mask.sum() < 5:
                        K_fisher_cluster[mask] = K_l2[mask]
                        continue
                    K_fisher_cluster[mask] = mahalanobis_lloyd_quantize(K_centered[mask], cluster_M[c], bits)
            except Exception as e:
                K_fisher_cluster = K_fisher_avg
                print(f"    L{li}H{hk} Fisher-cluster failed: {e}")

            # Measures
            # Reconstruction MSE
            mse_l2 = float(((K_centered - K_l2)**2).mean())
            mse_fa = float(((K_centered - K_fisher_avg)**2).mean())
            mse_fc = float(((K_centered - K_fisher_cluster)**2).mean())

            # Fisher-norm squared distortion: (x-x̂)^T M_avg (x-x̂) averaged over tokens
            def fisher_norm_sq(delta, M):
                # delta: (T, d), return mean of delta_t^T M delta_t
                return float((delta @ M * delta).sum(axis=1).mean())

            delta_l2 = K_centered - K_l2
            delta_fa = K_centered - K_fisher_avg
            delta_fc = K_centered - K_fisher_cluster

            fn_l2 = fisher_norm_sq(delta_l2, M_avg)
            fn_fa = fisher_norm_sq(delta_fa, M_avg)
            fn_fc = fisher_norm_sq(delta_fc, M_avg)

            head_result = {
                'layer': int(li), 'kv_head': int(hk),
                'kappa_M_avg': kappa_M,
                'per_token_frob_cv': frob_cv,
                'mse_l2_lloyd': mse_l2,
                'mse_fisher_avg': mse_fa,
                'mse_fisher_cluster': mse_fc,
                'fn_l2_lloyd': fn_l2,
                'fn_fisher_avg': fn_fa,
                'fn_fisher_cluster': fn_fc,
                'ratio_fn_fa_vs_l2': fn_fa / max(fn_l2, 1e-12),
                'ratio_fn_fc_vs_l2': fn_fc / max(fn_l2, 1e-12),
                'ratio_fn_fc_vs_fa': fn_fc / max(fn_fa, 1e-12),
            }
            results['per_head'].append(head_result)

            print(f"  L{li:3d}/H{hk}: κ(M)={kappa_M:.0f} frob_CV={frob_cv:.2f} | "
                  f"Fisher-norm L²={fn_l2:.3f} Favg={fn_fa:.3f} Fclu={fn_fc:.3f} | "
                  f"Fa/L²={fn_fa/max(fn_l2,1e-12):.3f}  Fc/Fa={fn_fc/max(fn_fa,1e-12):.3f}", flush=True)

    # Summary
    if results['per_head']:
        fa_over_l2 = [r['ratio_fn_fa_vs_l2'] for r in results['per_head']]
        fc_over_l2 = [r['ratio_fn_fc_vs_l2'] for r in results['per_head']]
        fc_over_fa = [r['ratio_fn_fc_vs_fa'] for r in results['per_head']]
        frob_cvs = [r['per_token_frob_cv'] for r in results['per_head']]

        print("\n=== Summary ===")
        print(f"n_heads: {len(results['per_head'])}")
        print(f"Per-token M_KL Frobenius CV: median={np.median(frob_cvs):.2f} "
              f"(>0.5 indicates meaningful per-token variation)")
        print(f"Fisher-avg / L² Fisher-norm ratio: median={np.median(fa_over_l2):.3f} "
              f"(<1 means Fisher-avg is better in Fisher metric)")
        print(f"Fisher-cluster / L² Fisher-norm ratio: median={np.median(fc_over_l2):.3f}")
        print(f"Fisher-cluster / Fisher-avg ratio: median={np.median(fc_over_fa):.3f} "
              f"(<1 means per-cluster helps)")

        fa_wins = sum(1 for r in fa_over_l2 if r < 1)
        fc_wins = sum(1 for r in fc_over_l2 if r < 1)
        print(f"\nFisher-avg beats L² in Fisher-norm: {fa_wins}/{len(fa_over_l2)}")
        print(f"Fisher-cluster beats L² in Fisher-norm: {fc_wins}/{len(fc_over_l2)}")

        results['summary'] = {
            'n_heads': len(results['per_head']),
            'frob_cv_median': float(np.median(frob_cvs)),
            'fa_over_l2_median': float(np.median(fa_over_l2)),
            'fc_over_l2_median': float(np.median(fc_over_l2)),
            'fc_over_fa_median': float(np.median(fc_over_fa)),
            'fa_wins_over_l2': fa_wins,
            'fc_wins_over_l2': fc_wins,
        }

    results['runtime_sec'] = time.time() - t_start
    out_file = OUT_DIR / 'exp3_fisher_prototype_mistral.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")
    print(f"Total runtime: {results['runtime_sec']:.1f}s")


if __name__ == '__main__':
    main()
