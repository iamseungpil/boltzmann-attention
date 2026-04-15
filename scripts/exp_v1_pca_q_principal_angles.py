#!/usr/bin/env python3
"""
V1: PCA-Q Principal Angle Measurement (Subspace-Level)
=======================================================

Critical for paper acceptance upgrade (5.5 → 7/10 per coworker review).

Previous measurement (exp_verify_qwwf_alignment_proof.json):
  - Spearman ρ(λ_K, σ_Q²) = 0.655 median (rank correlation of eigenvalues)

This was ALIGNMENT OF EIGENVALUE ORDERING, not SUBSPACE ALIGNMENT.

Correct measurement:
  - For each (layer, head), compute Σ_K and Σ_Q (d × d each)
  - Top-k eigenvectors of Σ_K: V_K ∈ R^(d × k)
  - Top-k eigenvectors of Σ_Q: V_Q ∈ R^(d × k)
  - Principal angles: arccos(singular values of V_K^T V_Q)
  - Return: mean principal angle across k directions

If all principal angles are small (< 5°), this is genuine subspace alignment —
a strong structural discovery about trained transformers.

If angles are large or high variance, it's only rank correlation (what we had).

Output: JSON with per-(model, layer, head) principal angle statistics.
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

DEVICE = 'cuda:1'
DTYPE = torch.bfloat16
N_CALIB_TOKENS = 2048   # larger for stable covariance estimation
K_VALUES = [8, 16, 32, 64]  # different subspace dimensions to test
N_LAYERS_SAMPLED = 10

MODELS = [
    ('mistralai/Mistral-7B-v0.3', 'mistral-7b'),
    ('Qwen/Qwen2.5-7B', 'qwen2.5-7b'),
    ('Qwen/Qwen2.5-1.5B', 'qwen2.5-1.5b'),  # smaller, fast sanity
]

OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def principal_angles(V_K, V_Q):
    """
    Compute principal angles between two k-dim subspaces.

    Args:
        V_K: (d, k) orthonormal basis
        V_Q: (d, k) orthonormal basis

    Returns:
        angles: (k,) array of angles in degrees (ascending order)
    """
    # Cross-Gram matrix
    M = V_K.T @ V_Q   # (k, k)
    singular_values = np.linalg.svd(M, compute_uv=False)
    # Clip to [-1, 1] for numerical safety
    singular_values = np.clip(singular_values, -1.0, 1.0)
    # Principal angles = arccos of singular values
    angles_rad = np.arccos(np.abs(singular_values))
    angles_deg = np.degrees(angles_rad)
    return np.sort(angles_deg)  # ascending (smallest angle first)


def compute_top_k_eigenvectors(cov, k):
    """Compute top-k eigenvectors of covariance matrix."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    order = np.argsort(eigvals)[::-1]
    V = eigvecs[:, order[:k]]
    top_eigvals = eigvals[order[:k]]
    return V, top_eigvals


def collect_k_and_q(model, input_ids, n_layers, n_kv, n_q, head_dim):
    """Capture K and Q projections per (layer, head)."""
    captured = {}
    handles = []

    def kh(li):
        def h(m, i, o):
            captured.setdefault(li, {})['k'] = o.detach().cpu().float().numpy()
        return h
    def qh(li):
        def h(m, i, o):
            captured.setdefault(li, {})['q'] = o.detach().cpu().float().numpy()
        return h

    for li in range(n_layers):
        mod = model.model.layers[li].self_attn
        handles.append(mod.k_proj.register_forward_hook(kh(li)))
        handles.append(mod.q_proj.register_forward_hook(qh(li)))

    with torch.no_grad():
        _ = model(input_ids, use_cache=False)
    for h in handles:
        h.remove()

    return captured


def get_calib_text():
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        texts = [t for t in ds['text'] if len(t.strip()) > 100]
        return '\n\n'.join(texts[:500])
    except Exception:
        return " ".join(["Calibration text for PCA-Q measurement."] * 5000)


def analyze_model(model_name, short_name):
    print(f"\n{'='*70}")
    print(f"V1 Principal Angle Measurement: {model_name}")
    print(f"{'='*70}", flush=True)
    t_start = time.time()

    print("Loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=DTYPE, device_map=DEVICE,
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t_start:.1f}s", flush=True)

    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_q
    print(f"  n_layers={n_layers}, n_kv={n_kv}, n_q={n_q}, head_dim={head_dim}", flush=True)

    # Sample layers
    layer_idx_list = np.linspace(2, n_layers - 2, N_LAYERS_SAMPLED, dtype=int).tolist()
    print(f"  Analyzing {N_LAYERS_SAMPLED} layers: {layer_idx_list}", flush=True)

    # Calibration
    text = get_calib_text()
    enc = tok(text, return_tensors='pt', truncation=True, max_length=N_CALIB_TOKENS)
    input_ids = enc['input_ids'].to(DEVICE)
    T = input_ids.shape[1]
    print(f"  Calibration: T={T} tokens", flush=True)

    print("  Forward pass (no attention output, K/Q only)...", flush=True)
    t_fwd = time.time()
    captured = collect_k_and_q(model, input_ids, n_layers, n_kv, n_q, head_dim)
    print(f"  Forward done in {time.time()-t_fwd:.1f}s", flush=True)

    # Free model
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Compute principal angles per (layer, kv_head)
    # For each KV head, the "effective" query covariance is from the associated q heads
    n_q_per_kv = n_q // n_kv
    per_head_results = []

    print(f"\n  Computing principal angles per (layer, kv_head)...", flush=True)
    for li in layer_idx_list:
        data = captured.get(li, {})
        if 'k' not in data or 'q' not in data:
            continue

        K_all = data['k'].reshape(T, n_kv, head_dim).astype(np.float32)
        Q_all = data['q'].reshape(T, n_q, head_dim).astype(np.float32)

        for hk in range(n_kv):
            K = K_all[:, hk, :]
            # Average over associated q heads (GQA)
            q_heads = list(range(hk * n_q_per_kv, (hk + 1) * n_q_per_kv))
            Q = Q_all[:, q_heads, :].mean(axis=1)

            # Center
            K_c = K - K.mean(axis=0, keepdims=True)
            Q_c = Q - Q.mean(axis=0, keepdims=True)

            # Covariances
            Sigma_K = (K_c.T @ K_c) / max(T - 1, 1)
            Sigma_Q = (Q_c.T @ Q_c) / max(T - 1, 1)

            # Per k: compute top-k eigenvectors + principal angles
            angles_by_k = {}
            eigvals_K_full, _ = np.linalg.eigh(Sigma_K)
            eigvals_K_desc = np.sort(eigvals_K_full)[::-1]

            for k in K_VALUES:
                if k > head_dim:
                    continue
                V_K, _ = compute_top_k_eigenvectors(Sigma_K, k)
                V_Q, _ = compute_top_k_eigenvectors(Sigma_Q, k)
                angles = principal_angles(V_K, V_Q)

                angles_by_k[k] = {
                    'mean': float(np.mean(angles)),
                    'median': float(np.median(angles)),
                    'min': float(angles.min()),
                    'max': float(angles.max()),
                    'p95': float(np.percentile(angles, 95)),
                    'angles_list': angles.tolist(),
                }

            # Also compute full-rank angles (all d directions)
            V_K_full = np.linalg.eigh(Sigma_K)[1][:, ::-1]
            V_Q_full = np.linalg.eigh(Sigma_Q)[1][:, ::-1]
            full_angles = principal_angles(V_K_full, V_Q_full)

            per_head_results.append({
                'layer': int(li),
                'kv_head': int(hk),
                'n_tokens': int(T),
                'angles_by_k': angles_by_k,
                'full_rank': {
                    'mean': float(np.mean(full_angles)),
                    'median': float(np.median(full_angles)),
                    'min': float(full_angles.min()),
                    'max': float(full_angles.max()),
                    'p95': float(np.percentile(full_angles, 95)),
                },
                'eigval_K_decay_ratio_top1_top10': float(eigvals_K_desc[0] / max(eigvals_K_desc[min(10, head_dim-1)], 1e-12)),
            })

    # Aggregate statistics per k
    aggregate_by_k = {}
    for k in K_VALUES:
        if k > head_dim:
            continue
        means = []
        medians = []
        maxes = []
        for r in per_head_results:
            if k in r['angles_by_k']:
                means.append(r['angles_by_k'][k]['mean'])
                medians.append(r['angles_by_k'][k]['median'])
                maxes.append(r['angles_by_k'][k]['max'])
        aggregate_by_k[k] = {
            'mean_of_means': float(np.mean(means)),
            'median_of_means': float(np.median(means)),
            'p95_of_means': float(np.percentile(means, 95)) if len(means) > 1 else float(means[0]),
            'max_of_means': float(np.max(means)),
            'mean_of_medians': float(np.mean(medians)),
            'mean_of_max': float(np.mean(maxes)),
            'n_samples': len(means),
        }

    full_rank_stats = {
        'mean': float(np.mean([r['full_rank']['mean'] for r in per_head_results])),
        'median': float(np.median([r['full_rank']['median'] for r in per_head_results])),
        'max_of_max': float(np.max([r['full_rank']['max'] for r in per_head_results])),
    }

    result = {
        'model': model_name,
        'short_name': short_name,
        'n_layers': n_layers,
        'n_layers_sampled': N_LAYERS_SAMPLED,
        'layer_idx_list': layer_idx_list,
        'n_kv_heads': n_kv,
        'head_dim': head_dim,
        'T_tokens': T,
        'per_head': per_head_results,
        'aggregate_by_k': aggregate_by_k,
        'full_rank_stats': full_rank_stats,
        'runtime_sec': time.time() - t_start,
    }

    # Print summary
    print(f"\n  === {short_name} Principal Angle Summary ===")
    for k, stats in aggregate_by_k.items():
        print(f"  k={k:3d}: mean={stats['mean_of_means']:5.2f}°, "
              f"median={stats['median_of_means']:5.2f}°, "
              f"p95={stats['p95_of_means']:5.2f}°, "
              f"max={stats['max_of_means']:5.2f}°", flush=True)
    print(f"  full_rank: mean={full_rank_stats['mean']:.2f}°, median={full_rank_stats['median']:.2f}°")

    return result


def main():
    print("=" * 70)
    print("V1: PCA-Q Principal Angle Measurement")
    print("(Critical for paper acceptance upgrade 5.5 → 7/10)")
    print("=" * 70, flush=True)
    t_total = time.time()

    all_results = {}
    for model_name, short_name in MODELS:
        try:
            res = analyze_model(model_name, short_name)
            all_results[short_name] = res
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results[short_name] = {'error': str(e)}

    # Cross-model summary
    print("\n" + "=" * 70)
    print("CROSS-MODEL SUMMARY")
    print("=" * 70)
    print(f"{'Model':<18} | " + " | ".join([f'k={k:3d} mean' for k in K_VALUES]) + " | full_rank mean")
    print('-' * 90)
    for short, res in all_results.items():
        if 'error' in res:
            print(f"{short:<18} | ERROR")
            continue
        row = [short]
        for k in K_VALUES:
            if k in res['aggregate_by_k']:
                row.append(f"{res['aggregate_by_k'][k]['mean_of_means']:7.2f}°")
            else:
                row.append('   N/A ')
        row.append(f"{res['full_rank_stats']['mean']:7.2f}°")
        print(' | '.join([f'{row[0]:<18}'] + row[1:]))

    # Verdict
    print("\n=== Verdict ===")
    print("Target (subspace-level alignment): mean principal angle < 5° for small k")
    print("If confirmed → strong structural discovery, paper upgrade to 7/10")
    print("If angles > 10° → only weak alignment, paper stays 5.5/10")

    # Save
    all_results['_meta'] = {
        'total_runtime_sec': time.time() - t_total,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'K_VALUES': K_VALUES,
        'N_CALIB_TOKENS': N_CALIB_TOKENS,
        'N_LAYERS_SAMPLED': N_LAYERS_SAMPLED,
    }
    out_file = OUT_DIR / 'exp_v1_pca_q_principal_angles.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_file}")
    print(f"Total runtime: {time.time()-t_total:.0f}s ({(time.time()-t_total)/60:.1f}m)")


if __name__ == '__main__':
    main()
