#!/usr/bin/env python3
"""
Next-4: PCA-unified comprehensive PPL measurement on Mistral-7B
================================================================

Fixes the critical issue in Next-1/Next-2: MISSING PCA ROTATION.
v3 uses Pre-RoPE PCA + quantizer. My previous scripts quantized k_proj
output directly, leading to catastrophic PPL (243, 37, etc.) that are
not comparable to v3's 6.46 scale.

This script adds per-(layer, head) Pre-RoPE PCA before quantization.
All configs use the same PCA basis, differing only in quantizer/bit allocation.

Configs tested:
  A. FP16 baseline
  B. PCA + Uniform 2-bit (v3 reference; expect ~6.46)
  C. PCA + L² Lloyd 2-bit (v3 catastrophe; expect ~32.68)
  D. PCA + Fisher-avg Mahalanobis Lloyd 2-bit (our method)
  E. PCA + L² Lloyd, layer 2-6 @ 3-bit, others @ 2-bit (outlier 3b)
  F. PCA + L² Lloyd, layer 2-6 @ 4-bit, others @ 2-bit (outlier 4b)
  G. PCA + L² Lloyd, layer 2 ONLY @ 4-bit (minimum cost)
  H. PCA + L² Lloyd, layer 2-6 @ 8-bit (upper bound)
  I. PCA + Fisher-Mahalanobis Lloyd, layer 2-6 @ 4-bit (combined fix)

Critical questions:
  - Does Fisher Mahalanobis beat L² Lloyd in PCA basis? (D vs C)
  - Does outlier layer preservation work with PCA? (E,F,G vs B)
  - Can we beat v3's Uniform 2-bit (6.46) at similar or lower bit budget?
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
OUTLIER_LAYERS = [2, 3, 4, 5, 6]

OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Lloyd-Max 1D
# ----------------------------------------------------------------------

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


def lloyd_apply(col, centroids):
    boundaries = (centroids[:-1] + centroids[1:]) / 2
    idx = np.searchsorted(boundaries, col)
    return centroids[idx]


def uniform_quantize_1d(col, bits, clip_std=3.0):
    n_levels = 2 ** bits
    sigma = col.std() + 1e-8
    clip = clip_std * sigma
    col_c = np.clip(col, -clip, clip)
    delta = 2 * clip / n_levels
    idx = np.floor((col_c + clip) / delta).astype(int)
    idx = np.clip(idx, 0, n_levels - 1)
    return -clip + (idx + 0.5) * delta


# ----------------------------------------------------------------------
# Per-head quantizer data (PCA + Lloyd)
# ----------------------------------------------------------------------

def fit_head_pca(K):
    """K: (T, d). Return (mean, V) where V is eigenvectors (d, d) descending."""
    K = K.astype(np.float32)
    K_mean = K.mean(axis=0)
    K_c = K - K_mean
    cov = (K_c.T @ K_c) / max(K.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    V = eigvecs[:, order]
    return K_mean, V


def fit_l2_lloyd_in_pca(K_pca, bits):
    """Fit Lloyd centroids per dim in PCA basis."""
    d = K_pca.shape[1]
    centroids = np.zeros((d, 2 ** bits), dtype=np.float32)
    for j in range(d):
        centroids[j] = lloyd_max_1d_fit(K_pca[:, j], bits, n_iter=20)
    return centroids


def fit_mahalanobis_lloyd_in_pca(K_pca, Q, attn, head_dim, bits):
    """
    Fit Fisher-avg Mahalanobis Lloyd in PCA basis.
    Fisher metric is computed from Q (queries) projected into same PCA basis.
    Returns: (W_sqrt, W_inv_sqrt, centroids_in_white_space)
    """
    # Fisher metric M = (1/T) Σ_t s_t q_t q_t^T  where s_t = Σ_j p_tj(1-p_tj)
    s_t = (attn * (1.0 - attn)).sum(axis=1)  # (T,)
    M = ((Q * s_t[:, None]).T @ Q) / max(Q.shape[0], 1)
    M = M + 1e-6 * np.eye(head_dim, dtype=np.float32)
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.clip(eigvals, 1e-8, None)
    sqrt_eig = np.sqrt(eigvals)
    W_sqrt = (eigvecs * sqrt_eig) @ eigvecs.T
    W_inv_sqrt = (eigvecs * (1.0 / sqrt_eig)) @ eigvecs.T

    K_white = K_pca @ W_sqrt
    centroids = np.zeros((head_dim, 2 ** bits), dtype=np.float32)
    for j in range(head_dim):
        centroids[j] = lloyd_max_1d_fit(K_white[:, j], bits, n_iter=20)
    return W_sqrt.astype(np.float32), W_inv_sqrt.astype(np.float32), centroids


# ----------------------------------------------------------------------
# Universal PCA-based quantization hook
# ----------------------------------------------------------------------

class PCAQuantHook:
    """Generic PCA-based quantization hook.

    Each kv_head has an entry in `head_data` describing the quantization:
    {
      'mode': 'uniform' | 'l2_lloyd' | 'mahalanobis' | 'fp16',
      'K_mean': (d,),
      'V': (d, d),             # PCA eigenvectors
      'centroids': (d, n_lvl), # For l2_lloyd or mahalanobis (in white space)
      'W_sqrt': (d, d),        # For mahalanobis only
      'W_inv_sqrt': (d, d),    # For mahalanobis only
      'bits': int,             # For uniform only
    }
    """
    def __init__(self, head_data, n_kv, head_dim):
        self.head_data = head_data
        self.n_kv = n_kv
        self.head_dim = head_dim

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x = output.view(B, T, self.n_kv, self.head_dim)
        x_np = x.float().cpu().numpy()
        x_q = np.zeros_like(x_np)

        for hk in range(self.n_kv):
            hd = self.head_data[hk]
            mode = hd['mode']
            if mode == 'fp16':
                x_q[:, :, hk, :] = x_np[:, :, hk, :]
                continue

            data = x_np[:, :, hk, :]    # (B, T, d)
            shape = data.shape
            K_flat = data.reshape(-1, self.head_dim)
            # Center + rotate to PCA basis
            K_pca = (K_flat - hd['K_mean']) @ hd['V']   # (N, d)

            if mode == 'uniform':
                bits = hd['bits']
                K_pca_q = np.zeros_like(K_pca)
                for j in range(self.head_dim):
                    K_pca_q[:, j] = uniform_quantize_1d(K_pca[:, j], bits)
            elif mode == 'l2_lloyd':
                c = hd['centroids']
                K_pca_q = np.zeros_like(K_pca)
                for j in range(self.head_dim):
                    K_pca_q[:, j] = lloyd_apply(K_pca[:, j], c[j])
            elif mode == 'mahalanobis':
                c = hd['centroids']
                K_white = K_pca @ hd['W_sqrt']
                K_white_q = np.zeros_like(K_white)
                for j in range(self.head_dim):
                    K_white_q[:, j] = lloyd_apply(K_white[:, j], c[j])
                K_pca_q = K_white_q @ hd['W_inv_sqrt']
            else:
                K_pca_q = K_pca

            # De-rotate + un-center
            K_recon = K_pca_q @ hd['V'].T + hd['K_mean']
            x_q[:, :, hk, :] = K_recon.reshape(shape)

        return torch.from_numpy(x_q).view(B, T, self.n_kv * self.head_dim).to(output.device).to(output.dtype)


# ----------------------------------------------------------------------
# Calibration & fitting
# ----------------------------------------------------------------------

def get_texts(tok):
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        texts = [t for t in ds['text'] if len(t.strip()) > 100]
        return '\n\n'.join(texts[:300]), '\n\n'.join(texts[300:600])
    except Exception:
        return " ".join(["Calib."] * 5000), " ".join(["Eval."] * 5000)


def collect_calib_data(model, tok, calib_text, n_layers, n_kv, n_q, head_dim):
    """Single calibration pass: capture K, Q, attention per layer."""
    print("  Calibration forward pass...", flush=True)
    enc = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB_TOKENS)
    input_ids = enc['input_ids'].to(DEVICE)

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
    def ah(li):
        def h(m, i, o):
            if isinstance(o, tuple) and len(o) >= 2 and o[1] is not None:
                captured.setdefault(li, {})['attn'] = o[1].detach().cpu().float().numpy()
        return h

    for li in range(n_layers):
        mod = model.model.layers[li].self_attn
        handles.append(mod.k_proj.register_forward_hook(kh(li)))
        handles.append(mod.q_proj.register_forward_hook(qh(li)))
        handles.append(mod.register_forward_hook(ah(li)))

    t0 = time.time()
    with torch.no_grad():
        _ = model(input_ids, output_attentions=True, use_cache=False)
    print(f"  Calib forward done in {time.time()-t0:.1f}s", flush=True)
    for h in handles:
        h.remove()

    T = input_ids.shape[1]
    return captured, T


def build_head_data(captured, T, n_layers, n_kv, n_q, head_dim, config_name):
    """
    Build per-layer head_data for a given config.
    config_name determines mode and bits per layer.
    """
    n_q_per_kv = n_q // n_kv
    per_layer = {}

    for li in range(n_layers):
        data = captured.get(li)
        if not data or any(k not in data for k in ['k', 'q', 'attn']):
            continue

        K_all = data['k'].reshape(T, n_kv, head_dim).astype(np.float32)
        Q_all = data['q'].reshape(T, n_q, head_dim).astype(np.float32)
        attn_all = data['attn'][0]

        head_data_list = []
        for hk in range(n_kv):
            K = K_all[:, hk, :]
            # Fit PCA
            K_mean, V = fit_head_pca(K)
            K_pca = (K - K_mean) @ V

            # Determine mode and bits for this (layer, head)
            mode, bits = resolve_config(config_name, li)

            entry = {'mode': mode, 'K_mean': K_mean, 'V': V, 'bits': bits}

            if mode == 'l2_lloyd':
                entry['centroids'] = fit_l2_lloyd_in_pca(K_pca, bits)
            elif mode == 'mahalanobis':
                q_heads = list(range(hk * n_q_per_kv, (hk+1) * n_q_per_kv))
                Q = Q_all[:, q_heads, :].mean(axis=1)
                attn = attn_all[q_heads, :, :].mean(axis=0)
                # Rotate Q into PCA basis (same V as keys)
                Q_pca = (Q - Q.mean(axis=0)) @ V
                W_sqrt, W_inv_sqrt, centroids = fit_mahalanobis_lloyd_in_pca(
                    K_pca, Q_pca, attn, head_dim, bits
                )
                entry['W_sqrt'] = W_sqrt
                entry['W_inv_sqrt'] = W_inv_sqrt
                entry['centroids'] = centroids

            head_data_list.append(entry)

        per_layer[li] = head_data_list

    return per_layer


def resolve_config(name, layer):
    """Return (mode, bits) for given config name and layer index."""
    is_outlier = layer in OUTLIER_LAYERS
    is_layer2 = (layer == 2)

    if name == 'B_pca_uniform_2b':
        return 'uniform', 2
    elif name == 'C_pca_l2lloyd_2b':
        return 'l2_lloyd', 2
    elif name == 'D_pca_mahal_2b':
        return 'mahalanobis', 2
    elif name == 'E_pca_l2_outlier_3b':
        return ('l2_lloyd', 3) if is_outlier else ('l2_lloyd', 2)
    elif name == 'F_pca_l2_outlier_4b':
        return ('l2_lloyd', 4) if is_outlier else ('l2_lloyd', 2)
    elif name == 'G_pca_l2_layer2_only_4b':
        return ('l2_lloyd', 4) if is_layer2 else ('l2_lloyd', 2)
    elif name == 'H_pca_l2_outlier_8b':
        return ('l2_lloyd', 8) if is_outlier else ('l2_lloyd', 2)
    elif name == 'I_pca_mahal_outlier_4b':
        return ('mahalanobis', 4) if is_outlier else ('mahalanobis', 2)
    else:
        raise ValueError(name)


def install_hooks(model, per_layer_data, n_kv, head_dim):
    handles = []
    for li, head_data in per_layer_data.items():
        hook = PCAQuantHook(head_data, n_kv, head_dim)
        h = model.model.layers[li].self_attn.k_proj.register_forward_hook(hook)
        handles.append(h)
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


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


def main():
    print("=" * 60)
    print("Next-4: PCA-unified comprehensive PPL on Mistral-7B")
    print("=" * 60, flush=True)
    t_start = time.time()

    print("Loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device_map=DEVICE,
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t_start:.1f}s", flush=True)

    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_q
    print(f"  n_layers={n_layers}, n_kv={n_kv}, n_q={n_q}, head_dim={head_dim}", flush=True)

    calib_text, eval_text = get_texts(tok)
    eval_enc = tok(eval_text, return_tensors='pt', truncation=True, max_length=N_EVAL_TOKENS)
    eval_ids = eval_enc['input_ids'].to(DEVICE)
    print(f"  Eval tokens: {eval_ids.shape[1]}", flush=True)

    # Baseline FP16
    print("\n[A] FP16 baseline...", flush=True)
    ppl_fp16, _ = compute_ppl(model, eval_ids)
    print(f"  FP16 PPL: {ppl_fp16:.4f}", flush=True)

    # Collect calibration data ONCE
    print("\n[Calib] Collecting K/Q/attention from all layers...", flush=True)
    captured, T_calib = collect_calib_data(model, tok, calib_text, n_layers, n_kv, n_q, head_dim)

    # All configs to test
    configs = [
        'B_pca_uniform_2b',
        'C_pca_l2lloyd_2b',
        'D_pca_mahal_2b',
        'E_pca_l2_outlier_3b',
        'F_pca_l2_outlier_4b',
        'G_pca_l2_layer2_only_4b',
        'H_pca_l2_outlier_8b',
        'I_pca_mahal_outlier_4b',
    ]

    results = {}
    results['ppl_fp16'] = ppl_fp16

    for cfg in configs:
        print(f"\n[{cfg}] Fitting...", flush=True)
        t0 = time.time()
        try:
            per_layer_data = build_head_data(
                captured, T_calib, n_layers, n_kv, n_q, head_dim, cfg
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FIT FAILED: {e}")
            results[cfg] = {'error': f'fit: {e}'}
            continue
        t_fit = time.time() - t0
        print(f"  Fit done in {t_fit:.1f}s", flush=True)

        # Compute avg bits
        total_bits = 0
        n_heads_total = 0
        for li, hd in per_layer_data.items():
            for entry in hd:
                total_bits += entry['bits']
                n_heads_total += 1
        avg_bits = total_bits / max(n_heads_total, 1)

        # Install hooks and measure PPL
        handles = install_hooks(model, per_layer_data, n_kv, head_dim)
        t0 = time.time()
        try:
            ppl, loss = compute_ppl(model, eval_ids)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  PPL FAILED: {e}")
            remove_hooks(handles)
            results[cfg] = {'error': f'ppl: {e}'}
            continue
        t_ppl = time.time() - t0
        remove_hooks(handles)

        delta = (ppl - ppl_fp16) / ppl_fp16 * 100
        print(f"  avg_bits={avg_bits:.3f} | PPL={ppl:.4f} | Δvs FP16={delta:+.2f}% | PPL time={t_ppl:.1f}s", flush=True)

        results[cfg] = {
            'avg_bits': avg_bits,
            'ppl': ppl,
            'loss': loss,
            'delta_vs_fp16_pct': delta,
            'fit_time_sec': t_fit,
            'ppl_time_sec': t_ppl,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  FP16 baseline: {ppl_fp16:.4f}")
    print()
    print(f"{'Config':<30} | {'avg_bits':>10} | {'PPL':>12} | {'Δ vs FP16':>12}")
    print('-' * 75)
    for cfg in configs:
        r = results.get(cfg)
        if r and 'ppl' in r:
            print(f"{cfg:<30} | {r['avg_bits']:>10.3f} | {r['ppl']:>12.4f} | {r['delta_vs_fp16_pct']:>+11.2f}%")
        else:
            print(f"{cfg:<30} | ERROR: {r.get('error','?') if r else 'not run'}")

    # Key comparisons
    print("\n=== Key comparisons ===")
    if 'B_pca_uniform_2b' in results and 'ppl' in results['B_pca_uniform_2b']:
        ref = results['B_pca_uniform_2b']['ppl']
        print(f"  v3 Mistral Uniform 2b reference: 6.4614, ours: {ref:.4f}")
    if 'C_pca_l2lloyd_2b' in results and 'ppl' in results['C_pca_l2lloyd_2b']:
        v3_lloyd = 32.6844
        ours = results['C_pca_l2lloyd_2b']['ppl']
        print(f"  v3 Mistral L² Lloyd 2b catastrophe: {v3_lloyd:.4f}, ours: {ours:.4f}")
    if 'D_pca_mahal_2b' in results and 'C_pca_l2lloyd_2b' in results:
        if 'ppl' in results['D_pca_mahal_2b'] and 'ppl' in results['C_pca_l2lloyd_2b']:
            d = results['D_pca_mahal_2b']['ppl']
            c = results['C_pca_l2lloyd_2b']['ppl']
            print(f"  Mahalanobis vs L² Lloyd: {d:.4f} vs {c:.4f} ({'BETTER' if d < c else 'WORSE'} by {(d-c)/c*100:+.1f}%)")
    if 'E_pca_l2_outlier_3b' in results and 'B_pca_uniform_2b' in results:
        if 'ppl' in results['E_pca_l2_outlier_3b'] and 'ppl' in results['B_pca_uniform_2b']:
            e = results['E_pca_l2_outlier_3b']['ppl']
            b = results['B_pca_uniform_2b']['ppl']
            print(f"  Outlier 3b vs Uniform 2b: {e:.4f} vs {b:.4f} ({'BETTER' if e < b else 'WORSE'})")

    results['_meta'] = {
        'model': MODEL_NAME,
        'n_calib_tokens': N_CALIB_TOKENS,
        'n_eval_tokens': eval_ids.shape[1],
        'outlier_layers': OUTLIER_LAYERS,
        'v3_reference': {
            'mistral_fp16': 5.5717,
            'mistral_uniform_2b': 6.4614,
            'mistral_lloyd_2b': 32.6844,
            'mistral_wf_floor2_2b': 5.8222,
        },
        'runtime_sec': time.time() - t_start,
    }

    out_file = OUT_DIR / 'exp_next4_pca_unified_mistral.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")
    print(f"Total runtime: {results['_meta']['runtime_sec']:.1f}s ({results['_meta']['runtime_sec']/60:.1f}m)")


if __name__ == '__main__':
    main()
