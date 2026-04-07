#!/usr/bin/env python3
"""
E1 + E2 combined: κ(M_KL) + Tail Index α measurement
======================================================

For each of 3 target models (Qwen, Llama, Mistral):
  1. Load model, run short calibration pass (2K tokens)
  2. Capture per-layer per-head keys (K) and attention probs (P) via hook
  3. Compute:
     - E1: F_avg = (1/T) Σ_j (Σ_t p_tj(1-p_tj)) q_t q_t^T, κ = λ_max/λ_min
     - E2: PCA basis of K, Hill estimator on each PCA dim → tail index α

Both experiments run in single forward pass — maximum efficiency.

From: AXIS2_THEORETICAL_VERIFICATION_EXPERIMENT_PLAN.md §2, §3
Author: Claude Opus 4.6 (2026-04-07)
"""
import json
import time
import gc
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

MODELS = [
    ('Qwen/Qwen2.5-7B',              'qwen2.5-7b'),
    ('meta-llama/Llama-3.1-8B',      'llama-3.1-8b'),
    ('mistralai/Mistral-7B-v0.3',    'mistral-7b-v0.3'),
]

N_CALIB_TOKENS = 2048         # calibration sequence length
N_LAYERS_SAMPLED = 8          # equally-spaced layers to analyze
DEVICE = 'cuda:0'
DTYPE = torch.bfloat16

OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Calibration data
# ----------------------------------------------------------------------

def get_calibration_text(n_tokens_approx):
    """Load WikiText-2 train subset, concatenate until enough tokens."""
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        texts = [t for t in ds['text'] if len(t.strip()) > 100]
        text = '\n\n'.join(texts[:500])
    except Exception as e:
        print(f"Dataset load failed ({e}), using fallback text")
        text = ' '.join([
            "The Lie group framework unifies rotation-based KV-cache quantization methods.",
            "Pre-RoPE PCA is MSE-optimal within Class C block-diagonal orthogonal rotations.",
            "Water-Filling with floor constraint prevents 1-bit channel catastrophe.",
            "Heavy-tail distributions require L1 Lloyd-Max median-based quantization.",
        ] * 1000)
    return text


# ----------------------------------------------------------------------
# Hill estimator for tail index
# ----------------------------------------------------------------------

def hill_estimator(x, tail_frac=0.1):
    """
    Hill estimator of tail index alpha.
    Uses top tail_frac of |x|.
    alpha = 1 / mean(log(X_top / X_threshold))
    """
    x_abs = np.abs(x)
    x_sorted = np.sort(x_abs)
    n = len(x_sorted)
    k = max(int(n * tail_frac), 10)
    if k >= n - 1:
        return float('nan')
    threshold = x_sorted[-k - 1]
    if threshold <= 1e-12:
        return float('nan')
    top = x_sorted[-k:]
    logs = np.log(top / threshold)
    mean_log = np.mean(logs)
    if mean_log <= 1e-12:
        return float('inf')
    return 1.0 / mean_log


# ----------------------------------------------------------------------
# Per-model analysis
# ----------------------------------------------------------------------

def analyze_model(model_name, short_name):
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}", flush=True)
    t_start = time.time()

    # Load tokenizer + model
    print("Loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    print("Loading model (this may take a moment)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=DTYPE,
        device_map=DEVICE,
        attn_implementation='eager',  # required for output_attentions
        low_cpu_mem_usage=True,
    )
    model.eval()
    t_load = time.time() - t_start
    print(f"  Loaded in {t_load:.1f}s. dtype={DTYPE}, device={DEVICE}", flush=True)

    # Prepare calibration
    text = get_calibration_text(N_CALIB_TOKENS * 2)
    enc = tok(text, return_tensors='pt', truncation=True, max_length=N_CALIB_TOKENS)
    input_ids = enc['input_ids'].to(DEVICE)
    T = input_ids.shape[1]
    print(f"  Calibration sequence: T={T} tokens", flush=True)

    # Determine layer indices to sample
    n_total_layers = model.config.num_hidden_layers
    layer_idx_list = np.linspace(2, n_total_layers - 2, N_LAYERS_SAMPLED, dtype=int).tolist()
    print(f"  Analyzing {N_LAYERS_SAMPLED} layers: {layer_idx_list}", flush=True)

    # Register hooks to capture k_states, q_states
    captured = {}  # layer_idx -> {'k': ..., 'q': ..., 'attn': ...}

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            # output may be (attn_output, attn_weights, ...) depending on model
            # We want attn_weights, which is at index 1 when output_attentions=True
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_weights = output[1].detach()  # (B, H, T, T)
                captured.setdefault(layer_idx, {})['attn'] = attn_weights.cpu()
        return hook

    def make_kq_hook(layer_idx):
        """Capture k_proj and q_proj outputs."""
        def k_hook(module, inputs, output):
            captured.setdefault(layer_idx, {})['k_raw'] = output.detach().cpu()
        def q_hook(module, inputs, output):
            captured.setdefault(layer_idx, {})['q_raw'] = output.detach().cpu()
        return k_hook, q_hook

    handles = []
    # Find layer objects
    try:
        layers = model.model.layers
    except AttributeError:
        layers = model.base_model.model.layers if hasattr(model, 'base_model') else None

    for li in layer_idx_list:
        attn_module = layers[li].self_attn
        # Hook attention output to capture attn_weights
        h1 = attn_module.register_forward_hook(make_hook(li))
        handles.append(h1)
        # Hook k_proj and q_proj for raw (post-projection, pre-RoPE) keys/queries
        k_hook_fn, q_hook_fn = make_kq_hook(li)
        if hasattr(attn_module, 'k_proj'):
            handles.append(attn_module.k_proj.register_forward_hook(k_hook_fn))
            handles.append(attn_module.q_proj.register_forward_hook(q_hook_fn))

    # Forward pass
    print("  Running forward pass with output_attentions=True...", flush=True)
    t_fwd = time.time()
    with torch.no_grad():
        out = model(input_ids, output_attentions=True, use_cache=False)
    print(f"  Forward done in {time.time()-t_fwd:.1f}s", flush=True)

    # Remove hooks
    for h in handles:
        h.remove()

    # Extract per-layer metrics
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    n_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    n_q_heads = model.config.num_attention_heads

    print(f"  head_dim={head_dim}, n_q_heads={n_q_heads}, n_kv_heads={n_kv_heads}", flush=True)

    layer_results = {}
    for li in layer_idx_list:
        data = captured.get(li, {})
        if 'attn' not in data or 'k_raw' not in data or 'q_raw' not in data:
            print(f"  layer {li}: capture missing, skip")
            continue

        attn = data['attn']    # (1, H, T, T) — uses num_q_heads
        k_raw = data['k_raw']  # (1, T, n_kv_heads * head_dim)
        q_raw = data['q_raw']  # (1, T, n_q_heads * head_dim)

        # Reshape
        k_raw = k_raw.view(1, T, n_kv_heads, head_dim)  # (1, T, Hkv, d)
        q_raw = q_raw.view(1, T, n_q_heads, head_dim)   # (1, T, Hq, d)

        head_metrics = []
        for h_kv in range(n_kv_heads):
            # For GQA: each KV head serves multiple Q heads
            n_q_per_kv = n_q_heads // n_kv_heads
            q_heads_for_this_kv = list(range(h_kv * n_q_per_kv, (h_kv + 1) * n_q_per_kv))

            # Keys for this kv head
            K = k_raw[0, :, h_kv, :].float().numpy()   # (T, d)

            # Queries: average Fisher across the associated q heads
            kappa_list = []
            for hq in q_heads_for_this_kv:
                Q_hq = q_raw[0, :, hq, :].float().numpy()     # (T, d)
                A_hq = attn[0, hq, :, :].float().numpy()      # (T, T) — p[t, j]

                # Fisher metric F_avg = (1/T) Σ_t s_t * q_t q_t^T
                # where s_t = Σ_j p[t,j](1-p[t,j])
                p_var = A_hq * (1.0 - A_hq)
                s_t = p_var.sum(axis=1)    # (T,)
                # weighted outer product sum
                Q_weighted = Q_hq * s_t[:, None]   # (T, d)
                F_avg = (Q_weighted.T @ Q_hq) / max(T, 1)  # (d, d)

                # Regularize for numerical stability
                F_avg = F_avg + 1e-8 * np.eye(F_avg.shape[0])
                eigvals = np.linalg.eigvalsh(F_avg)
                eigvals = np.clip(eigvals, 1e-12, None)
                kappa = float(eigvals.max() / eigvals.min())
                kappa_list.append(kappa)

            kappa_mean = float(np.mean(kappa_list))

            # E2: PCA + Hill estimator on K (this kv head)
            K_centered = K - K.mean(axis=0, keepdims=True)
            cov = (K_centered.T @ K_centered) / max(T - 1, 1)
            eigvals_K, eigvecs_K = np.linalg.eigh(cov)
            # Sort descending
            order = np.argsort(eigvals_K)[::-1]
            eigvals_K = eigvals_K[order]
            eigvecs_K = eigvecs_K[:, order]
            K_pca = K_centered @ eigvecs_K  # (T, d)

            # Hill estimator per dimension
            alphas_per_dim = []
            for j in range(head_dim):
                col = K_pca[:, j]
                alpha = hill_estimator(col, tail_frac=0.1)
                if np.isfinite(alpha):
                    alphas_per_dim.append(float(alpha))

            alpha_median = float(np.median(alphas_per_dim)) if alphas_per_dim else float('nan')
            alpha_min = float(np.min(alphas_per_dim)) if alphas_per_dim else float('nan')

            # PCA spectrum stats
            eigvals_K_pos = np.clip(eigvals_K, 1e-12, None)
            R_aniso = float(eigvals_K_pos[0] / eigvals_K_pos[-1])
            energy_top10 = float(eigvals_K_pos[:head_dim // 10].sum() / eigvals_K_pos.sum())

            head_metrics.append({
                'kv_head': h_kv,
                'kappa_F_avg_mean': kappa_mean,
                'kappa_F_avg_per_qhead': kappa_list,
                'alpha_median': alpha_median,
                'alpha_min': alpha_min,
                'R_aniso': R_aniso,
                'energy_top_10pct': energy_top10,
                'n_alpha_finite': len(alphas_per_dim),
            })

        layer_results[li] = head_metrics

    # Aggregate model-level stats
    all_kappas = []
    all_alphas_median = []
    all_R_aniso = []
    for li, heads in layer_results.items():
        for h in heads:
            all_kappas.append(h['kappa_F_avg_mean'])
            if np.isfinite(h['alpha_median']):
                all_alphas_median.append(h['alpha_median'])
            all_R_aniso.append(h['R_aniso'])

    summary = {
        'model': model_name,
        'short_name': short_name,
        'n_layers_sampled': len(layer_results),
        'layer_indices': layer_idx_list,
        'head_dim': head_dim,
        'n_q_heads': n_q_heads,
        'n_kv_heads': n_kv_heads,
        'n_tokens': T,
        # E1: kappa(F_avg)
        'kappa_F_avg': {
            'median': float(np.median(all_kappas)) if all_kappas else None,
            'mean': float(np.mean(all_kappas)) if all_kappas else None,
            'p95': float(np.percentile(all_kappas, 95)) if all_kappas else None,
            'max': float(np.max(all_kappas)) if all_kappas else None,
            'min': float(np.min(all_kappas)) if all_kappas else None,
            'n_samples': len(all_kappas),
        },
        # E2: tail index alpha (Hill estimator)
        'tail_index_alpha': {
            'median': float(np.median(all_alphas_median)) if all_alphas_median else None,
            'mean': float(np.mean(all_alphas_median)) if all_alphas_median else None,
            'p05': float(np.percentile(all_alphas_median, 5)) if all_alphas_median else None,
            'p95': float(np.percentile(all_alphas_median, 95)) if all_alphas_median else None,
            'min': float(np.min(all_alphas_median)) if all_alphas_median else None,
            'n_samples': len(all_alphas_median),
        },
        # R_aniso sanity check (v3 reports Qwen=4.27, Llama=7.97, Mistral=131.62)
        'R_aniso': {
            'median': float(np.median(all_R_aniso)) if all_R_aniso else None,
            'mean': float(np.mean(all_R_aniso)) if all_R_aniso else None,
        },
        'per_layer': {int(k): v for k, v in layer_results.items()},
        'runtime_sec': time.time() - t_start,
    }

    # Cleanup
    del model, out, captured
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n  E1 κ(F_avg): median={summary['kappa_F_avg']['median']:.2f}, "
          f"p95={summary['kappa_F_avg']['p95']:.2f}, max={summary['kappa_F_avg']['max']:.2f}", flush=True)
    print(f"  E2 α (Hill): median={summary['tail_index_alpha']['median']:.3f}, "
          f"min={summary['tail_index_alpha']['min']:.3f}", flush=True)
    print(f"  R_aniso: median={summary['R_aniso']['median']:.2f} "
          f"(v3 ref: Qwen=4.27, Llama=7.97, Mistral=131.62)", flush=True)
    print(f"  Model total runtime: {summary['runtime_sec']:.1f}s", flush=True)

    return summary


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    print(f"E1+E2: κ(M_KL) + Tail Index α measurement")
    print(f"Device: {DEVICE}, dtype: {DTYPE}")
    print(f"Calibration tokens: {N_CALIB_TOKENS}")
    print(f"Layers sampled per model: {N_LAYERS_SAMPLED}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    print()

    t0 = time.time()
    all_results = {}

    for model_name, short_name in MODELS:
        try:
            summary = analyze_model(model_name, short_name)
            all_results[short_name] = summary
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"!!! FAILED on {model_name}: {e}")
            all_results[short_name] = {'error': str(e)}

    total_time = time.time() - t0
    all_results['_meta'] = {
        'total_runtime_sec': total_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_tokens': N_CALIB_TOKENS,
        'n_layers_sampled': N_LAYERS_SAMPLED,
        'dtype': str(DTYPE),
    }

    # Save
    out_json = OUT_DIR / 'e1e2_kappa_tail_index_results.json'
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Final summary table
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — E1 (κ) + E2 (α) across 3 models")
    print("=" * 70)
    print(f"{'Model':<20} | {'κ median':>12} | {'κ p95':>12} | {'α median':>10} | {'R_aniso':>10}")
    print('-' * 75)
    for short, res in all_results.items():
        if short.startswith('_'):
            continue
        if 'error' in res:
            print(f"{short:<20} | ERROR: {res['error']}")
            continue
        k_med = res['kappa_F_avg']['median']
        k_p95 = res['kappa_F_avg']['p95']
        a_med = res['tail_index_alpha']['median']
        R = res['R_aniso']['median']
        print(f"{short:<20} | {k_med:>12.2f} | {k_p95:>12.2f} | {a_med:>10.3f} | {R:>10.2f}")

    # Theoretical prediction check
    print()
    print("Proposition A prediction: Mistral κ > Llama κ > Qwen κ (matches Lloyd failure order)")
    print("Proposition B prediction: Mistral α < 4 (heavy-tail) → L¹ Lloyd will win on Mistral")
    print()
    print(f"Results saved: {out_json}")
    print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f}m)")


if __name__ == '__main__':
    main()
