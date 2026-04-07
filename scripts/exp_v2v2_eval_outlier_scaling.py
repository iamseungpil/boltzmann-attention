#!/usr/bin/env python3
"""
V2v2: Eval-Time Outlier Scaling — Why does Nemo catastrophe jump at L=32768?
==============================================================================

v2v showed calibration-based signatures only discriminate Mistral (pos 0
concentration) from Nemo/Qwen, but cannot separate Nemo (Grid wins) from Qwen
(Lloyd+sink wins). And Nemo's Lloyd catastrophe gap jumps non-monotonically
(+2.5 → +2.5 → +11.2 across L=2048/8192/32768) which calibration-only signals
miss entirely.

The hypothesis: the EVAL-time K distribution changes with length in a model-
specific way. Nemo might have "secondary sinks" that only appear at long
context. Mistral's sink stays at pos 0 regardless of length.

Protocol: for each model, compute forward pass at L ∈ {2048, 8192, 32768}
and measure on the top-κ heads (using calibration-fit PCA basis):
  M1: fraction of top-PCA variance in top-1 position (any position)
  M2: fraction of top-PCA variance in top-10 positions
  M3: number of positions with |projection| > 5 * std (5σ outliers)
  M4: density of 5σ outliers per 1000 tokens
  M5: max |projection| / 99.9th percentile ratio

If M1 drops as L grows (Nemo) → outliers are new tokens at long context
If M1 stays high (Mistral) → same sink dominates regardless of length
If M4 grows linearly with L (Nemo) → distributed tail model
If M4 stays constant (Mistral) → localized sink model

This should give a clean discriminator.
"""
import json, os, time
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

DTYPE = torch.bfloat16
N_CALIB = 2048
EVAL_LENGTHS = [2048, 8192, 32768]
OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')

MODELS = [
    ('mistralai/Mistral-7B-v0.3', 'mistral-7b'),
    ('mistralai/Mistral-Nemo-Base-2407', 'nemo-12b'),
    ('Qwen/Qwen2.5-7B', 'qwen-7b'),
]


def calibrate(model, ids, n_layers, n_kv, head_dim):
    pl = {}
    def mk(li):
        def h(m, i, o): pl[li] = o.detach().cpu().float().numpy()
        return h
    handles = [model.model.layers[li].self_attn.k_proj.register_forward_hook(mk(li)) for li in range(n_layers)]
    with torch.no_grad():
        _ = model(ids, use_cache=False)
    for h in handles: h.remove()
    basis = {}
    for li in range(n_layers):
        K_all = pl[li].reshape(-1, n_kv, head_dim).astype(np.float32)
        ph = []
        for hk in range(n_kv):
            K = K_all[:, hk, :]; mean = K.mean(axis=0); Kc = K - mean
            cov = (Kc.T @ Kc) / max(K.shape[0]-1, 1)
            ev, vv = np.linalg.eigh(cov)
            order = np.argsort(ev)[::-1]
            V = vv[:, order].astype(np.float32)
            ph.append({'V': V, 'mean': mean.astype(np.float32), 'eigvals': ev[order]})
        basis[li] = ph
    return basis


def capture_eval(model, ids, n_layers):
    pl = {}
    def mk(li):
        def h(m, i, o): pl[li] = o.detach().cpu().float().numpy()
        return h
    handles = [model.model.layers[li].self_attn.k_proj.register_forward_hook(mk(li)) for li in range(n_layers)]
    with torch.no_grad():
        _ = model(ids, use_cache=False)
    for h in handles: h.remove()
    return pl


def measure_head_outliers(proj):
    """Measure per-head outlier distribution."""
    var = float(np.sum(proj**2))
    if var < 1e-12:
        return None
    ap = np.abs(proj)
    T = len(proj)
    std = float(np.std(proj))

    # M1: top-1 fraction of variance
    sorted_idx = np.argsort(-ap)
    top1 = float(proj[sorted_idx[0]]**2 / var)

    # M2: top-10 fraction
    top10 = float(np.sum(proj[sorted_idx[:min(10, T)]]**2) / var)

    # M3: count of 5σ outliers
    n_5sig = int(np.sum(ap > 5 * std))

    # M4: 5σ outliers per 1000 tokens
    density = n_5sig * 1000.0 / T

    # M5: max / p99.9
    p999 = float(np.percentile(ap, 99.9))
    m5 = float(ap.max() / max(p999, 1e-12))

    # M6: argmax position
    argmax = int(sorted_idx[0])

    return {
        'top1_var_frac': top1,
        'top10_var_frac': top10,
        'n_5sigma': n_5sig,
        'density_5sigma': density,
        'max_over_p999': m5,
        'argmax_pos': argmax,
        'argmax_is_pos0': int(argmax == 0),
    }


def analyze_model(model_id, sn):
    print(f"\n{'='*70}\n  {sn}: {model_id}\n{'='*70}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='sdpa', low_cpu_mem_usage=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, 'head_dim', None) or (model.config.hidden_size // model.config.num_attention_heads)
    print(f"  loaded {sn} in {time.time()-t0:.1f}s", flush=True)

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in ds['text'] if len(t.strip()) > 100]
    calib = '\n\n'.join(texts[:300])
    eval_t = '\n\n'.join(texts[300:3000])

    calib_ids = tok(calib, return_tensors='pt', truncation=True, max_length=N_CALIB)['input_ids'].to('cuda:0')
    basis = calibrate(model, calib_ids, n_layers, n_kv, head_dim)

    # Select top-32 high-κ heads from calibration
    all_k_heads = []
    for li in range(n_layers):
        for hk in range(n_kv):
            ev = basis[li][hk]['eigvals']
            kappa = float(ev[0] / max(ev[-1], 1e-12))
            all_k_heads.append({'layer': li, 'kv_head': hk, 'kappa': kappa})
    all_k_heads.sort(key=lambda x: -x['kappa'])
    top32 = all_k_heads[:32]
    top32_set = {(h['layer'], h['kv_head']) for h in top32}
    print(f"  Top-32 κ range: {top32[0]['kappa']:.1e} to {top32[-1]['kappa']:.1e}", flush=True)

    result = {'model': model_id, 'short_name': sn, 'top32_heads': top32, 'by_length': {}}

    for L in EVAL_LENGTHS:
        eval_ids = tok(eval_t, return_tensors='pt', truncation=True, max_length=L)['input_ids'].to('cuda:0')
        T = eval_ids.shape[1]
        K_per_layer = capture_eval(model, eval_ids, n_layers)

        # For each top-32 head, compute eval-time top-PCA projection and measure
        measurements = []
        for rec in top32:
            li, hk = rec['layer'], rec['kv_head']
            K_all = K_per_layer[li].reshape(-1, n_kv, head_dim).astype(np.float32)
            K = K_all[:, hk, :]
            mean = basis[li][hk]['mean']  # USE CALIBRATION mean
            V = basis[li][hk]['V']        # USE CALIBRATION basis
            proj = (K - mean) @ V[:, 0]   # top eigenvector projection at eval time
            m = measure_head_outliers(proj)
            if m is not None:
                m['layer'] = li
                m['kv_head'] = hk
                m['kappa'] = rec['kappa']
                measurements.append(m)

        # Aggregate
        m1_mean = float(np.mean([m['top1_var_frac'] for m in measurements]))
        m1_med = float(np.median([m['top1_var_frac'] for m in measurements]))
        m2_mean = float(np.mean([m['top10_var_frac'] for m in measurements]))
        m3_mean = float(np.mean([m['n_5sigma'] for m in measurements]))
        m3_med = float(np.median([m['n_5sigma'] for m in measurements]))
        m4_mean = float(np.mean([m['density_5sigma'] for m in measurements]))
        m5_mean = float(np.mean([m['max_over_p999'] for m in measurements]))
        frac_pos0 = float(np.mean([m['argmax_is_pos0'] for m in measurements]))

        print(f"\n  L={L} (T={T}, {len(measurements)} heads):")
        print(f"    M1 top-1 var frac   : mean={m1_mean:.4f}, median={m1_med:.4f}")
        print(f"    M2 top-10 var frac  : mean={m2_mean:.4f}")
        print(f"    M3 # of 5σ outliers : mean={m3_mean:.1f}, median={m3_med:.1f}")
        print(f"    M4 density/1000 tok : mean={m4_mean:.2f}")
        print(f"    M5 max/p99.9        : mean={m5_mean:.2f}")
        print(f"    argmax==pos0 frac   : {frac_pos0*100:.1f}%", flush=True)

        result['by_length'][L] = {
            'n_tokens': T,
            'm1_top1_var_frac_mean': m1_mean,
            'm1_top1_var_frac_median': m1_med,
            'm2_top10_var_frac_mean': m2_mean,
            'm3_n_5sigma_mean': m3_mean,
            'm3_n_5sigma_median': m3_med,
            'm4_density_per_1000_mean': m4_mean,
            'm5_max_over_p999_mean': m5_mean,
            'frac_argmax_at_pos0': frac_pos0,
            'per_head_measurements': measurements,
        }

    del model
    torch.cuda.empty_cache()
    return result


def main():
    print("="*70)
    print("V2v2: Eval-Time Outlier Scaling")
    print("="*70, flush=True)
    t_start = time.time()

    results = {}
    for mid, sn in MODELS:
        try:
            results[sn] = analyze_model(mid, sn)
        except Exception as e:
            print(f"ERROR on {sn}: {e}")
            import traceback; traceback.print_exc()

    # Comparison table
    print("\n" + "="*70)
    print("SCALING SUMMARY — top-κ head outlier statistics by length")
    print("="*70)
    print(f"  {'model':<14}|{'length':<8}|{'M1 top1':>10}|{'M3 #5σ':>10}|{'M4 /1k':>10}|{'argmax==0':>12}")
    for sn, r in results.items():
        for L in EVAL_LENGTHS:
            d = r['by_length'][L]
            print(f"  {sn:<14}|{L:<8}|{d['m1_top1_var_frac_mean']:>10.4f}|"
                  f"{d['m3_n_5sigma_mean']:>10.1f}|{d['m4_density_per_1000_mean']:>10.2f}|"
                  f"{d['frac_argmax_at_pos0']*100:>11.1f}%")

    out = OUT_DIR / 'exp_v2v2_eval_outlier_scaling.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Runtime: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f}m)")


if __name__ == '__main__':
    main()
