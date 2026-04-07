#!/usr/bin/env python3
"""
V2v: Mode Detection Signature — localized sink vs distributed tail
====================================================================

v2u revealed that the three tested models require different optimal strategies:
  - Mistral-7B  : PCA + Lloyd + sink_k=1  (Mode A: localized BOS sink)
  - Nemo-12B    : PCA + Grid  (no sink)   (Mode B: distributed tail)
  - Qwen-7B     : PCA + Lloyd + sink_k=1  (Mode A, mild)

We want a calibration-only scalar signature that predicts which mode a model
lives in, so that the method can be auto-selected without a sweep.

Candidate signatures (computed per (layer, head), then aggregated):

  S1: fraction of top-PCA projection variance at position 0
      = (V[:,0] ⋅ (K[0] - mean))² / Σ_t (V[:,0] ⋅ (K[t] - mean))²
      High = sink at pos 0 (Mode A)

  S2: kurtosis of top-PCA projection values
      = E[p⁴] / E[p²]² - 3
      High = single extreme token (Mode A)
      Moderate = distributed heavy tail (Mode B)

  S3: ratio p99.9 / p99 of |top projection|
      Large = one extreme token stands out (Mode A)
      Small = many similar-extreme tokens (Mode B)

  S4: position index of max |top projection|
      pos 0 = Mode A; elsewhere = Mode B

  S5: tail index (Hill estimator) on |top projection|
      Low α = heavy tail (Mode B?); high α = light tail
      Note: interpretation is indirect

  S6: Entropy of normalized |top projection|² distribution
      Low entropy = concentrated (Mode A)
      High entropy = distributed (Mode B)

No forward/eval needed. Runtime: ~2 min total.
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
OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')

MODELS = [
    ('mistralai/Mistral-7B-v0.3', 'mistral-7b',   'ModeA'),  # expected
    ('mistralai/Mistral-Nemo-Base-2407', 'nemo-12b', 'ModeB'),
    ('Qwen/Qwen2.5-7B', 'qwen-7b',   'ModeA-mild'),
]


def hill_alpha(x, k_frac=0.05):
    """Hill estimator for tail index of |x| (larger = lighter tail)."""
    ax = np.sort(np.abs(x))[::-1]
    k = max(2, int(k_frac * len(ax)))
    log_ratios = np.log(ax[:k] / ax[k])
    if np.any(~np.isfinite(log_ratios)):
        return float('nan')
    return float(1.0 / np.mean(log_ratios))


def head_signatures(K_pca_top, T):
    """
    K_pca_top: 1D array of length T (top PCA projection of head)
    Returns dict of scalar signatures for this head.
    """
    p = K_pca_top.astype(np.float64)
    ap = np.abs(p)
    var = float(np.sum(p**2))
    if var < 1e-12:
        return None

    # S1: fraction at pos 0
    s1 = float(p[0]**2 / var)

    # S2: kurtosis (excess)
    m2 = float(np.mean(p**2))
    m4 = float(np.mean(p**4))
    s2 = m4 / (m2**2) - 3.0 if m2 > 1e-12 else 0.0

    # S3: p99.9 / p99 of |p|
    p99 = float(np.percentile(ap, 99))
    p999 = float(np.percentile(ap, 99.9))
    s3 = p999 / max(p99, 1e-12)

    # S4: is argmax at pos 0?
    s4_pos = int(np.argmax(ap))
    s4_is_pos0 = int(s4_pos == 0)

    # S5: Hill alpha on top 5%
    s5 = hill_alpha(p, 0.05)

    # S6: normalized entropy of |p|² (over T bins)
    prob = (p**2) / var
    prob = prob[prob > 1e-15]
    s6 = float(-np.sum(prob * np.log(prob))) / np.log(T)  # normalized by log(T)

    return {
        'S1_frac_at_pos0': s1,
        'S2_kurtosis': s2,
        'S3_p999_over_p99': s3,
        'S4_argmax_pos': s4_pos,
        'S4_argmax_is_pos0': s4_is_pos0,
        'S5_hill_alpha': s5,
        'S6_entropy_norm': s6,
    }


def analyze_model(model_id, sn, expected_mode):
    print(f"\n{'='*70}\n  {sn} [{expected_mode}]: {model_id}\n{'='*70}", flush=True)
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
    print(f"  loaded n_layers={n_layers} n_kv={n_kv} head_dim={head_dim} in {time.time()-t0:.1f}s", flush=True)

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in ds['text'] if len(t.strip()) > 100]
    calib = '\n\n'.join(texts[:300])
    ids = tok(calib, return_tensors='pt', truncation=True, max_length=N_CALIB)['input_ids'].to('cuda:0')
    T = ids.shape[1]

    # Capture k_proj outputs
    captured = {}
    def mk(li):
        def h(m, i, o): captured[li] = o.detach().cpu().float().numpy()
        return h
    handles = [model.model.layers[li].self_attn.k_proj.register_forward_hook(mk(li)) for li in range(n_layers)]
    with torch.no_grad():
        _ = model(ids, use_cache=False)
    for h in handles: h.remove()

    del model
    torch.cuda.empty_cache()

    # Compute per-head signatures
    all_head_sigs = []
    for li in range(n_layers):
        K_all = captured[li].reshape(-1, n_kv, head_dim).astype(np.float32)
        for hk in range(n_kv):
            K = K_all[:, hk, :]
            mean = K.mean(axis=0)
            Kc = K - mean
            cov = (Kc.T @ Kc) / max(K.shape[0]-1, 1)
            ev, vv = np.linalg.eigh(cov)
            order = np.argsort(ev)[::-1]
            ev = ev[order]; V = vv[:, order]
            kappa = float(ev[0] / max(ev[-1], 1e-12))
            lam_ratio = float(ev[0] / max(np.median(ev), 1e-12))
            # Project onto top eigenvector
            top_proj = Kc @ V[:, 0]  # (T,)
            sigs = head_signatures(top_proj, T)
            if sigs is None:
                continue
            sigs['layer'] = li
            sigs['kv_head'] = hk
            sigs['kappa'] = kappa
            sigs['lam_ratio'] = lam_ratio
            all_head_sigs.append(sigs)

    # Aggregate
    def agg(key):
        vals = [h[key] for h in all_head_sigs if h[key] is not None and np.isfinite(h[key])]
        if not vals: return {}
        return {
            'mean': float(np.mean(vals)),
            'median': float(np.median(vals)),
            'p95': float(np.percentile(vals, 95)),
            'max': float(np.max(vals)),
        }

    s1 = agg('S1_frac_at_pos0')
    s2 = agg('S2_kurtosis')
    s3 = agg('S3_p999_over_p99')
    s5 = agg('S5_hill_alpha')
    s6 = agg('S6_entropy_norm')

    frac_argmax_is_pos0 = float(np.mean([h['S4_argmax_is_pos0'] for h in all_head_sigs]))
    n_heads_pos0_argmax = int(sum(h['S4_argmax_is_pos0'] for h in all_head_sigs))

    # High-κ head subset (κ > 1e4, or top-10% if few)
    sorted_k = sorted(all_head_sigs, key=lambda h: -h['kappa'])
    n_high = max(1, sum(1 for h in all_head_sigs if h['kappa'] > 1e4))
    top_k_count = max(n_high, len(all_head_sigs) // 10)
    high_k_heads = sorted_k[:top_k_count]
    high_k_s1 = float(np.mean([h['S1_frac_at_pos0'] for h in high_k_heads]))
    high_k_s4 = float(np.mean([h['S4_argmax_is_pos0'] for h in high_k_heads]))
    high_k_kappa_med = float(np.median([h['kappa'] for h in high_k_heads]))

    print(f"\n  Aggregates over {len(all_head_sigs)} heads:")
    print(f"    S1 (frac at pos 0)    : mean={s1['mean']:.4f}, median={s1['median']:.4f}, p95={s1['p95']:.4f}, max={s1['max']:.4f}")
    print(f"    S2 (kurtosis)          : mean={s2['mean']:.2f}, median={s2['median']:.2f}, p95={s2['p95']:.2f}, max={s2['max']:.2f}")
    print(f"    S3 (p99.9/p99)         : mean={s3['mean']:.2f}, median={s3['median']:.2f}, p95={s3['p95']:.2f}, max={s3['max']:.2f}")
    print(f"    S4 (argmax at pos 0)   : {n_heads_pos0_argmax}/{len(all_head_sigs)} heads ({frac_argmax_is_pos0*100:.1f}%)")
    print(f"    S5 (Hill α, larger=lighter tail): mean={s5.get('mean',float('nan')):.2f}, median={s5.get('median',float('nan')):.2f}")
    print(f"    S6 (entropy, normalized): mean={s6['mean']:.4f}, median={s6['median']:.4f}")

    print(f"\n  High-κ subset (top {top_k_count} heads, median κ={high_k_kappa_med:.1e}):")
    print(f"    S1 at top heads: {high_k_s1:.4f}  (Mistral expects large; Nemo small)")
    print(f"    S4 argmax==pos0 at top heads: {high_k_s4*100:.1f}%  (Mistral expects high; Nemo low)")

    return {
        'model': model_id,
        'short_name': sn,
        'expected_mode': expected_mode,
        'n_heads': len(all_head_sigs),
        'aggregates': {
            'S1_frac_at_pos0': s1,
            'S2_kurtosis': s2,
            'S3_p999_over_p99': s3,
            'S4_frac_argmax_at_pos0': frac_argmax_is_pos0,
            'S4_n_heads_pos0_argmax': n_heads_pos0_argmax,
            'S5_hill_alpha': s5,
            'S6_entropy_norm': s6,
        },
        'high_kappa_subset': {
            'n_heads': top_k_count,
            'median_kappa': high_k_kappa_med,
            'S1_mean': high_k_s1,
            'S4_frac_argmax_at_pos0': high_k_s4,
        },
    }


def main():
    print("="*70)
    print("V2v: Mode Detection Signature")
    print("="*70, flush=True)
    t_start = time.time()

    results = {}
    for mid, sn, mode in MODELS:
        try:
            results[sn] = analyze_model(mid, sn, mode)
        except Exception as e:
            print(f"ERROR on {sn}: {e}")
            import traceback; traceback.print_exc()

    # Summary — the key discriminative signature
    print("\n" + "="*70)
    print("SIGNATURE SUMMARY — which signal cleanly separates Mode A vs Mode B?")
    print("="*70)
    print(f"  {'model':<20}|{'mode':<12}|{'S1_allh_med':>12}|{'S1_topκ':>10}|"
          f"{'S4_allh %':>12}|{'S4_topκ %':>12}|{'S2_med':>10}|{'S6_med':>10}")
    for sn, r in results.items():
        ag = r['aggregates']
        hk = r['high_kappa_subset']
        print(f"  {sn:<20}|{r['expected_mode']:<12}|"
              f"{ag['S1_frac_at_pos0']['median']:>12.4f}|"
              f"{hk['S1_mean']:>10.4f}|"
              f"{ag['S4_frac_argmax_at_pos0']*100:>11.1f}%|"
              f"{hk['S4_frac_argmax_at_pos0']*100:>11.1f}%|"
              f"{ag['S2_kurtosis']['median']:>10.2f}|"
              f"{ag['S6_entropy_norm']['median']:>10.4f}")

    out = OUT_DIR / 'exp_v2v_mode_detection.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Runtime: {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
