#!/usr/bin/env python3
"""
Experiment 1: Per-head κ outlier analysis
===========================================

Re-analyze e1e2_kappa_tail_index_results.json to identify per-head outliers.
Hypothesis: Lloyd-Max failure is driven by a SMALL NUMBER of outlier heads,
not the global κ median.

Output: identify outlier heads per model, correlate with v3 Lloyd failure order.
"""
import json
import numpy as np
from pathlib import Path

IN_JSON = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification/e1e2_kappa_tail_index_results.json')
OUT_JSON = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification/exp1_outlier_analysis_results.json')

# v3 reported Lloyd-Max 2-bit PPL failure ratios
V3_LLOYD_FAILURE = {
    'qwen2.5-7b':      {'uniform_2bit_ppl': 7.98,  'lloyd_2bit_ppl': 8.34,  'ratio': 1.05},
    'llama-3.1-8b':    {'uniform_2bit_ppl': 10.14, 'lloyd_2bit_ppl': 65.46, 'ratio': 6.46},
    'mistral-7b-v0.3': {'uniform_2bit_ppl': 6.46,  'lloyd_2bit_ppl': 32.68, 'ratio': 5.06},
}


def analyze_model(name, data):
    if 'error' in data or 'per_layer' not in data:
        return None

    # Flatten all per-head κ values across all layers
    per_head_kappas = []
    per_head_alphas = []
    per_head_R_aniso = []
    head_locations = []  # (layer, kv_head) tuples

    for layer_idx_str, heads in data['per_layer'].items():
        layer_idx = int(layer_idx_str)
        for h in heads:
            kv_head = h['kv_head']
            kappa = h['kappa_F_avg_mean']
            alpha = h['alpha_median']
            R_aniso = h['R_aniso']

            per_head_kappas.append(kappa)
            per_head_alphas.append(alpha if np.isfinite(alpha) else None)
            per_head_R_aniso.append(R_aniso)
            head_locations.append((layer_idx, kv_head))

    per_head_kappas = np.array(per_head_kappas)

    # Distribution statistics
    stats = {
        'n_heads_measured': len(per_head_kappas),
        'kappa_stats': {
            'min': float(per_head_kappas.min()),
            'median': float(np.median(per_head_kappas)),
            'mean': float(per_head_kappas.mean()),
            'std': float(per_head_kappas.std()),
            'p75': float(np.percentile(per_head_kappas, 75)),
            'p90': float(np.percentile(per_head_kappas, 90)),
            'p95': float(np.percentile(per_head_kappas, 95)),
            'p99': float(np.percentile(per_head_kappas, 99)),
            'max': float(per_head_kappas.max()),
        },
        'spread_metrics': {
            'p95_over_median': float(np.percentile(per_head_kappas, 95) / np.median(per_head_kappas)),
            'max_over_median': float(per_head_kappas.max() / np.median(per_head_kappas)),
            'std_over_mean': float(per_head_kappas.std() / per_head_kappas.mean()),
        },
    }

    # Outlier identification — top 10% by κ
    n = len(per_head_kappas)
    n_top = max(1, n // 10)
    sorted_indices = np.argsort(per_head_kappas)[::-1]  # descending
    top_outliers = sorted_indices[:n_top]

    outlier_heads = []
    for idx in top_outliers:
        layer, kv_head = head_locations[idx]
        outlier_heads.append({
            'layer': int(layer),
            'kv_head': int(kv_head),
            'kappa': float(per_head_kappas[idx]),
            'alpha': per_head_alphas[idx] if per_head_alphas[idx] is not None else None,
            'R_aniso': float(per_head_R_aniso[idx]),
        })
    stats['top_10pct_outlier_heads'] = outlier_heads

    # Fraction of heads with κ > threshold (extreme outliers)
    thresholds = [1e4, 1e5, 1e6, 1e7]
    stats['fraction_above_threshold'] = {
        f'>{int(t):g}': float((per_head_kappas > t).sum() / n)
        for t in thresholds
    }

    # Extreme outlier count
    median_k = np.median(per_head_kappas)
    stats['n_extreme_outliers'] = {
        '>5x_median':  int((per_head_kappas > 5 * median_k).sum()),
        '>10x_median': int((per_head_kappas > 10 * median_k).sum()),
        '>100x_median': int((per_head_kappas > 100 * median_k).sum()),
    }

    return stats


def correlate_with_v3(all_stats):
    """Test: does any per-head statistic correlate with v3 Lloyd failure ratio?"""
    print("\n" + "=" * 70)
    print("Correlation analysis: per-head stats vs v3 Lloyd failure")
    print("=" * 70)

    results = {}
    for name, v3 in V3_LLOYD_FAILURE.items():
        if name not in all_stats:
            continue
        s = all_stats[name]
        if s is None:
            continue
        row = {
            'v3_ratio': v3['ratio'],
            'kappa_median': s['kappa_stats']['median'],
            'kappa_p95': s['kappa_stats']['p95'],
            'kappa_max': s['kappa_stats']['max'],
            'p95_over_median': s['spread_metrics']['p95_over_median'],
            'n_extreme_outliers_10x': s['n_extreme_outliers']['>10x_median'],
            'fraction_above_1e5': s['fraction_above_threshold']['>100000'],
        }
        results[name] = row

    # Print table
    print()
    print(f"{'Model':<20} | {'v3 ratio':>10} | {'κ median':>12} | {'κ max':>14} | {'p95/med':>10} | {'outliers>10x':>12}")
    print('-' * 95)
    for name, r in results.items():
        print(f"{name:<20} | {r['v3_ratio']:>10.2f} | {r['kappa_median']:>12.0f} | "
              f"{r['kappa_max']:>14.0f} | {r['p95_over_median']:>10.2f} | {r['n_extreme_outliers_10x']:>12d}")
    print()

    # Compute Spearman correlation between each stat and v3_ratio
    # Only 2-3 data points, so Spearman is indicative not significant
    v3_ratios = [r['v3_ratio'] for r in results.values()]
    if len(v3_ratios) >= 2:
        for key in ['kappa_median', 'kappa_p95', 'kappa_max', 'p95_over_median',
                    'n_extreme_outliers_10x', 'fraction_above_1e5']:
            vals = [r[key] for r in results.values()]
            # Spearman = Pearson on ranks
            ranks_v3 = np.argsort(np.argsort(v3_ratios))
            ranks_val = np.argsort(np.argsort(vals))
            if len(set(ranks_val)) > 1:
                corr = np.corrcoef(ranks_v3, ranks_val)[0, 1]
            else:
                corr = 0.0
            print(f"  Spearman ρ ({key:>25} vs v3_ratio): {corr:+.3f}")

    return results


def main():
    with open(IN_JSON) as f:
        data = json.load(f)

    all_stats = {}
    print("=== Per-head κ outlier analysis ===\n")

    for name, model_data in data.items():
        if name.startswith('_'):
            continue
        print(f"--- {name} ---")
        stats = analyze_model(name, model_data)
        all_stats[name] = stats
        if stats is None:
            print(f"  (no data)\n")
            continue

        print(f"  n_heads_measured: {stats['n_heads_measured']}")
        ks = stats['kappa_stats']
        print(f"  κ min={ks['min']:.0f}, median={ks['median']:.0f}, p95={ks['p95']:.0f}, max={ks['max']:.0f}")
        print(f"  spread: p95/median={stats['spread_metrics']['p95_over_median']:.1f}x, "
              f"max/median={stats['spread_metrics']['max_over_median']:.1f}x")
        print(f"  extreme outliers: >5×med={stats['n_extreme_outliers']['>5x_median']}, "
              f">10×med={stats['n_extreme_outliers']['>10x_median']}, "
              f">100×med={stats['n_extreme_outliers']['>100x_median']}")
        print(f"  top-3 outlier heads:")
        for h in stats['top_10pct_outlier_heads'][:3]:
            print(f"    layer {h['layer']}, kv_head {h['kv_head']}: κ={h['kappa']:.0f}, R_aniso={h['R_aniso']:.0f}")
        print()

    # Cross-model correlation
    corr_results = correlate_with_v3(all_stats)

    # Save
    out_data = {
        'per_model_stats': all_stats,
        'v3_correlation': corr_results,
        'v3_reference': V3_LLOYD_FAILURE,
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(out_data, f, indent=2)

    print(f"\n=== Saved: {OUT_JSON} ===")


if __name__ == '__main__':
    main()
