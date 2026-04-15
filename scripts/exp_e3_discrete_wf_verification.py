#!/usr/bin/env python3
"""
E3: Discrete-WF Theorem verification
=====================================

Measure D_uniform(b) for b-bit scalar quantizer on Gaussian source.
Compare against Shannon rate-distortion D_Shannon(b) = sigma^2 * 2^(-2b).

Goal: Show that D_uniform(b) / D_Shannon(b) has a "knee" at b=2,
      justifying WF floor=2 as theoretical consequence, not ad-hoc.

From: AXIS2_THEORETICAL_VERIFICATION_EXPERIMENT_PLAN.md §4
Author: Claude Opus 4.6 (2026-04-07)
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar


# ----------------------------------------------------------------------
# Quantizer implementations
# ----------------------------------------------------------------------

def uniform_quantize_fixed_clip(x, bits, clip_std=3.0, sigma=1.0):
    """Fixed-clip uniform quantizer: [-clip, clip] divided into 2^bits bins."""
    n_levels = 2 ** bits
    clip = clip_std * sigma
    x_clipped = np.clip(x, -clip, clip)
    # Bin width
    delta = 2 * clip / n_levels
    # Bin indices in [0, n_levels)
    idx = np.floor((x_clipped + clip) / delta).astype(int)
    idx = np.clip(idx, 0, n_levels - 1)
    # Reconstruction: bin center
    recon = -clip + (idx + 0.5) * delta
    return recon


def uniform_quantize_optimal(x, bits, sigma=1.0):
    """
    Optimal uniform quantizer: minimize MSE by choosing clip range.
    For Gaussian, optimal clip is determined by Lloyd's condition.
    We do a 1D line search over clip_std.
    """
    def mse_fn(clip_std):
        recon = uniform_quantize_fixed_clip(x, bits, clip_std=clip_std, sigma=sigma)
        return float(np.mean((x - recon) ** 2))

    # Search over reasonable clip range
    res = minimize_scalar(mse_fn, bounds=(0.5, 6.0), method='bounded',
                          options={'xatol': 1e-4})
    best_clip = res.x
    best_mse = res.fun
    return best_mse, best_clip


def lloyd_max_quantize(x, bits, n_iter=100, tol=1e-8):
    """
    Lloyd-Max quantizer (non-uniform, optimal for MSE).
    Centroid = mean of bin, boundaries = midpoints.
    For reference / sanity check.
    """
    n_levels = 2 ** bits
    # Initialize with percentile-based centroids
    pcts = np.linspace(0, 100, n_levels + 2)[1:-1]
    centroids = np.percentile(x, pcts)
    centroids = np.sort(centroids)

    for _ in range(n_iter):
        # Boundaries are midpoints of adjacent centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        # Assign
        idx = np.searchsorted(boundaries, x)
        # Update centroids
        new_centroids = centroids.copy()
        for j in range(n_levels):
            mask = idx == j
            if mask.sum() > 0:
                new_centroids[j] = x[mask].mean()
        if np.max(np.abs(new_centroids - centroids)) < tol:
            break
        centroids = new_centroids

    # Final reconstruction
    boundaries = (centroids[:-1] + centroids[1:]) / 2
    idx = np.searchsorted(boundaries, x)
    return centroids[idx]


# ----------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------

def run_gaussian_experiment(n_samples=1_000_000, seed=42, bits_range=(1, 7)):
    """Run E3 on synthetic Gaussian N(0,1)."""
    print(f"=== E3: Discrete-WF Theorem Verification (Gaussian) ===")
    print(f"n_samples = {n_samples:,}, seed = {seed}")
    print()

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_samples)
    sigma2 = 1.0  # std = 1

    results = {
        'n_samples': n_samples,
        'seed': seed,
        'source': 'gaussian_N(0,1)',
        'sigma_squared': sigma2,
        'per_bit': [],
    }

    header = f"{'b':>3} | {'D_fixed(3σ)':>12} | {'D_opt':>12} | {'clip_opt':>8} | {'D_lloyd':>12} | {'D_Shannon':>12} | {'r_fixed':>8} | {'r_opt':>8} | {'r_lloyd':>8}"
    print(header)
    print('-' * len(header))

    for b in range(bits_range[0], bits_range[1] + 1):
        # Fixed clip = 3 sigma
        recon_fixed = uniform_quantize_fixed_clip(x, b, clip_std=3.0, sigma=1.0)
        D_fixed = float(np.mean((x - recon_fixed) ** 2))

        # Optimal clip
        D_opt, clip_opt = uniform_quantize_optimal(x, b, sigma=1.0)

        # Lloyd-Max (reference, not uniform)
        recon_lloyd = lloyd_max_quantize(x, b, n_iter=200)
        D_lloyd = float(np.mean((x - recon_lloyd) ** 2))

        # Shannon
        D_shannon = sigma2 * (2 ** (-2 * b))

        r_fixed = D_fixed / D_shannon
        r_opt = D_opt / D_shannon
        r_lloyd = D_lloyd / D_shannon

        print(f"{b:>3} | {D_fixed:>12.6f} | {D_opt:>12.6f} | {clip_opt:>8.4f} | "
              f"{D_lloyd:>12.6f} | {D_shannon:>12.6f} | "
              f"{r_fixed:>8.3f} | {r_opt:>8.3f} | {r_lloyd:>8.3f}")

        results['per_bit'].append({
            'b': b,
            'D_uniform_fixed_3sigma': D_fixed,
            'D_uniform_optimal': D_opt,
            'clip_opt': clip_opt,
            'D_lloyd_max': D_lloyd,
            'D_shannon': D_shannon,
            'ratio_fixed_vs_shannon': r_fixed,
            'ratio_optimal_vs_shannon': r_opt,
            'ratio_lloyd_vs_shannon': r_lloyd,
        })

    print()
    return results


def analyze_knee(results):
    """
    Analyze whether there's a 'knee' at b=2:
    - Compute marginal distortion improvement Delta(b-1) - Delta(b)
    - Check if the b=0->1 gain is smaller (per bit) than b=0->2 gain
    - Compute 'convex deficit' at b=1
    """
    print("=== Knee Analysis (Discrete-WF Theorem verification) ===")
    print()

    per_bit = {r['b']: r for r in results['per_bit']}

    # Add b=0 (no quantization, D = sigma^2)
    D_at_0 = results['sigma_squared']

    print("Marginal distortion improvements (using optimal uniform quantizer):")
    print(f"{'transition':>14} | {'delta_D':>12} | {'delta_D / bits':>16}")
    print('-' * 50)

    prev_D = D_at_0
    prev_b = 0
    improvements = []
    for b in sorted(per_bit.keys()):
        D_b = per_bit[b]['D_uniform_optimal']
        delta = prev_D - D_b
        delta_per_bit = delta / (b - prev_b) if (b - prev_b) > 0 else delta
        print(f"{prev_b:>5} -> {b:<5} | {delta:>12.6f} | {delta_per_bit:>16.6f}")
        improvements.append({
            'from_b': prev_b, 'to_b': b,
            'delta_D': delta, 'delta_D_per_bit': delta_per_bit
        })
        prev_D = D_b
        prev_b = b

    print()

    # Direct jump tests:
    # Option A: spend 2 bits as 1 bit (b=1) -> save D(0)-D(1)
    # Option B: spend 2 bits as 2 bits (b=2) -> save D(0)-D(2)
    # Per-bit gain at b=1: (D(0)-D(1))/1
    # Per-bit gain at b=2: (D(0)-D(2))/2
    # If per-bit gain at b=2 > per-bit gain at b=1, then floor=2 is rational
    D1 = per_bit[1]['D_uniform_optimal']
    D2 = per_bit[2]['D_uniform_optimal']
    D3 = per_bit[3]['D_uniform_optimal']

    gain_b1_per_bit = (D_at_0 - D1) / 1
    gain_b2_per_bit = (D_at_0 - D2) / 2
    gain_b3_per_bit = (D_at_0 - D3) / 3

    print("Per-bit gain from 0 bits (higher = more efficient):")
    print(f"  0 -> 1 bit:  {gain_b1_per_bit:.6f} per bit  "
          f"({'!! SUBOPTIMAL !!' if gain_b1_per_bit < gain_b2_per_bit else 'ok'})")
    print(f"  0 -> 2 bits: {gain_b2_per_bit:.6f} per bit")
    print(f"  0 -> 3 bits: {gain_b3_per_bit:.6f} per bit")
    print()

    if gain_b1_per_bit < gain_b2_per_bit:
        print(">>> KNEE CONFIRMED: 1-bit allocation is per-bit suboptimal vs 2-bit.")
        print(">>> Discrete-WF would avoid b=1 -> floor=2 is theoretically justified.")
    else:
        print(">>> No knee detected at b=1 (hypothesis needs revision).")
    print()

    # Classical reference values (Max 1960, Lloyd 1982)
    # Optimal uniform Gaussian quantizer MSEs:
    # b=1: (1 - 2/pi) ≈ 0.3634  (1 level, but Lloyd-Max)
    # b=2: ~0.1175
    # b=3: ~0.03454
    # b=4: ~0.009497
    # Lloyd-Max optimal (non-uniform):
    # b=1: 0.3634 (2 symmetric levels, centroids at ±sqrt(2/pi))
    # b=2: 0.1175
    # b=3: 0.03454
    print("Reference (Max 1960, optimal scalar quantizer for Gaussian):")
    ref = {1: 0.3634, 2: 0.1175, 3: 0.03454, 4: 0.009497, 5: 0.002805, 6: 0.0008215}
    print(f"{'b':>3} | {'ours_optimal_uniform':>22} | {'ours_lloyd':>12} | {'Max_1960':>12}")
    print('-' * 60)
    for b in sorted(per_bit.keys()):
        if b in ref:
            print(f"{b:>3} | {per_bit[b]['D_uniform_optimal']:>22.6f} | "
                  f"{per_bit[b]['D_lloyd_max']:>12.6f} | {ref[b]:>12.6f}")
    print()

    return {
        'knee_detected': bool(gain_b1_per_bit < gain_b2_per_bit),
        'gain_per_bit_at_b1': float(gain_b1_per_bit),
        'gain_per_bit_at_b2': float(gain_b2_per_bit),
        'gain_per_bit_at_b3': float(gain_b3_per_bit),
        'marginal_improvements': [
            {k: float(v) if not isinstance(v, int) else v for k, v in imp.items()}
            for imp in improvements
        ],
    }


def run_alternative_distributions(n_samples=500_000, seed=42):
    """
    Additional check: repeat on heavy-tailed distributions.
    Student-t with df=3 (heavy tail, alpha ≈ 3)
    Laplace (heavy tail, exponential)
    """
    print("=== Alternative distributions (heavy-tail check) ===")
    print()
    rng = np.random.default_rng(seed)

    distributions = {
        'gaussian_N(0,1)': rng.standard_normal(n_samples),
        'student_t_df3': rng.standard_t(df=3, size=n_samples),
        'laplace_scale1': rng.laplace(loc=0.0, scale=1.0, size=n_samples),
    }

    all_results = {}
    for dist_name, x in distributions.items():
        # Normalize to unit variance for fair comparison
        x = x / x.std()
        sigma2 = 1.0
        print(f"--- {dist_name} (normalized to sigma=1) ---")
        dist_results = []
        for b in range(1, 5):
            D_opt, clip_opt = uniform_quantize_optimal(x, b, sigma=1.0)
            D_lloyd = float(np.mean((x - lloyd_max_quantize(x, b, n_iter=150)) ** 2))
            D_shannon = sigma2 * 2 ** (-2 * b)
            print(f"  b={b}: D_opt_uni={D_opt:.5f}, D_lloyd={D_lloyd:.5f}, "
                  f"D_shannon={D_shannon:.5f}, ratio_opt={D_opt/D_shannon:.3f}")
            dist_results.append({
                'b': b, 'D_opt_uniform': D_opt, 'D_lloyd_max': D_lloyd,
                'D_shannon': D_shannon,
                'ratio_opt_vs_shannon': D_opt / D_shannon,
                'ratio_lloyd_vs_shannon': D_lloyd / D_shannon,
            })
        all_results[dist_name] = dist_results
        print()

    return all_results


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    t0 = time.time()
    out_dir = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Primary Gaussian experiment
    results = run_gaussian_experiment(n_samples=1_000_000, seed=42, bits_range=(1, 7))
    knee_analysis = analyze_knee(results)
    results['knee_analysis'] = knee_analysis

    # Alternative distributions
    alt = run_alternative_distributions(n_samples=500_000, seed=42)
    results['alternative_distributions'] = alt

    results['runtime_sec'] = time.time() - t0
    results['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

    out_json = out_dir / 'e3_discrete_wf_results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"=== Results saved: {out_json} ===")
    print(f"Total runtime: {results['runtime_sec']:.1f} sec")
    print()

    # Summary for paper
    print("=== SUMMARY FOR PAPER ===")
    print()
    print("Proposition (Discrete-WF Theorem) says:")
    print("  If D_uniform(b=1)/sigma^2 > 2 * D_uniform(b=2)/sigma^2,")
    print("  then allocating 1 bit is per-bit suboptimal vs 2 bits,")
    print("  and floor=2 is justified.")
    print()
    D1 = results['per_bit'][0]['D_uniform_optimal']
    D2 = results['per_bit'][1]['D_uniform_optimal']
    print(f"Measured (Gaussian):")
    print(f"  D_opt(1)/sigma^2 = {D1:.4f}")
    print(f"  D_opt(2)/sigma^2 = {D2:.4f}")
    print(f"  2 * D_opt(2)     = {2*D2:.4f}")
    print(f"  D_opt(1) > 2 * D_opt(2)? {'YES' if D1 > 2*D2 else 'NO'}")
    print(f"  Gain at b=1: {1 - D1:.4f} (per bit: {1-D1:.4f})")
    print(f"  Gain at b=2: {1 - D2:.4f} (per bit: {(1-D2)/2:.4f})")
    print(f"  Knee confirmed: {knee_analysis['knee_detected']}")


if __name__ == '__main__':
    main()
