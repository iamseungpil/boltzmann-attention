#!/usr/bin/env python3
"""
E3b: Heterogeneous channel Water-Filling — the REAL floor=2 test
================================================================

E3 (Gaussian) showed that for a single channel, per-bit gain decreases
monotonically (Shannon's standard R-D). This does NOT justify floor=2
in the naive "knee" sense.

The REAL test for floor=2 is: given heterogeneous channels with variances
{σ_j²} (PCA-like), does floor=2 give lower total distortion than floor=1?

This simulates discrete Water-Filling on:
  - Gaussian channels (control)
  - PCA-like power-law spectra (realistic)
  - Actual Mistral/Llama/Qwen PCA spectra (if available — else synthetic)

For each: compute total distortion with floor ∈ {0, 1, 2} over a budget B.
"""
import json
import time
from pathlib import Path
from itertools import product

import numpy as np
from scipy.stats import norm


# Optimal scalar quantizer distortion (Max 1960 reference values, measured earlier)
# D(b) / sigma^2 for optimal uniform quantizer of unit Gaussian
D_UNIFORM_PER_SIGMA2 = {
    0: 1.0,       # no quantization = full variance
    1: 0.3642,
    2: 0.1193,
    3: 0.03764,
    4: 0.01163,
    5: 0.003537,
    6: 0.001063,
    7: 0.000317,
    8: 0.0000940,
}

# For Lloyd-Max (non-uniform) which is slightly better
D_LLOYD_PER_SIGMA2 = {
    0: 1.0,
    1: 0.3634,
    2: 0.1175,
    3: 0.03454,
    4: 0.009497,
    5: 0.002805,
    6: 0.000821,
    7: 0.000237,
    8: 0.0000664,
}


def distortion(b, sigma2, table=D_UNIFORM_PER_SIGMA2):
    """Distortion of a b-bit quantizer for a Gaussian channel with variance sigma2."""
    if b in table:
        return table[b] * sigma2
    # Asymptotic: D(b) ≈ sigma^2 * 2^(-2b) * (sqrt(3)*pi/2) for optimal uniform
    return sigma2 * 2 ** (-2 * b) * (np.sqrt(3) * np.pi / 2)


def discrete_wf_allocate(sigma2s, budget, b_min=0, b_max=8, table=D_UNIFORM_PER_SIGMA2):
    """
    Solve discrete Water-Filling by greedy marginal allocation.

    Given N channels with variances sigma2s and total bit budget B (= N * avg_bits),
    allocate bits to minimize total distortion.

    Starting from all b=b_min, greedily add 1 bit to the channel with largest
    marginal distortion reduction, until budget is exhausted.

    Args:
        sigma2s: list of channel variances
        budget: total bits (must be >= N * b_min)
        b_min: minimum bits per channel
        b_max: maximum bits per channel
    Returns:
        allocation: list of bit counts per channel
        total_distortion: sum of per-channel distortions
    """
    n = len(sigma2s)
    if budget < n * b_min:
        raise ValueError(f"budget {budget} < n*b_min {n*b_min}")
    if budget > n * b_max:
        budget = n * b_max

    allocation = [b_min] * n
    spent = n * b_min

    while spent < budget:
        # Find channel with largest marginal gain
        best_gain = -1.0
        best_j = -1
        for j in range(n):
            if allocation[j] >= b_max:
                continue
            cur_D = distortion(allocation[j], sigma2s[j], table)
            new_D = distortion(allocation[j] + 1, sigma2s[j], table)
            gain = cur_D - new_D
            if gain > best_gain:
                best_gain = gain
                best_j = j
        if best_j < 0:
            break
        allocation[best_j] += 1
        spent += 1

    total_D = sum(distortion(b, s, table) for b, s in zip(allocation, sigma2s))
    return allocation, total_D


def generate_spectrum(kind, n, seed=0, decay=None):
    """Generate a spectrum of channel variances."""
    rng = np.random.default_rng(seed)
    if kind == 'gaussian_iid':
        # All channels equal variance
        return np.ones(n)
    elif kind == 'power_law':
        # PCA-like: sigma_j^2 ~ j^(-decay)
        j = np.arange(1, n + 1)
        s = j ** (-(decay or 1.5))
        return s / s.mean()  # normalize so mean = 1
    elif kind == 'exponential':
        j = np.arange(n)
        s = np.exp(-(decay or 0.3) * j)
        return s / s.mean()
    elif kind == 'bimodal':
        # Few high-variance + many low-variance
        s = np.concatenate([
            10.0 * np.ones(n // 4),
            0.1 * np.ones(n - n // 4)
        ])
        return s / s.mean()
    elif kind == 'realistic_llama_like':
        # Approximate: first 10% have high variance, rapid decay
        j = np.arange(1, n + 1)
        s = np.where(j <= n // 10, 5.0 * j ** (-0.5), j ** (-2.0))
        return s / s.mean()
    else:
        raise ValueError(kind)


def test_floors(spectrum_name, sigma2s, avg_bits, floors=(0, 1, 2)):
    """Test different b_min (floor) values under the same budget."""
    n = len(sigma2s)
    budget = int(n * avg_bits)
    results = {}

    for floor in floors:
        if budget < n * floor:
            results[floor] = {
                'allocation': None, 'total_D': None,
                'error': f'budget {budget} < n*floor {n*floor}'
            }
            continue
        alloc, D = discrete_wf_allocate(sigma2s, budget, b_min=floor)
        # Count how many got exactly `floor` bits
        at_floor = sum(1 for b in alloc if b == floor)
        max_alloc = max(alloc)
        results[floor] = {
            'allocation': alloc,
            'total_D': float(D),
            'channels_at_floor': at_floor,
            'max_bits': max_alloc,
            'min_bits': min(alloc),
        }

    return results


def run():
    print("=== E3b: Heterogeneous Water-Filling floor test ===")
    print(f"Using optimal uniform quantizer distortion table (verified vs Max 1960 in E3)")
    print()

    n = 128  # typical head_dim
    spectra = [
        ('iid_equal', generate_spectrum('gaussian_iid', n)),
        ('power_law_1.0', generate_spectrum('power_law', n, decay=1.0)),
        ('power_law_1.5', generate_spectrum('power_law', n, decay=1.5)),
        ('power_law_2.0', generate_spectrum('power_law', n, decay=2.0)),
        ('exponential_0.1', generate_spectrum('exponential', n, decay=0.1)),
        ('exponential_0.3', generate_spectrum('exponential', n, decay=0.3)),
        ('bimodal', generate_spectrum('bimodal', n)),
        ('realistic_llama_like', generate_spectrum('realistic_llama_like', n)),
    ]

    avg_bits_list = [2.0, 3.0, 4.0]

    all_results = {}

    for spec_name, sigma2s in spectra:
        print(f"\n--- Spectrum: {spec_name} (n={n}) ---")
        print(f"  sigma2: min={sigma2s.min():.4f}, max={sigma2s.max():.4f}, "
              f"ratio={sigma2s.max()/sigma2s.min():.2e}, mean={sigma2s.mean():.4f}")

        spec_results = {}
        for avg_bits in avg_bits_list:
            res = test_floors(spec_name, sigma2s, avg_bits, floors=(0, 1, 2, 3))
            spec_results[f'avg_bits_{avg_bits}'] = res

            print(f"  [avg_bits={avg_bits}]")
            baseline_D = None
            for floor, r in res.items():
                if r.get('error'):
                    print(f"    floor={floor}: {r['error']}")
                    continue
                D = r['total_D']
                if baseline_D is None and floor == 0:
                    baseline_D = D
                gain_pct = (1 - D / baseline_D) * 100 if baseline_D and baseline_D > 0 else 0
                at_floor = r['channels_at_floor']
                min_b = r['min_bits']
                max_b = r['max_bits']
                print(f"    floor={floor}: D={D:.6f} ({gain_pct:+5.2f}% vs floor=0) "
                      f"[{at_floor}/{n} at floor, range={min_b}-{max_b}]")

        # Summary: which floor wins?
        best_floors = {}
        for ab_key, res in spec_results.items():
            Ds = [(f, r['total_D']) for f, r in res.items() if r.get('total_D') is not None]
            if Ds:
                best = min(Ds, key=lambda x: x[1])
                best_floors[ab_key] = {'floor': best[0], 'D': best[1]}
        print(f"  -> Best floor per budget: {best_floors}")

        all_results[spec_name] = {
            'sigma2_stats': {
                'min': float(sigma2s.min()),
                'max': float(sigma2s.max()),
                'mean': float(sigma2s.mean()),
                'ratio': float(sigma2s.max() / sigma2s.min()),
            },
            'results': spec_results,
            'best_floor_per_budget': best_floors,
        }

    # Write results
    out_dir = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / 'e3b_heterogeneous_wf_results.json'

    # Serialize: convert numpy/lists
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.integer, np.int64)):
            return int(x)
        if isinstance(x, (np.floating, np.float64)):
            return float(x)
        return x

    with open(out_json, 'w') as f:
        json.dump(clean(all_results), f, indent=2)

    print(f"\n=== Results saved: {out_json} ===")

    # Final summary
    print("\n=== FINAL SUMMARY: Does floor=2 ever win? ===")
    floor_wins = {0: 0, 1: 0, 2: 0, 3: 0}
    for spec_name, sr in all_results.items():
        for ab_key, bf in sr['best_floor_per_budget'].items():
            floor_wins[bf['floor']] += 1
    total = sum(floor_wins.values())
    print(f"Across {total} (spectrum × budget) combinations:")
    for f, cnt in floor_wins.items():
        pct = 100 * cnt / total if total > 0 else 0
        print(f"  floor={f} wins: {cnt}/{total} ({pct:.0f}%)")


if __name__ == '__main__':
    t0 = time.time()
    run()
    print(f"\nTotal runtime: {time.time()-t0:.1f} sec")
