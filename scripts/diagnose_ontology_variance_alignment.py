#!/usr/bin/env python3
"""
diagnose_ontology_variance_alignment — Phase 0 diagnostic (supersedes η_facet).

Problem: the previous pipeline (exp_facet_basis.py / ontology_facet_basis.py)
reported η_facet ≈ 0 for every (layer, head) and concluded that "only 12 of
128 K dimensions effectively span the ontology".  Full-text re-analysis shows
this was a misinterpretation:

  * η_facet = det(B^T Σ B) / Π diag(B^T Σ B) was designed for the OLD quant
    +steering unification framing, which required a complete orthonormal
    basis B ∈ ℝ^{d×d}.  When B is rank-deficient (r_tot ≈ 12 << 128) and
    padded with PCA of the orthogonal complement, det(M) collapses
    numerically because of scale disparity between the facet block and the
    PCA block.  η_facet near 0 is therefore NOT evidence that the ontology
    is poorly aligned — it is a metric-choice artifact.

  * The correct diagnostic for the K-bias steering framing is:
        var_frac = trace(B^T Σ B) / trace(Σ)
        gain    = var_frac / (r_tot / d)
    This asks: what fraction of K-space variance does the ontology basis
    capture, and how much more concentrated is that capture than uniform
    allocation.

This script re-computes the diagnostic from the existing per-pair results
in reports/axis2_theoretical_verification/ontology_facet_basis.json.  No
forward pass is needed — it only re-reads diag_sum and sigma_trace which
were already serialized.

Output: reports/axis2_theoretical_verification/ontology_variance_alignment.json

Finding (2026-04-09, Mistral-7B-v0.3):

  Early layers (L0):    var_frac 0.85–0.98, gain 14–20×
  Middle layers (L15):  var_frac 0.21–0.38, gain 2.4–3.7×
  Deep layers (L31):    var_frac 0.21–0.29, gain 2.1–3.1×

The ontology basis concentrates K-space variance across all layers, not
despite the small r_tot but because of it.  "12 of 128 dimensions" is the
right rank for the signal, not a limitation.
"""

import json
from pathlib import Path
import numpy as np


IN_JSON = Path(
    '/home/woori/workspace_common/boltzmann-attention/reports/'
    'axis2_theoretical_verification/ontology_facet_basis.json'
)
OUT_JSON = Path(
    '/home/woori/workspace_common/boltzmann-attention/reports/'
    'axis2_theoretical_verification/ontology_variance_alignment.json'
)
D_MODEL_PER_HEAD = 128  # Mistral-7B head dimension


def main() -> None:
    if not IN_JSON.exists():
        raise SystemExit(
            f"Input not found: {IN_JSON}\n"
            f"Run scripts/ontology_facet_basis.py first."
        )

    data = json.loads(IN_JSON.read_text())
    per_pair = data['per_pair_results']

    new_per_pair: dict = {}
    var_fracs: list[float] = []
    gains: list[float] = []
    r_tots: list[int] = []
    per_layer_stats: dict[int, dict[str, list]] = {}

    for key, v in per_pair.items():
        r = v['r_tot']
        diag_sum = v['diag_sum']
        sigma_trace = v['sigma_trace']
        if sigma_trace <= 0 or r <= 0:
            continue
        var_frac = diag_sum / sigma_trace
        var_per_dim_in_B = diag_sum / r
        var_per_dim_uniform = sigma_trace / D_MODEL_PER_HEAD
        gain = var_per_dim_in_B / var_per_dim_uniform if var_per_dim_uniform > 0 else 0.0
        layer = v['layer']

        new_per_pair[key] = {
            'layer': layer,
            'head': v['head'],
            'r_tot': r,
            'r_per_facet': v['r_per_facet'],
            'var_frac': float(var_frac),
            'var_per_dim_in_B': float(var_per_dim_in_B),
            'var_per_dim_uniform': float(var_per_dim_uniform),
            'gain_over_uniform': float(gain),
            'eta_facet_legacy': v.get('eta_facet', None),
        }
        var_fracs.append(var_frac)
        gains.append(gain)
        r_tots.append(r)

        per_layer_stats.setdefault(layer, {
            'var_fracs': [], 'gains': [], 'r_tots': [],
        })
        per_layer_stats[layer]['var_fracs'].append(var_frac)
        per_layer_stats[layer]['gains'].append(gain)
        per_layer_stats[layer]['r_tots'].append(r)

    def _summary(xs: list[float]) -> dict:
        arr = np.asarray(xs, dtype=float)
        return {
            'n': int(arr.size),
            'median': float(np.median(arr)),
            'mean': float(np.mean(arr)),
            'q25': float(np.percentile(arr, 25)),
            'q75': float(np.percentile(arr, 75)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
        }

    overall_summary = {
        'n_pairs': len(var_fracs),
        'var_frac': _summary(var_fracs),
        'gain_over_uniform': _summary(gains),
        'r_tot': _summary([float(r) for r in r_tots]),
    }

    layer_summary = {}
    for layer, stats in sorted(per_layer_stats.items()):
        layer_summary[f'L{layer}'] = {
            'var_frac': _summary(stats['var_fracs']),
            'gain_over_uniform': _summary(stats['gains']),
            'r_tot': _summary([float(r) for r in stats['r_tots']]),
        }

    if overall_summary['var_frac']['median'] >= 0.5 and overall_summary['gain_over_uniform']['median'] >= 3.0:
        verdict = (
            "STRONG ALIGNMENT — ontology basis captures a disproportionate "
            "fraction of K-space variance.  K-bias injection along these "
            "directions should produce large attention pattern changes.  "
            "The '12 of 128' framing was a misinterpretation: the ontology "
            "concentrates signal, not limits it."
        )
    elif overall_summary['gain_over_uniform']['median'] >= 2.0:
        verdict = (
            "MODERATE ALIGNMENT — ontology basis is significantly above "
            "uniform allocation (≥2× gain) but does not dominate K-space "
            "variance globally.  K-bias injection should produce "
            "non-trivial but layer-dependent effects.  Proceed with Phase 1."
        )
    else:
        verdict = (
            "WEAK ALIGNMENT — ontology basis is near uniform allocation "
            "(gain < 2×).  Ontology may be poorly suited to this model's "
            "K-space geometry, or the ontology itself may not encode "
            "semantic contrasts that the model has learned to represent.  "
            "Consider a different ontology before proceeding."
        )

    result = {
        'source': str(IN_JSON),
        'd_model_per_head': D_MODEL_PER_HEAD,
        'metric_note': (
            "var_frac = trace(B^T Σ B) / trace(Σ) computed from "
            "diag_sum/sigma_trace in source.  gain_over_uniform = "
            "(diag_sum/r_tot) / (sigma_trace/d).  Supersedes the legacy "
            "eta_facet Hadamard-ratio metric, which was designed for a "
            "quantization framing that required a complete basis and is "
            "numerically unstable when B is rank-deficient."
        ),
        'overall_summary': overall_summary,
        'per_layer_summary': layer_summary,
        'per_pair': new_per_pair,
        'verdict': verdict,
    }

    OUT_JSON.write_text(json.dumps(result, indent=2, default=float))

    print("=" * 72)
    print("diagnose_ontology_variance_alignment")
    print("=" * 72)
    print()
    print(f"Read from : {IN_JSON.name}")
    print(f"Wrote to  : {OUT_JSON.name}")
    print()
    print(f"n pairs   : {overall_summary['n_pairs']}")
    print()
    print("Overall (all layers):")
    vf = overall_summary['var_frac']
    g = overall_summary['gain_over_uniform']
    rt = overall_summary['r_tot']
    print(f"  var_frac: median={vf['median']:.4f}  mean={vf['mean']:.4f}  "
          f"IQR=[{vf['q25']:.4f}, {vf['q75']:.4f}]")
    print(f"  gain    : median={g['median']:.1f}x  mean={g['mean']:.1f}x  "
          f"range=[{g['min']:.1f}x, {g['max']:.1f}x]")
    print(f"  r_tot   : median={int(rt['median'])}  range=[{int(rt['min'])}, {int(rt['max'])}]")
    print()
    print("Per layer:")
    for layer_key, stats in layer_summary.items():
        vf = stats['var_frac']
        g = stats['gain_over_uniform']
        rt = stats['r_tot']
        print(f"  {layer_key:<4}: var_frac median={vf['median']:.4f}  "
              f"gain median={g['median']:.1f}x  r_tot median={int(rt['median'])}")
    print()
    print(f"Verdict: {verdict}")


if __name__ == '__main__':
    main()
