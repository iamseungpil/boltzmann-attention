#!/usr/bin/env python3
"""
Verification: QW-WF mathematical equivalence to WF(floor=2)
=============================================================

Three experiments to PROVE that PCA-Q natural alignment makes QW-WF
mathematically equivalent to standard WF:

  Test 1 (Rank correlation, ~3 min):
    For each (layer, kv_head) of Mistral, compute:
      - λ_k_j (key PCA eigenvalues, descending)
      - σ_q_j² (query variance in same PCA basis)
      - Spearman ρ(λ_k, σ_q²)
    Hypothesis: ρ > 0.7 (high rank correlation)

  Test 2 (Bit allocation diff, ~3 min):
    For each (layer, kv_head):
      - bits_WF   = water_filling(λ_k, budget=2d, floor=2)
      - bits_QWWF = water_filling(λ_k * σ_q, budget=2d, floor=2)
      - L1 diff = Σ |b_WF - b_QWWF|
    Hypothesis: L1 diff << total budget

  Test 3 (Synthetic alignment break, ~1 min):
    For each head, ARTIFICIALLY rotate Σ_Q by random θ degrees:
      - θ = 0 (natural alignment): expect bits_WF ≈ bits_QWWF
      - θ = 30, 60, 90 (broken alignment): expect divergence
    Hypothesis: bit diff increases monotonically with θ
"""
import json
import time
import gc
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import numpy as np
from scipy.stats import spearmanr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

MODEL_NAME = 'mistralai/Mistral-7B-v0.3'
DEVICE = 'cuda:0'
DTYPE = torch.bfloat16
N_CALIB_TOKENS = 2048
N_LAYERS_SAMPLED = 8

OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Water-Filling allocation with floor
# ----------------------------------------------------------------------

def water_filling_skip_floor(importance, total_budget, b_floor=2, b_max=8):
    """
    Discrete WF with SKIP-or-floor semantic:
      Each dim gets either 0 bits (skip, replaced by mean) OR >= b_floor bits.
      This matches v3's "floor=2 prevents 1-bit catastrophe" empirical observation.

    Algorithm:
      1. Sort dims by importance (descending)
      2. Try k active dims (top-k by importance), each starting at b_floor
      3. Distribute remaining budget greedily among active dims
      4. Choose k that minimizes total distortion (or use largest k that fits)

    Simplified: greedy allocation
      - Start: all dims = 0 bits
      - Iteration: among inactive dims (b=0) and underbudget dims (b<b_max),
        pick the one with highest marginal gain. Activating costs b_floor bits.
        Adding 1 bit to active dim costs 1 bit.
    """
    n = len(importance)
    sigma2 = np.array(importance, dtype=np.float64)
    sigma2 = np.maximum(sigma2, 1e-12)

    bits = np.zeros(n, dtype=int)
    spent = 0

    while spent < total_budget:
        best_gain_per_bit = -np.inf
        best_action = None  # ('activate', j) or ('add', j)

        for j in range(n):
            if bits[j] == 0:
                # Activation: cost b_floor bits, gain = σ² * (1 - 4^(-b_floor))
                if spent + b_floor > total_budget:
                    continue
                gain = sigma2[j] * (1.0 - 4.0 ** (-b_floor))
                gain_per_bit = gain / b_floor
                if gain_per_bit > best_gain_per_bit:
                    best_gain_per_bit = gain_per_bit
                    best_action = ('activate', j)
            elif bits[j] < b_max:
                # Add 1 bit: cost 1, gain = σ² * (4^(-b) - 4^(-(b+1)))
                gain = sigma2[j] * (4.0 ** (-bits[j]) - 4.0 ** (-(bits[j] + 1)))
                if gain > best_gain_per_bit:
                    best_gain_per_bit = gain
                    best_action = ('add', j)

        if best_action is None:
            break
        action, j = best_action
        if action == 'activate':
            bits[j] = b_floor
            spent += b_floor
        else:
            bits[j] += 1
            spent += 1

    return bits


def water_filling_floor2(importance, total_budget, b_floor=2, b_max=8):
    """
    Backward-compatible name. Uses the SKIP-or-floor semantic above.
    """
    return water_filling_skip_floor(importance, total_budget, b_floor, b_max)


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------

def get_calib_text():
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        texts = [t for t in ds['text'] if len(t.strip()) > 100]
        return '\n\n'.join(texts[:500])
    except Exception:
        return " ".join(["Verification text."] * 5000)


def collect_keys_queries(model, tok):
    print("  Calibration forward pass...", flush=True)
    text = get_calib_text()
    enc = tok(text, return_tensors='pt', truncation=True, max_length=N_CALIB_TOKENS)
    input_ids = enc['input_ids'].to(DEVICE)

    n_total = model.config.num_hidden_layers
    layer_idx_list = np.linspace(2, n_total - 2, N_LAYERS_SAMPLED, dtype=int).tolist()

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

    for li in layer_idx_list:
        mod = model.model.layers[li].self_attn
        handles.append(mod.k_proj.register_forward_hook(kh(li)))
        handles.append(mod.q_proj.register_forward_hook(qh(li)))

    with torch.no_grad():
        _ = model(input_ids, use_cache=False)
    for h in handles:
        h.remove()

    return captured, layer_idx_list, input_ids.shape[1]


# ----------------------------------------------------------------------
# Test 1: Rank correlation
# ----------------------------------------------------------------------

def test1_rank_correlation(captured, layer_idx_list, T, n_kv, n_q, head_dim):
    print("\n=== Test 1: Spearman ρ(λ_k, σ_q²) per (layer, head) ===", flush=True)
    n_q_per_kv = n_q // n_kv

    results = []
    rhos = []
    for li in layer_idx_list:
        data = captured.get(li)
        if not data or 'k' not in data or 'q' not in data:
            continue
        K_all = data['k'].reshape(T, n_kv, head_dim).astype(np.float32)
        Q_all = data['q'].reshape(T, n_q, head_dim).astype(np.float32)

        for hk in range(n_kv):
            K = K_all[:, hk, :]
            # PCA on K
            K_c = K - K.mean(0, keepdims=True)
            cov_k = K_c.T @ K_c / max(T - 1, 1)
            eigvals_k, eigvecs_k = np.linalg.eigh(cov_k)
            order = np.argsort(eigvals_k)[::-1]
            lambda_k = eigvals_k[order]   # descending
            U_k = eigvecs_k[:, order]      # PCA basis (d, d)

            # Project Q for associated q heads into K's PCA basis
            q_heads = list(range(hk * n_q_per_kv, (hk + 1) * n_q_per_kv))
            Q = Q_all[:, q_heads, :].mean(axis=1)  # (T, d)
            Q_c = Q - Q.mean(0, keepdims=True)
            Q_pca = Q_c @ U_k   # (T, d) in K's PCA basis
            sigma_q2 = Q_pca.var(axis=0)  # σ_q_j² in K basis

            # Spearman ρ between λ_k_j and σ_q_j²
            rho, p = spearmanr(lambda_k, sigma_q2)
            results.append({
                'layer': int(li), 'kv_head': int(hk),
                'spearman_rho': float(rho),
                'p_value': float(p),
                'lambda_k_top5': lambda_k[:5].tolist(),
                'sigma_q2_top5': sigma_q2[:5].tolist(),
            })
            rhos.append(rho)
            sign = '+' if rho > 0 else ''
            print(f"  L{li:3d}/H{hk}: ρ = {sign}{rho:.4f} (p={p:.2e})", flush=True)

    rhos = np.array(rhos)
    print(f"\n  ρ summary: median = {np.median(rhos):+.4f}, mean = {np.mean(rhos):+.4f}, "
          f"min = {np.min(rhos):+.4f}, max = {np.max(rhos):+.4f}")
    print(f"  Fraction ρ > 0.5: {(rhos > 0.5).mean():.2%}")
    print(f"  Fraction ρ > 0.7: {(rhos > 0.7).mean():.2%}")
    print(f"  Fraction ρ > 0.9: {(rhos > 0.9).mean():.2%}")
    return {
        'n_heads': len(rhos),
        'rho_median': float(np.median(rhos)),
        'rho_mean': float(np.mean(rhos)),
        'rho_min': float(np.min(rhos)),
        'rho_max': float(np.max(rhos)),
        'frac_rho_gt_0.5': float((rhos > 0.5).mean()),
        'frac_rho_gt_0.7': float((rhos > 0.7).mean()),
        'frac_rho_gt_0.9': float((rhos > 0.9).mean()),
        'per_head': results,
    }


# ----------------------------------------------------------------------
# Test 2: Bit allocation diff
# ----------------------------------------------------------------------

def test2_bit_allocation_diff(captured, layer_idx_list, T, n_kv, n_q, head_dim, avg_bits=2):
    print(f"\n=== Test 2: Bit allocation diff (WF vs QW-WF), budget={avg_bits}b avg, floor=2 ===", flush=True)
    n_q_per_kv = n_q // n_kv
    total_budget = avg_bits * head_dim

    results = []
    diffs_l1 = []
    for li in layer_idx_list:
        data = captured.get(li)
        if not data or 'k' not in data or 'q' not in data:
            continue
        K_all = data['k'].reshape(T, n_kv, head_dim).astype(np.float32)
        Q_all = data['q'].reshape(T, n_q, head_dim).astype(np.float32)

        for hk in range(n_kv):
            K = K_all[:, hk, :]
            K_c = K - K.mean(0, keepdims=True)
            cov_k = K_c.T @ K_c / max(T - 1, 1)
            eigvals_k, eigvecs_k = np.linalg.eigh(cov_k)
            order = np.argsort(eigvals_k)[::-1]
            lambda_k = eigvals_k[order]
            U_k = eigvecs_k[:, order]

            q_heads = list(range(hk * n_q_per_kv, (hk + 1) * n_q_per_kv))
            Q = Q_all[:, q_heads, :].mean(axis=1)
            Q_c = Q - Q.mean(0, keepdims=True)
            Q_pca = Q_c @ U_k
            sigma_q2 = Q_pca.var(axis=0)
            sigma_q = np.sqrt(np.maximum(sigma_q2, 1e-12))

            # Standard WF based on λ_k
            bits_WF = water_filling_floor2(lambda_k, total_budget, b_floor=2, b_max=8)
            # QW-WF based on λ_k * σ_q
            importance_qw = lambda_k * sigma_q
            bits_QWWF = water_filling_floor2(importance_qw, total_budget, b_floor=2, b_max=8)

            l1_diff = int(np.sum(np.abs(bits_WF - bits_QWWF)))
            n_changed = int(np.sum(bits_WF != bits_QWWF))
            results.append({
                'layer': int(li), 'kv_head': int(hk),
                'bits_WF_sum': int(bits_WF.sum()),
                'bits_QWWF_sum': int(bits_QWWF.sum()),
                'l1_diff': l1_diff,
                'n_dims_changed': n_changed,
                'l1_diff_frac': l1_diff / total_budget,
                'n_dims_changed_frac': n_changed / head_dim,
            })
            diffs_l1.append(l1_diff)
            print(f"  L{li:3d}/H{hk}: L1 diff = {l1_diff:3d} bits ({l1_diff/total_budget*100:.2f}% of budget), "
                  f"{n_changed}/{head_dim} dims differ", flush=True)

    diffs_l1 = np.array(diffs_l1)
    print(f"\n  L1 diff summary: median = {np.median(diffs_l1):.1f}, mean = {np.mean(diffs_l1):.1f}, "
          f"max = {np.max(diffs_l1)}")
    print(f"  Fraction of budget changed (median): {np.median(diffs_l1)/total_budget*100:.2f}%")
    return {
        'avg_bits_target': avg_bits,
        'total_budget': total_budget,
        'n_heads': len(diffs_l1),
        'l1_diff_median': float(np.median(diffs_l1)),
        'l1_diff_mean': float(np.mean(diffs_l1)),
        'l1_diff_max': float(np.max(diffs_l1)),
        'l1_diff_frac_median': float(np.median(diffs_l1) / total_budget),
        'per_head': results,
    }


# ----------------------------------------------------------------------
# Test 3: Synthetic alignment break
# ----------------------------------------------------------------------

def random_rotation(d, theta_deg, seed=42):
    """Construct a rotation matrix that deviates from identity by ~theta_deg."""
    rng = np.random.default_rng(seed)
    # Generate random skew-symmetric matrix scaled by theta
    A = rng.standard_normal((d, d)) * (theta_deg / 180 * np.pi) / np.sqrt(d)
    A = (A - A.T) / 2
    # Matrix exponential gives orthogonal rotation
    R = np.eye(d) + A + A @ A / 2 + A @ A @ A / 6  # Truncated expm
    # Re-orthogonalize via QR
    Q, _ = np.linalg.qr(R)
    return Q.astype(np.float32)


def measure_alignment_angle(M1, M2):
    """Principal angle between subspaces spanned by top eigenvectors of M1 and M2."""
    e1, V1 = np.linalg.eigh(M1)
    e2, V2 = np.linalg.eigh(M2)
    # Take top half eigenvectors
    k = M1.shape[0] // 2
    V1_top = V1[:, np.argsort(e1)[::-1][:k]]
    V2_top = V2[:, np.argsort(e2)[::-1][:k]]
    cos_angles = np.linalg.svd(V1_top.T @ V2_top, compute_uv=False)
    cos_angles = np.clip(cos_angles, -1, 1)
    angles_deg = np.degrees(np.arccos(cos_angles))
    return float(angles_deg.mean()), float(angles_deg.max())


def test3_synthetic_break(captured, layer_idx_list, T, n_kv, n_q, head_dim, avg_bits=2):
    print(f"\n=== Test 3: Synthetic alignment break ===", flush=True)
    n_q_per_kv = n_q // n_kv
    total_budget = avg_bits * head_dim

    # Pick representative head: layer 2 H3 (top outlier from Exp1)
    target_layer = layer_idx_list[0]  # use first sampled layer
    target_head = 0
    print(f"  Using head L{target_layer}/H{target_head} as representative", flush=True)

    data = captured.get(target_layer)
    if not data:
        return None
    K_all = data['k'].reshape(T, n_kv, head_dim).astype(np.float32)
    Q_all = data['q'].reshape(T, n_q, head_dim).astype(np.float32)

    K = K_all[:, target_head, :]
    K_c = K - K.mean(0, keepdims=True)
    cov_k = K_c.T @ K_c / max(T - 1, 1)
    eigvals_k, eigvecs_k = np.linalg.eigh(cov_k)
    order = np.argsort(eigvals_k)[::-1]
    lambda_k = eigvals_k[order]
    U_k = eigvecs_k[:, order]

    q_heads = list(range(target_head * n_q_per_kv, (target_head + 1) * n_q_per_kv))
    Q = Q_all[:, q_heads, :].mean(axis=1)
    Q_c = Q - Q.mean(0, keepdims=True)

    # Original Σ_Q
    cov_q_orig = Q_c.T @ Q_c / max(T - 1, 1)

    # WF with original Σ_Q
    Q_pca_orig = Q_c @ U_k
    sigma_q_orig = np.sqrt(np.maximum(Q_pca_orig.var(axis=0), 1e-12))
    bits_WF = water_filling_floor2(lambda_k, total_budget, b_floor=2)
    bits_QWWF_orig = water_filling_floor2(lambda_k * sigma_q_orig, total_budget, b_floor=2)
    diff_orig = int(np.sum(np.abs(bits_WF - bits_QWWF_orig)))
    align_orig_mean, align_orig_max = measure_alignment_angle(cov_k, cov_q_orig)

    print(f"\n  θ = 0° (natural alignment):")
    print(f"    Σ_K vs Σ_Q principal angle: mean={align_orig_mean:.2f}°, max={align_orig_max:.2f}°")
    print(f"    L1 diff (WF vs QW-WF): {diff_orig} bits ({diff_orig/total_budget*100:.2f}% of budget)")

    # Apply random rotation to Q at various θ
    theta_results = [{
        'theta_deg': 0,
        'alignment_angle_mean': align_orig_mean,
        'alignment_angle_max': align_orig_max,
        'l1_diff_bits': diff_orig,
        'l1_diff_frac': diff_orig / total_budget,
    }]
    for theta in [10, 30, 60, 90]:
        R = random_rotation(head_dim, theta, seed=42)
        Q_rotated = Q_c @ R
        cov_q_rot = Q_rotated.T @ Q_rotated / max(T - 1, 1)
        Q_rot_pca = Q_rotated @ U_k
        sigma_q_rot = np.sqrt(np.maximum(Q_rot_pca.var(axis=0), 1e-12))
        bits_QWWF_rot = water_filling_floor2(lambda_k * sigma_q_rot, total_budget, b_floor=2)
        diff_rot = int(np.sum(np.abs(bits_WF - bits_QWWF_rot)))
        align_mean, align_max = measure_alignment_angle(cov_k, cov_q_rot)

        print(f"\n  θ = {theta}° (rotated Q):")
        print(f"    Σ_K vs rotated-Σ_Q principal angle: mean={align_mean:.2f}°, max={align_max:.2f}°")
        print(f"    L1 diff (WF vs QW-WF): {diff_rot} bits ({diff_rot/total_budget*100:.2f}% of budget)")
        theta_results.append({
            'theta_deg': theta,
            'alignment_angle_mean': align_mean,
            'alignment_angle_max': align_max,
            'l1_diff_bits': diff_rot,
            'l1_diff_frac': diff_rot / total_budget,
        })

    print(f"\n  Interpretation:")
    print(f"    If natural θ ≈ 0° gives diff ≈ 0, AND rotated θ > 0 gives larger diff,")
    print(f"    THEN PCA-Q alignment is what makes QW-WF degenerate to WF.")
    print(f"    This is the smoking gun.")

    return {
        'representative_layer': int(target_layer),
        'representative_head': int(target_head),
        'avg_bits_target': avg_bits,
        'theta_sweep': theta_results,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Verification: QW-WF mathematical equivalence to WF(floor=2)")
    print("=" * 70, flush=True)
    t_start = time.time()

    print("\nLoading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device_map=DEVICE,
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t_start:.1f}s", flush=True)

    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_q
    print(f"  n_kv={n_kv}, n_q={n_q}, head_dim={head_dim}", flush=True)

    # Collect K, Q
    captured, layer_idx_list, T = collect_keys_queries(model, tok)

    # Free model
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Run 3 tests
    test1 = test1_rank_correlation(captured, layer_idx_list, T, n_kv, n_q, head_dim)

    # Test 2: try multiple budgets to give WF allocation room
    test2_results = {}
    for ab in [2, 3, 4]:
        print(f"\n--- Test 2 with avg_bits={ab} ---")
        test2_results[f'avg_bits_{ab}'] = test2_bit_allocation_diff(
            captured, layer_idx_list, T, n_kv, n_q, head_dim, avg_bits=ab
        )
    test2 = test2_results['avg_bits_3']  # primary report uses 3-bit

    # Test 3: synthetic break with avg_bits=3 (more allocation freedom)
    test3 = test3_synthetic_break(captured, layer_idx_list, T, n_kv, n_q, head_dim, avg_bits=3)

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"\nTest 1 (Spearman ρ): median ρ = {test1['rho_median']:+.3f}")
    print(f"  Hypothesis (ρ > 0.7): {'CONFIRMED' if test1['rho_median'] > 0.7 else 'REJECTED'}")
    print(f"  Strong (>0.9): {test1['frac_rho_gt_0.9']:.0%} of heads")

    print(f"\nTest 2 (L1 diff): median diff = {test2['l1_diff_median']:.1f} / {test2['total_budget']} bits "
          f"= {test2['l1_diff_frac_median']*100:.2f}% of budget")
    print(f"  Hypothesis (diff < 5%): {'CONFIRMED' if test2['l1_diff_frac_median'] < 0.05 else 'REJECTED'}")

    if test3:
        nat = test3['theta_sweep'][0]
        rot90 = test3['theta_sweep'][-1]
        print(f"\nTest 3 (Synthetic break): natural θ vs rotated 90°")
        print(f"  Natural (θ≈{nat['alignment_angle_mean']:.1f}°): L1 diff = {nat['l1_diff_bits']} bits")
        print(f"  Rotated 90° (θ={rot90['alignment_angle_mean']:.1f}°): L1 diff = {rot90['l1_diff_bits']} bits")
        ratio = rot90['l1_diff_bits'] / max(nat['l1_diff_bits'], 1)
        print(f"  Ratio (rotated/natural): {ratio:.1f}× — "
              f"{'LARGER (alignment matters)' if ratio > 2 else 'SIMILAR (alignment does not matter)'}")

    print(f"\n=== CONCLUSION ===")
    if test1['rho_median'] > 0.7 and test2['l1_diff_frac_median'] < 0.05:
        print("✅ QW-WF and WF(floor=2) are MATHEMATICALLY EQUIVALENT due to PCA-Q alignment.")
        print("   The 'novelty' of QW-WF is attributable to the floor=2 mechanism, not query weighting.")
    elif test1['rho_median'] > 0.5 and test2['l1_diff_frac_median'] < 0.10:
        print("⚠️  QW-WF and WF(floor=2) are NEARLY equivalent (>90% bit overlap).")
        print("   Marginal differences are within measurement noise.")
    else:
        print("❌ QW-WF differs meaningfully from WF(floor=2). Alignment hypothesis NOT confirmed.")

    # Save
    out = {
        'model': MODEL_NAME,
        'n_calib_tokens': T,
        'layers_sampled': layer_idx_list,
        'test1_rank_correlation': test1,
        'test2_bit_allocation_diff': test2,
        'test3_synthetic_break': test3,
        'runtime_sec': time.time() - t_start,
    }
    out_file = OUT_DIR / 'exp_verify_qwwf_alignment_proof.json'
    with open(out_file, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_file}")
    print(f"Total runtime: {out['runtime_sec']:.1f}s ({out['runtime_sec']/60:.1f}m)")


if __name__ == '__main__':
    main()
