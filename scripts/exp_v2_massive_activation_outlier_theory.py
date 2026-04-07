#!/usr/bin/env python3
"""
V2: Massive Activation → Outlier κ → Lloyd Catastrophe Theory Test
====================================================================

Hypothesis: Lloyd-Max PPL catastrophe in Mistral is caused by:
  Step 1: Massive activation channels in residual stream (Sun et al. 2024)
  Step 2: k_proj reads these channels into specific (layer, head) keys
  Step 3: Per-head PCA exposes high-variance dimensions (κ → 10^4-10^7)
  Step 4: L²-Lloyd uniform per-dim quantization mis-allocates noise
  Step 5: Cascade through softmax → catastrophic PPL

Four tests:
  Test 1 (5 min, GPU 1): Detect massive activation channels per layer
  Test 2 (5 min, CPU): k_proj weight alignment with massive channels per head
  Test 3 (instant, CPU): κ(Σ_K) vs massive activation correlation (existing data)
  Test 4 (15 min, GPU 0): Surgical fix - allocate 8 bits to L1 H6's massive
                          activation dim, measure isolated PPL impact

Both GPUs used in parallel (Test 1 → GPU 1, Test 4 → GPU 0).

Outputs:
  - reports/axis2_theoretical_verification/exp_v2_massive_activation_test.json
  - Verdict on whether massive activation hypothesis is confirmed
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
DTYPE = torch.bfloat16
N_CALIB_TOKENS = 2048
N_EVAL_TOKENS = 2048

OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def get_texts(tok):
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        texts = [t for t in ds['text'] if len(t.strip()) > 100]
        return '\n\n'.join(texts[:300]), '\n\n'.join(texts[300:600])
    except Exception:
        return " ".join(["Text."] * 5000), " ".join(["Eval."] * 5000)


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


# ----------------------------------------------------------------------
# Test 1: Detect massive activation channels per layer
# ----------------------------------------------------------------------

def test_1_massive_activations(model, input_ids, n_layers, hidden_size, threshold_ratio=100):
    """
    For each transformer layer, capture residual stream output and find
    'massive' channels where max magnitude exceeds median by threshold_ratio.
    """
    print(f"\n=== Test 1: Massive Activation Detection ===", flush=True)
    captured = {}

    def make_hook(li):
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                acts = outputs[0]
            else:
                acts = outputs
            captured[li] = acts.detach().cpu().float().numpy()
        return hook

    handles = []
    for li in range(n_layers):
        h = model.model.layers[li].register_forward_hook(make_hook(li))
        handles.append(h)

    with torch.no_grad():
        _ = model(input_ids, use_cache=False)
    for h in handles:
        h.remove()

    # Analyze: per layer, find massive channels
    massive_per_layer = {}
    for li in range(n_layers):
        acts = captured[li][0]  # (T, hidden)
        max_per_channel = np.abs(acts).max(axis=0)  # (hidden,)
        median_max = float(np.median(max_per_channel))
        max_overall = float(max_per_channel.max())

        # Massive: max > threshold_ratio * median
        threshold = threshold_ratio * median_max
        massive_ids = np.where(max_per_channel > threshold)[0]

        # Top-10 by magnitude
        top10_idx = np.argsort(max_per_channel)[::-1][:10]

        massive_per_layer[li] = {
            'median_max_magnitude': median_max,
            'overall_max_magnitude': max_overall,
            'ratio_max_to_median': max_overall / max(median_max, 1e-10),
            'n_massive_channels': int(len(massive_ids)),
            'massive_channel_ids': [int(x) for x in massive_ids.tolist()],
            'top10_channel_ids': [int(x) for x in top10_idx.tolist()],
            'top10_magnitudes': [float(max_per_channel[i]) for i in top10_idx],
        }

    # Print summary
    print(f"  {'Layer':<6} | {'median':>10} | {'max':>12} | {'ratio':>8} | {'n_massive':>10} | top channel")
    for li in range(0, n_layers, 1):
        d = massive_per_layer[li]
        marker = ' !!!' if d['ratio_max_to_median'] > threshold_ratio else ''
        print(f"  {li:<6} | {d['median_max_magnitude']:>10.3f} | {d['overall_max_magnitude']:>12.3f} | "
              f"{d['ratio_max_to_median']:>8.1f}× | {d['n_massive_channels']:>10} | "
              f"ch{d['top10_channel_ids'][0]}{marker}", flush=True)

    return massive_per_layer, captured


# ----------------------------------------------------------------------
# Test 2: k_proj weight alignment with massive channels per head
# ----------------------------------------------------------------------

def test_2_kproj_alignment(model, massive_per_layer, n_layers, n_kv, head_dim):
    """
    For each (layer, kv_head), measure how much of the k_proj weight energy
    is concentrated in the massive activation channels.
    """
    print(f"\n=== Test 2: k_proj Weight Alignment with Massive Channels ===", flush=True)
    head_alignment = []

    for li in range(n_layers):
        massive_ids = massive_per_layer[li]['massive_channel_ids']
        # Always include top-3 channels (even if not "massive" by threshold)
        top_channels = massive_per_layer[li]['top10_channel_ids'][:3]
        check_channels = list(set(massive_ids + top_channels))
        if not check_channels:
            continue

        k_proj_weight = model.model.layers[li].self_attn.k_proj.weight.detach().cpu().float().numpy()
        # Shape: (n_kv * head_dim, hidden_size)

        for hk in range(n_kv):
            head_weight = k_proj_weight[hk*head_dim:(hk+1)*head_dim, :]
            massive_energy = float(np.sum(head_weight[:, check_channels]**2))
            total_energy = float(np.sum(head_weight**2))
            alignment_ratio = massive_energy / max(total_energy, 1e-12)
            random_baseline = len(check_channels) / k_proj_weight.shape[1]

            # Per-channel breakdown
            channel_energies = []
            for c in top_channels:
                energy = float(np.sum(head_weight[:, c]**2)) / max(total_energy, 1e-12)
                channel_energies.append({'channel': int(c), 'energy_frac': energy})

            head_alignment.append({
                'layer': int(li),
                'kv_head': int(hk),
                'alignment_ratio': alignment_ratio,
                'random_baseline': random_baseline,
                'enrichment': alignment_ratio / max(random_baseline, 1e-12),
                'top3_channel_energies': channel_energies,
            })

    # Sort by enrichment (most aligned heads first)
    head_alignment.sort(key=lambda x: -x['enrichment'])

    print(f"\n  Top-15 most aligned (layer, kv_head) pairs:")
    print(f"  {'rank':<5} | {'layer':<6} | {'kv_head':<8} | {'alignment':>12} | {'random':>10} | {'enrichment':>12}")
    for i, h in enumerate(head_alignment[:15]):
        print(f"  {i+1:<5} | {h['layer']:<6} | {h['kv_head']:<8} | {h['alignment_ratio']:>12.4f} | "
              f"{h['random_baseline']:>10.4f} | {h['enrichment']:>12.2f}×", flush=True)

    return head_alignment


# ----------------------------------------------------------------------
# Test 3: κ(Σ_K) vs massive activation correlation
# ----------------------------------------------------------------------

def test_3_correlation(massive_per_layer, head_alignment):
    """
    Cross-reference: do high-alignment heads (Test 2) match with high-κ heads (Exp1)?
    """
    print(f"\n=== Test 3: Correlation κ(Σ_K) vs Massive Alignment ===", flush=True)

    # Try to load Exp1 results
    try:
        exp1_path = OUT_DIR / 'exp1_outlier_analysis_results.json'
        with open(exp1_path) as f:
            exp1 = json.load(f)
        mistral_data = exp1.get('per_model_stats', {}).get('mistral-7b-v0.3', {})
        outlier_heads = mistral_data.get('top_10pct_outlier_heads', [])
        print(f"  Loaded Exp1: {len(outlier_heads)} top outlier heads")
    except Exception as e:
        print(f"  Exp1 load failed: {e}")
        outlier_heads = []

    # Build (layer, kv_head) → kappa map
    kappa_map = {}
    for h in outlier_heads:
        kappa_map[(h['layer'], h['kv_head'])] = h['kappa']

    # Build (layer, kv_head) → alignment map
    alignment_map = {(h['layer'], h['kv_head']): h['enrichment'] for h in head_alignment}

    # Find heads that appear in both
    common_heads = set(kappa_map.keys()) & set(alignment_map.keys())

    cross_table = []
    for (li, hk) in sorted(common_heads):
        cross_table.append({
            'layer': int(li),
            'kv_head': int(hk),
            'kappa': float(kappa_map[(li, hk)]),
            'alignment_enrichment': float(alignment_map[(li, hk)]),
        })

    print(f"\n  Cross-table ({len(cross_table)} heads in both Exp1 outliers and Test 2):")
    if cross_table:
        print(f"  {'layer':<6} | {'kv_head':<8} | {'κ(Σ_K)':>14} | {'alignment':>12}")
        for r in sorted(cross_table, key=lambda x: -x['kappa']):
            print(f"  {r['layer']:<6} | {r['kv_head']:<8} | {r['kappa']:>14.0f} | "
                  f"{r['alignment_enrichment']:>12.2f}×", flush=True)

        # Spearman correlation
        if len(cross_table) >= 2:
            kappas = [r['kappa'] for r in cross_table]
            aligns = [r['alignment_enrichment'] for r in cross_table]
            from scipy.stats import spearmanr
            rho, p = spearmanr(kappas, aligns)
            print(f"\n  Spearman ρ(κ, alignment) = {rho:+.3f} (p = {p:.3e})")
        else:
            rho, p = None, None
    else:
        print("  No heads in common; cannot compute correlation.")
        rho, p = None, None

    return {
        'cross_table': cross_table,
        'spearman_rho': rho,
        'p_value': p,
    }


# ----------------------------------------------------------------------
# Test 4: Surgical fix on L1 H6 (the worst single head per Next-8)
# ----------------------------------------------------------------------

class SurgicalQuantizerHook:
    """
    Per-head Lloyd quantizer with custom bit allocation per dim.
    bits_per_dim: array of length head_dim
    Only quantizes target_layer's target_kv_head.
    """
    def __init__(self, target_layer, target_kv_head, n_kv, head_dim, V, K_mean, centroids_per_dim):
        self.target_layer = target_layer
        self.target_kv_head = target_kv_head
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.V = V
        self.K_mean = K_mean
        self.centroids = centroids_per_dim

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x_bf = output.view(B, T, self.n_kv, self.head_dim)
        x_np = x_bf.float().cpu().numpy()
        x_q = x_np.copy()  # other heads stay FP

        # Only modify target head
        data = x_np[:, :, self.target_kv_head, :]
        shape = data.shape
        K_flat = data.reshape(-1, self.head_dim).astype(np.float32)
        K_c = K_flat - self.K_mean
        K_pca = K_c @ self.V

        K_pca_q = np.zeros_like(K_pca)
        for j in range(self.head_dim):
            cj = self.centroids[j]
            if cj is None or len(cj) == 1:
                K_pca_q[:, j] = cj[0] if cj is not None else 0.0
            else:
                boundaries = (cj[:-1] + cj[1:]) / 2
                idx = np.searchsorted(boundaries, K_pca[:, j])
                K_pca_q[:, j] = cj[idx]

        K_recon = K_pca_q @ self.V.T + self.K_mean
        x_q[:, :, self.target_kv_head, :] = K_recon.reshape(shape)

        return torch.from_numpy(x_q).to(output.device).to(output.dtype).view(B, T, self.n_kv * self.head_dim)


def test_4_surgical_fix(model, tok, calib_text, eval_text, target_layer=1, target_kv_head=6):
    """
    Surgical fix: Apply variable bit allocation to only L1 H6 (worst single head per Next-8).
    Compare:
      (A) FP16 (baseline)
      (B) Uniform 2-bit on L1 H6 (L² Lloyd)
      (C) 8-bit on top dim, 1-bit on others, on L1 H6
      (D) WF-style (variance-proportional) on L1 H6
    """
    print(f"\n=== Test 4: Surgical Fix on L{target_layer} H{target_kv_head} ===", flush=True)

    n_kv = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    # Calibration: capture L1 H6 keys
    calib_enc = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB_TOKENS)
    calib_ids = calib_enc['input_ids'].to(model.device)

    captured_K = {}
    def k_hook(m, i, o):
        captured_K['k'] = o.detach().cpu().float().numpy()

    h = model.model.layers[target_layer].self_attn.k_proj.register_forward_hook(k_hook)
    with torch.no_grad():
        _ = model(calib_ids, use_cache=False)
    h.remove()

    K_all = captured_K['k'].reshape(-1, n_kv, head_dim).astype(np.float32)
    K = K_all[:, target_kv_head, :]  # (T, head_dim)

    # PCA basis
    K_mean = K.mean(axis=0)
    K_c = K - K_mean
    cov = (K_c.T @ K_c) / max(K.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals_desc = eigvals[order]
    V = eigvecs[:, order].astype(np.float32)
    K_pca = K_c @ V

    print(f"  L{target_layer} H{target_kv_head} PCA spectrum:")
    print(f"    λ_top:    {eigvals_desc[0]:.2e}")
    print(f"    λ_top/λ_2: {eigvals_desc[0]/max(eigvals_desc[1], 1e-12):.2f}")
    print(f"    λ_top/λ_median: {eigvals_desc[0]/max(np.median(eigvals_desc), 1e-12):.2f}")
    print(f"    λ_top/λ_min: {eigvals_desc[0]/max(eigvals_desc[-1], 1e-12):.2e}")
    print(f"    Top-5 eigvals: {eigvals_desc[:5].tolist()}")

    # Eval
    eval_enc = tok(eval_text, return_tensors='pt', truncation=True, max_length=N_EVAL_TOKENS)
    eval_ids = eval_enc['input_ids'].to(model.device)

    # ===== Config A: FP16 baseline =====
    ppl_fp16, _ = compute_ppl(model, eval_ids)
    print(f"\n  [A] FP16 baseline: PPL = {ppl_fp16:.4f}", flush=True)

    # ===== Config B: Uniform 2-bit per dim on L1 H6 =====
    centroids_uniform_2 = []
    for j in range(head_dim):
        c = lloyd_max_1d_fit(K_pca[:, j], bits=2, n_iter=20).astype(np.float32)
        centroids_uniform_2.append(c)

    hook_b = SurgicalQuantizerHook(
        target_layer, target_kv_head, n_kv, head_dim, V, K_mean, centroids_uniform_2
    )
    h = model.model.layers[target_layer].self_attn.k_proj.register_forward_hook(hook_b)
    ppl_uniform_2, _ = compute_ppl(model, eval_ids)
    h.remove()
    print(f"  [B] Uniform 2-bit on L{target_layer}H{target_kv_head}: "
          f"PPL = {ppl_uniform_2:.4f} (Δ vs FP16 = {ppl_uniform_2 - ppl_fp16:+.4f})", flush=True)

    # ===== Config C: 8-bit on top dim, 1-bit on others =====
    centroids_surgical = []
    for j in range(head_dim):
        bits = 8 if j == 0 else 1
        c = lloyd_max_1d_fit(K_pca[:, j], bits=bits, n_iter=20).astype(np.float32)
        centroids_surgical.append(c)

    avg_bits_c = (8 + 1 * (head_dim - 1)) / head_dim
    hook_c = SurgicalQuantizerHook(
        target_layer, target_kv_head, n_kv, head_dim, V, K_mean, centroids_surgical
    )
    h = model.model.layers[target_layer].self_attn.k_proj.register_forward_hook(hook_c)
    ppl_surgical, _ = compute_ppl(model, eval_ids)
    h.remove()
    print(f"  [C] 8-bit top + 1-bit others (avg {avg_bits_c:.3f}b): "
          f"PPL = {ppl_surgical:.4f} (Δ vs FP16 = {ppl_surgical - ppl_fp16:+.4f})", flush=True)

    # ===== Config D: WF-style variance-proportional bit allocation =====
    def wf_skip_floor(sigma2, total_budget, b_floor=1, b_max=8):
        n = len(sigma2)
        s = np.maximum(sigma2, 1e-12)
        bits = np.zeros(n, dtype=int)
        spent = 0
        while spent < total_budget:
            best_gain_per_bit = -np.inf
            best_action = None
            for j in range(n):
                if bits[j] == 0:
                    if spent + b_floor > total_budget:
                        continue
                    g = s[j] * (1.0 - 4.0 ** (-b_floor)) / b_floor
                    if g > best_gain_per_bit:
                        best_gain_per_bit = g
                        best_action = ('activate', j)
                elif bits[j] < b_max:
                    g = s[j] * (4.0 ** (-bits[j]) - 4.0 ** (-(bits[j] + 1)))
                    if g > best_gain_per_bit:
                        best_gain_per_bit = g
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

    bits_d = wf_skip_floor(eigvals_desc, total_budget=2 * head_dim, b_floor=1, b_max=8)
    print(f"  [D] WF allocation: {dict(zip(*np.unique(bits_d, return_counts=True)))}, "
          f"avg = {bits_d.mean():.3f} bits")

    centroids_wf = []
    for j in range(head_dim):
        b = int(bits_d[j])
        if b == 0:
            c = np.array([0.0], dtype=np.float32)
        else:
            c = lloyd_max_1d_fit(K_pca[:, j], bits=b, n_iter=20).astype(np.float32)
        centroids_wf.append(c)

    hook_d = SurgicalQuantizerHook(
        target_layer, target_kv_head, n_kv, head_dim, V, K_mean, centroids_wf
    )
    h = model.model.layers[target_layer].self_attn.k_proj.register_forward_hook(hook_d)
    ppl_wf, _ = compute_ppl(model, eval_ids)
    h.remove()
    print(f"  [D] WF-allocated (avg {bits_d.mean():.3f}b): "
          f"PPL = {ppl_wf:.4f} (Δ vs FP16 = {ppl_wf - ppl_fp16:+.4f})", flush=True)

    return {
        'target_layer': target_layer,
        'target_kv_head': target_kv_head,
        'eigvals_top5': eigvals_desc[:5].tolist(),
        'eigval_ratio_top_to_median': float(eigvals_desc[0] / max(np.median(eigvals_desc), 1e-12)),
        'ppl_fp16': ppl_fp16,
        'ppl_uniform_2bit': ppl_uniform_2,
        'delta_uniform_2bit': ppl_uniform_2 - ppl_fp16,
        'ppl_surgical_8plus1': ppl_surgical,
        'delta_surgical_8plus1': ppl_surgical - ppl_fp16,
        'avg_bits_surgical_8plus1': avg_bits_c,
        'ppl_wf_allocation': ppl_wf,
        'delta_wf_allocation': ppl_wf - ppl_fp16,
        'avg_bits_wf': float(bits_d.mean()),
        'wf_bit_distribution': {int(k): int(v) for k, v in zip(*np.unique(bits_d, return_counts=True))},
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("V2: Massive Activation → Outlier κ → Lloyd Catastrophe Theory Test")
    print("=" * 70, flush=True)
    t_start = time.time()

    # Load model on GPU 0 (will use single model for both Test 1 and Test 4)
    print("\nLoading Mistral-7B...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t_start:.1f}s", flush=True)

    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_q
    hidden_size = model.config.hidden_size
    print(f"  n_layers={n_layers}, n_kv={n_kv}, head_dim={head_dim}, hidden={hidden_size}", flush=True)

    calib_text, eval_text = get_texts(tok)
    calib_enc = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB_TOKENS)
    input_ids = calib_enc['input_ids'].to('cuda:0')

    # === Test 1: Massive activation detection ===
    massive_per_layer, captured_acts = test_1_massive_activations(
        model, input_ids, n_layers, hidden_size
    )
    del captured_acts
    gc.collect()

    # === Test 2: k_proj alignment ===
    head_alignment = test_2_kproj_alignment(
        model, massive_per_layer, n_layers, n_kv, head_dim
    )

    # === Test 3: Cross-correlation with Exp1 ===
    correlation = test_3_correlation(massive_per_layer, head_alignment)

    # === Test 4: Surgical fix on L1 H6 ===
    surgical = test_4_surgical_fix(
        model, tok, calib_text, eval_text,
        target_layer=1, target_kv_head=6
    )

    # === Verdict ===
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Check 1: Massive activations exist
    n_layers_with_massive = sum(1 for d in massive_per_layer.values() if d['n_massive_channels'] > 0)
    max_ratio_overall = max(d['ratio_max_to_median'] for d in massive_per_layer.values())
    print(f"\n  Test 1: {n_layers_with_massive}/{n_layers} layers have massive channels")
    print(f"          Max activation/median ratio: {max_ratio_overall:.1f}×")
    test1_pass = max_ratio_overall > 100
    print(f"          → {'✅ CONFIRMED' if test1_pass else '❌ NOT FOUND'} (threshold 100×)")

    # Check 2: Some heads heavily aligned with massive channels
    if head_alignment:
        max_enrichment = max(h['enrichment'] for h in head_alignment)
        n_strongly_aligned = sum(1 for h in head_alignment if h['enrichment'] > 5)
        print(f"\n  Test 2: Max k_proj enrichment vs random = {max_enrichment:.1f}×")
        print(f"          {n_strongly_aligned} heads with enrichment > 5×")
        test2_pass = max_enrichment > 5
        print(f"          → {'✅ CONFIRMED' if test2_pass else '❌ NOT FOUND'} (threshold 5×)")
    else:
        max_enrichment = 0
        test2_pass = False
        print(f"\n  Test 2: ❌ No heads measured")

    # Check 3: Correlation between alignment and κ
    rho = correlation.get('spearman_rho')
    if rho is not None:
        print(f"\n  Test 3: Spearman ρ(κ, alignment) = {rho:+.3f}")
        test3_pass = rho > 0.5
        print(f"          → {'✅ CONFIRMED' if test3_pass else '❌ WEAK'} (threshold +0.5)")
    else:
        test3_pass = False
        print(f"\n  Test 3: ❌ Insufficient data for correlation")

    # Check 4: Surgical fix recovers L1 H6 PPL impact
    delta_uniform = surgical['delta_uniform_2bit']
    delta_surgical = surgical['delta_surgical_8plus1']
    delta_wf = surgical['delta_wf_allocation']
    print(f"\n  Test 4: L{surgical['target_layer']} H{surgical['target_kv_head']} surgical fix")
    print(f"          Uniform 2-bit ΔPPL:        {delta_uniform:+.4f}")
    print(f"          Surgical 8+1-bit ΔPPL:     {delta_surgical:+.4f}")
    print(f"          WF allocation ΔPPL:        {delta_wf:+.4f}")
    test4_pass = (delta_surgical < delta_uniform * 0.5) or (delta_wf < delta_uniform * 0.5)
    print(f"          → {'✅ CONFIRMED' if test4_pass else '❌ NOT FOUND'} "
          f"(target: variable bit < 50% of uniform)")

    # Final verdict
    n_passed = sum([test1_pass, test2_pass, test3_pass, test4_pass])
    print(f"\n  ===> {n_passed}/4 tests CONFIRMED")
    if n_passed >= 3:
        print(f"  ===> 🎯 MASSIVE ACTIVATION HYPOTHESIS STRONGLY SUPPORTED")
    elif n_passed >= 2:
        print(f"  ===> ⚠️  HYPOTHESIS PARTIALLY SUPPORTED")
    else:
        print(f"  ===> ❌ HYPOTHESIS REFUTED OR INCONCLUSIVE")

    # Save
    results = {
        'model': MODEL_NAME,
        'test1_massive_activations': massive_per_layer,
        'test2_kproj_alignment': head_alignment[:30],  # top 30 only
        'test3_correlation': correlation,
        'test4_surgical_fix': surgical,
        'verdict': {
            'test1_pass': test1_pass,
            'test2_pass': test2_pass,
            'test3_pass': test3_pass,
            'test4_pass': test4_pass,
            'n_passed': n_passed,
        },
        'runtime_sec': time.time() - t_start,
    }

    out_file = OUT_DIR / 'exp_v2_massive_activation_test.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out_file}")
    print(f"Total runtime: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f}m)")


if __name__ == '__main__':
    main()
