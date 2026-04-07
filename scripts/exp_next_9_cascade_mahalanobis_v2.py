#!/usr/bin/env python3
"""
Next-9: Cascade-Aware Fisher Mahalanobis Lloyd (Gap 2 Resolution)
==================================================================

Combines:
  - Gap 1 Track A: Measure cascade factor g_{l,h} via backward pass
  - Gap 2 Approach A: Two-level decoupled optimization
    (Inter-head budget via Theorem B + Intra-head Mahalanobis Lloyd)
  - Numerical stability: eigenvalue clipping + float32 critical path

Goal: Fix Next-4 config D (Fisher Mahalanobis 982 PPL catastrophe)
      while beating Config E (L² Lloyd + outlier preservation, 6.95 PPL).

Success criteria (in order):
  1. PPL < 982 (Next-4 D) — numerical stability
  2. PPL < 9.12 (Next-4 C, L² Lloyd all-2b) — Theorem A
  3. PPL < 7.90 (Next-4 B, Uniform 2b) — Fisher gain
  4. PPL < 6.95 (Next-4 E, outlier preservation) — main method
  5. PPL < 6.46 (v3 Uniform 2b reference) — new SOTA
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
AVG_BITS = 2.0     # target average bits across (layer, kv_head)
B_FLOOR = 2
B_MAX = 6
EIGVAL_CLIP_RATIO = 1e-4

OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Phase 1a: Cascade factor g_{l,h} measurement via backward
# ----------------------------------------------------------------------

def measure_cascade_factors(model, input_ids, n_layers, n_kv, n_q, head_dim):
    """
    Measure g_{l,h} = ||∂loss/∂attn_out_{l,h}||² per (layer, kv_head).

    For GQA: kv_head receives gradient summed over its Q heads.
    """
    # Register backward hooks on self_attn output
    grad_captures = {}

    def make_bh(li):
        def h(grad):
            # grad shape: (B, T, n_q * head_dim) or (B, T, hidden_size)
            grad_captures[li] = grad.detach().cpu().float().numpy()
        return h

    handles = []
    for li in range(n_layers):
        # Hook on the attention output tensor (the first output of self_attn)
        # We need to register on the tensor after it's computed.
        # Strategy: register hook on a dummy identity via forward hook that saves input for backward
        attn_mod = model.model.layers[li].self_attn

        # Use a forward hook to grab the attention output and register its grad hook
        def make_fh(li_local):
            def fh(mod, inputs, output):
                # output may be (attn_out, attn_weights, past_key_values) or just tensor
                if isinstance(output, tuple):
                    attn_out = output[0]
                else:
                    attn_out = output
                # Only register if tensor requires grad
                if attn_out.requires_grad:
                    attn_out.register_hook(make_bh(li_local))
            return fh

        handles.append(attn_mod.register_forward_hook(make_fh(li)))

    # Enable grad on input_ids? No - we need grad on model params.
    # Actually we want grad of loss w.r.t. intermediate activations.
    # Forward pass with grad enabled:
    model.zero_grad()
    outputs = model(input_ids, use_cache=False)
    logits = outputs.logits
    targets = input_ids[:, 1:].contiguous()
    shift_logits = logits[:, :-1, :].contiguous()
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        targets.reshape(-1),
        reduction='mean'
    )
    loss.backward()

    for h in handles:
        h.remove()

    # Process captured grads
    n_q_per_kv = n_q // n_kv
    g_table = np.zeros((n_layers, n_kv), dtype=np.float32)

    for li, grad in grad_captures.items():
        # grad shape: (1, T, hidden_size) where hidden_size = n_q * head_dim
        T = grad.shape[1]
        grad_reshaped = grad.reshape(1, T, n_q, head_dim)  # (1, T, H_q, d)
        # Per-kv grouping: sum squared norms over associated Q heads
        for hk in range(n_kv):
            q_slice = grad_reshaped[0, :, hk * n_q_per_kv : (hk+1) * n_q_per_kv, :]
            # q_slice: (T, n_q_per_kv, d)
            # g = mean over tokens of mean over q_heads of ||grad||²
            g_val = np.mean(np.sum(q_slice ** 2, axis=-1))  # mean over (T, q_heads)
            g_table[li, hk] = float(g_val)

    return g_table


# ----------------------------------------------------------------------
# Phase 1b: Fisher metric M_{l,h} from calibration forward
# ----------------------------------------------------------------------

def compute_fisher_metrics_and_keys(model, input_ids, n_layers, n_kv, n_q, head_dim):
    """
    Forward pass with output_attentions=True to capture K, Q, attn.
    Returns per-layer dict with 'k', 'q', 'attn'.
    """
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

    with torch.no_grad():
        _ = model(input_ids, output_attentions=True, use_cache=False)

    for h in handles:
        h.remove()

    return captured


# ----------------------------------------------------------------------
# Phase 2: Global Budget Allocation (Theorem B)
# ----------------------------------------------------------------------

def water_filling_global(importance, total_budget, b_floor=2, b_max=6):
    """
    Greedy Water-Filling with floor constraint.
    importance: array of (n_heads,) = g_{l,h} * tr(M_{l,h})
    total_budget: integer total bits
    Returns: bits array (n_heads,)
    """
    n = len(importance)
    imp = np.array(importance, dtype=np.float64)
    imp = np.maximum(imp, 1e-12)

    # Initialize all at floor
    bits = np.full(n, b_floor, dtype=int)
    spent = n * b_floor

    if spent > total_budget:
        # Over-budget even at floor; return floor
        return bits

    # Greedy: add 1 bit to head with largest marginal gain
    # D(b) ≈ c · 4^(-b), marginal gain per head = imp * (4^(-b) - 4^(-(b+1)))
    while spent < total_budget:
        valid = bits < b_max
        if not valid.any():
            break
        gains = np.where(
            valid,
            imp * (4.0 ** (-bits.astype(float)) - 4.0 ** (-(bits + 1).astype(float))),
            -np.inf
        )
        j_best = int(np.argmax(gains))
        bits[j_best] += 1
        spent += 1

    return bits


# ----------------------------------------------------------------------
# Phase 3: Per-head Mahalanobis Lloyd fit (numerically stable)
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


def fit_mahalanobis_lloyd_stable(K, M_avg, bits, eigval_clip_ratio=EIGVAL_CLIP_RATIO):
    """
    Numerically stable Fisher-avg Mahalanobis Lloyd fit.

    Args:
        K: (T, d) key vectors (float32)
        M_avg: (d, d) averaged Fisher metric (float32)
        bits: bit count for Lloyd quantizer
        eigval_clip_ratio: minimum eigenvalue as fraction of max (stability)

    Returns: dict {
        'K_mean', 'W_sqrt' (float32), 'W_inv_sqrt' (float32), 'centroids'
    }
    """
    K = K.astype(np.float32)
    M = M_avg.astype(np.float64)  # use double for stability in eigendecomp

    # Eigendecomposition with clipping
    eigvals, eigvecs = np.linalg.eigh(M)
    max_eig = max(eigvals.max(), 1e-12)
    min_eig_allowed = eigval_clip_ratio * max_eig
    eigvals_clipped = np.maximum(eigvals, min_eig_allowed)

    # Whitening matrices (double precision then cast to float32)
    sqrt_eig = np.sqrt(eigvals_clipped)
    W_sqrt = (eigvecs * sqrt_eig) @ eigvecs.T     # M^{1/2}
    W_inv_sqrt = (eigvecs * (1.0 / sqrt_eig)) @ eigvecs.T  # M^{-1/2}
    W_sqrt = W_sqrt.astype(np.float32)
    W_inv_sqrt = W_inv_sqrt.astype(np.float32)

    # Center and whiten
    K_mean = K.mean(axis=0)
    K_c = K - K_mean
    K_white = K_c @ W_sqrt   # float32

    # Per-dim Lloyd fit
    d = K.shape[1]
    centroids = np.zeros((d, 2 ** bits), dtype=np.float32)
    for j in range(d):
        centroids[j] = lloyd_max_1d_fit(K_white[:, j], bits, n_iter=20).astype(np.float32)

    return {
        'K_mean': K_mean,
        'W_sqrt': W_sqrt,
        'W_inv_sqrt': W_inv_sqrt,
        'centroids': centroids,
        'bits': bits,
    }


# ----------------------------------------------------------------------
# Phase 4: Forward pass quantization hook
# ----------------------------------------------------------------------

class CascadeMahalanobisHook:
    """
    Numerically stable per-head Mahalanobis Lloyd hook.
    Uses float32 for whitening critical path.
    """
    def __init__(self, head_quantizers, n_kv, head_dim):
        # head_quantizers: list of dicts (one per kv_head)
        self.hq = head_quantizers
        self.n_kv = n_kv
        self.head_dim = head_dim

    def __call__(self, module, inputs, output):
        # output: (B, T, n_kv * head_dim) in bfloat16
        B, T, _ = output.shape
        x_bf = output.view(B, T, self.n_kv, self.head_dim)
        # Critical path in float32
        x_np = x_bf.float().cpu().numpy()   # (B, T, H, d)

        x_q = np.zeros_like(x_np)
        for hk in range(self.n_kv):
            q = self.hq[hk]
            data = x_np[:, :, hk, :]
            shape = data.shape
            K_flat = data.reshape(-1, self.head_dim).astype(np.float32)

            # Center
            K_c = K_flat - q['K_mean']
            # Whiten (float32)
            K_white = K_c @ q['W_sqrt']
            # Per-dim Lloyd apply
            K_white_q = np.zeros_like(K_white)
            c = q['centroids']
            for j in range(self.head_dim):
                boundaries = (c[j, :-1] + c[j, 1:]) / 2
                idx = np.searchsorted(boundaries, K_white[:, j])
                K_white_q[:, j] = c[j, idx]
            # De-whiten (float32)
            K_dq = K_white_q @ q['W_inv_sqrt']
            # Un-center
            K_recon = K_dq + q['K_mean']
            x_q[:, :, hk, :] = K_recon.reshape(shape)

        # Cast back to original dtype
        result = torch.from_numpy(x_q).to(output.device).to(output.dtype)
        return result.view(B, T, self.n_kv * self.head_dim)


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------

def get_texts(tok):
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        texts = [t for t in ds['text'] if len(t.strip()) > 100]
        return '\n\n'.join(texts[:300]), '\n\n'.join(texts[300:600])
    except Exception:
        return " ".join(["Calib."] * 5000), " ".join(["Eval."] * 5000)


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
    print("=" * 70)
    print("Next-9: Cascade-Aware Fisher Mahalanobis Lloyd (Gap 2 Resolution)")
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

    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_q
    print(f"  n_layers={n_layers}, n_kv={n_kv}, n_q={n_q}, head_dim={head_dim}", flush=True)

    calib_text, eval_text = get_texts(tok)
    calib_enc = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB_TOKENS)
    calib_ids = calib_enc['input_ids'].to(DEVICE)
    eval_enc = tok(eval_text, return_tensors='pt', truncation=True, max_length=N_EVAL_TOKENS)
    eval_ids = eval_enc['input_ids'].to(DEVICE)
    print(f"  Calib tokens: {calib_ids.shape[1]}, Eval tokens: {eval_ids.shape[1]}", flush=True)

    # Baseline FP16
    print("\n[Baseline] FP16 PPL...", flush=True)
    ppl_fp16, _ = compute_ppl(model, eval_ids)
    print(f"  FP16 PPL: {ppl_fp16:.4f}", flush=True)

    # Phase 1a: Measure g_{l,h}
    print("\n[Phase 1a] Measuring cascade factor g_{l,h} via backward pass...", flush=True)
    t0 = time.time()
    try:
        g_table = measure_cascade_factors(model, calib_ids, n_layers, n_kv, n_q, head_dim)
        print(f"  Cascade measurement done in {time.time()-t0:.1f}s", flush=True)
        g_per_layer = g_table.sum(axis=1)  # sum over kv_heads
        print(f"  Top-10 layers by Σ_h g_{{l,h}}:", flush=True)
        top_layers = np.argsort(g_per_layer)[::-1][:10]
        for li in top_layers:
            print(f"    Layer {li}: g_total = {g_per_layer[li]:.4e}", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  Cascade measurement FAILED: {e}", flush=True)
        # Fallback: uniform g
        g_table = np.ones((n_layers, n_kv), dtype=np.float32)

    # Phase 1b: Fisher metrics + keys (forward, grad-free)
    print("\n[Phase 1b] Collecting K/Q/attention for Fisher metric...", flush=True)
    t0 = time.time()
    captured = compute_fisher_metrics_and_keys(model, calib_ids, n_layers, n_kv, n_q, head_dim)
    print(f"  Collection done in {time.time()-t0:.1f}s", flush=True)

    # Free model briefly? No - need it for PPL eval later.

    # Compute M_{l,h}^avg per (layer, kv_head)
    print("\n[Phase 2] Computing Fisher metrics M_{l,h}^avg...", flush=True)
    n_q_per_kv = n_q // n_kv
    M_table = {}    # (li, hk) -> (d, d)
    K_table = {}    # (li, hk) -> (T, d)
    trace_table = np.zeros((n_layers, n_kv), dtype=np.float32)

    for li in range(n_layers):
        data = captured.get(li, {})
        if not all(k in data for k in ['k', 'q', 'attn']):
            continue
        T_c = data['k'].shape[1]
        K_all = data['k'].reshape(T_c, n_kv, head_dim).astype(np.float32)
        Q_all = data['q'].reshape(T_c, n_q, head_dim).astype(np.float32)
        attn_all = data['attn'][0].astype(np.float32)   # (n_q, T, T)

        for hk in range(n_kv):
            K = K_all[:, hk, :]
            q_heads = list(range(hk * n_q_per_kv, (hk+1) * n_q_per_kv))
            Q = Q_all[:, q_heads, :].mean(axis=1)
            attn_mean = attn_all[q_heads, :, :].mean(axis=0)
            s_t = (attn_mean * (1.0 - attn_mean)).sum(axis=1)
            M = ((Q * s_t[:, None]).T @ Q) / max(T_c, 1)
            M = M + 1e-6 * np.eye(head_dim, dtype=np.float32)

            M_table[(li, hk)] = M
            K_table[(li, hk)] = K
            trace_table[li, hk] = float(np.trace(M))

    print(f"  Fisher metrics computed for {len(M_table)} heads", flush=True)

    # Phase 3: Global budget allocation (Theorem B)
    print("\n[Phase 3] Global budget allocation (Theorem B)...", flush=True)
    total_budget = int(n_layers * n_kv * AVG_BITS)
    print(f"  Total budget: {total_budget} bits ({AVG_BITS} avg per head)", flush=True)

    # Normalize g per layer to prevent scale blowup
    g_norm = np.zeros_like(g_table)
    for li in range(n_layers):
        max_g = g_table[li].max()
        if max_g > 0:
            g_norm[li] = g_table[li] / max_g
        else:
            g_norm[li] = 1.0

    # Importance = g_norm × trace(M)
    # Flatten (layer, kv_head) to linear index
    importance_flat = []
    index_map = []
    for li in range(n_layers):
        for hk in range(n_kv):
            imp = float(g_norm[li, hk]) * float(trace_table[li, hk])
            importance_flat.append(imp)
            index_map.append((li, hk))

    importance_flat = np.array(importance_flat)
    bits_flat = water_filling_global(importance_flat, total_budget, b_floor=B_FLOOR, b_max=B_MAX)

    # Unflatten
    bits_table = np.zeros((n_layers, n_kv), dtype=int)
    for k, (li, hk) in enumerate(index_map):
        bits_table[li, hk] = bits_flat[k]

    # Stats on bit allocation
    avg_bits_actual = bits_table.mean()
    max_bits = bits_table.max()
    min_bits = bits_table.min()
    print(f"  Allocated avg_bits={avg_bits_actual:.3f}, min={min_bits}, max={max_bits}", flush=True)

    # Top allocated heads
    flat_sorted = sorted(
        [(li, hk, bits_table[li, hk]) for li in range(n_layers) for hk in range(n_kv)],
        key=lambda x: -x[2]
    )
    print(f"  Top-10 (layer, kv_head, bits):", flush=True)
    for li, hk, b in flat_sorted[:10]:
        print(f"    L{li}/H{hk}: {b} bits (g_norm={g_norm[li,hk]:.3f}, tr(M)={trace_table[li,hk]:.4f})", flush=True)

    # Phase 4: Per-head Mahalanobis Lloyd fit
    print("\n[Phase 4] Per-head Mahalanobis Lloyd fit...", flush=True)
    t0 = time.time()
    per_layer_head_data = {li: [None] * n_kv for li in range(n_layers)}
    fit_failures = 0
    for (li, hk), M in M_table.items():
        K = K_table[(li, hk)]
        b = int(bits_table[li, hk])
        try:
            hd = fit_mahalanobis_lloyd_stable(K, M, b)
            per_layer_head_data[li][hk] = hd
        except Exception as e:
            print(f"    L{li}/H{hk} fit failed: {e}", flush=True)
            fit_failures += 1

    print(f"  Fit done in {time.time()-t0:.1f}s ({fit_failures} failures)", flush=True)

    # Check coverage
    missing = sum(1 for li in range(n_layers) for hd in per_layer_head_data[li] if hd is None)
    if missing > 0:
        print(f"  WARNING: {missing} heads have no quantizer (will use passthrough)", flush=True)

    # Phase 5: Install hooks and measure PPL
    print("\n[Phase 5] Installing hooks and measuring PPL...", flush=True)
    handles = []
    for li in range(n_layers):
        head_data_list = per_layer_head_data[li]
        # Replace None with passthrough (a quantizer that returns input)
        for hk in range(n_kv):
            if head_data_list[hk] is None:
                # Identity passthrough
                head_data_list[hk] = {
                    'K_mean': np.zeros(head_dim, dtype=np.float32),
                    'W_sqrt': np.eye(head_dim, dtype=np.float32),
                    'W_inv_sqrt': np.eye(head_dim, dtype=np.float32),
                    'centroids': np.stack([np.linspace(-3, 3, 2**B_FLOOR, dtype=np.float32) for _ in range(head_dim)]),
                    'bits': B_FLOOR,
                }
        hook = CascadeMahalanobisHook(head_data_list, n_kv, head_dim)
        h = model.model.layers[li].self_attn.k_proj.register_forward_hook(hook)
        handles.append(h)

    t0 = time.time()
    try:
        ppl_next9, loss_next9 = compute_ppl(model, eval_ids)
        print(f"  Next-9 PPL: {ppl_next9:.4f} ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        ppl_next9, loss_next9 = float('inf'), float('inf')
        print(f"  PPL FAILED: {e}", flush=True)
    finally:
        for h in handles:
            h.remove()

    # Success criteria
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA")
    print("=" * 70)
    ref = {
        'FP16 baseline': ppl_fp16,
        'Next-4 D (Mahalanobis catastrophe)': 982.25,
        'Next-4 C (L² Lloyd all-2b)': 9.12,
        'Next-4 B (Uniform 2b)': 7.90,
        'Next-4 E (L² outlier preservation)': 6.95,
        'v3 Uniform 2b reference': 6.46,
        'v3 WF floor=2 (best known)': 5.82,
    }
    print(f"  Next-9 PPL: {ppl_next9:.4f}")
    print()
    for name, val in ref.items():
        if ppl_next9 < val:
            marker = "✅ BEAT"
        elif ppl_next9 < 1.05 * val:
            marker = "≈ MATCH"
        else:
            marker = "❌ above"
        print(f"    {marker} {name}: {val:.4f}")

    # Save results
    results = {
        'model': MODEL_NAME,
        'avg_bits_target': AVG_BITS,
        'avg_bits_actual': float(avg_bits_actual),
        'b_floor': B_FLOOR,
        'b_max': B_MAX,
        'eigval_clip_ratio': EIGVAL_CLIP_RATIO,
        'n_calib_tokens': N_CALIB_TOKENS,
        'n_eval_tokens': int(eval_ids.shape[1]),
        'ppl_fp16': ppl_fp16,
        'ppl_next9_cascade_mahalanobis': ppl_next9,
        'loss_next9': loss_next9,
        'reference_ppls': ref,
        'g_table': g_table.tolist(),
        'g_per_layer': g_per_layer.tolist(),
        'bits_table': bits_table.tolist(),
        'trace_table': trace_table.tolist(),
        'fit_failures': fit_failures,
        'runtime_sec': time.time() - t_start,
    }
    out_file = OUT_DIR / 'exp_next9_cascade_mahalanobis_v2.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")
    print(f"Total runtime: {results['runtime_sec']:.1f}s ({results['runtime_sec']/60:.1f}m)")


if __name__ == '__main__':
    main()
