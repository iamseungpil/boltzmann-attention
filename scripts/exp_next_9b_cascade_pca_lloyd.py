#!/usr/bin/env python3
"""
Next-9b: Cascade-Aware PCA + L² Lloyd (Gap 2 simplified resolution)
=====================================================================

Simpler approach than Next-9:
  1. PCA rotation first (diagonalize key covariance)
  2. L² Lloyd per dim (no Mahalanobis whitening) — MSE-optimal scalar
  3. Global cascade-aware budget allocation via Theorem B
     importance[l,h] = g[l,h] × tr(M[l,h])
  4. Budget > n_heads × floor to allow WF variation

This isolates the "cascade-aware allocation" gain from the Mahalanobis
numerical issue. If this beats Next-4 Config E, the Theorem B allocation
logic is validated separately from the Mahalanobis question.

Key difference from Next-4 config E:
  - Config E: hand-picked layers 2-6 @ 3b, others 2b (empirical)
  - Next-9b: ALL heads allocated by WF with g × tr(M) importance (principled)
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
DEVICE = 'cuda:0'
DTYPE = torch.bfloat16
N_CALIB_TOKENS = 1024
N_EVAL_TOKENS = 2048

# Multiple average bit targets to test
AVG_BITS_TARGETS = [2.0, 2.156, 2.3, 2.5]  # match Next-4 E=2.156, F=2.3, etc.
B_FLOOR = 2
B_MAX = 6

OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Lloyd-Max 1D
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


# ----------------------------------------------------------------------
# Cascade factor measurement (Gap 1 Track A)
# ----------------------------------------------------------------------

def measure_cascade_factors(model, input_ids, n_layers, n_kv, n_q, head_dim):
    """Measure g_{l,h} via backward pass."""
    grad_captures = {}

    def make_fh(li_local):
        def fh(mod, inputs, output):
            if isinstance(output, tuple):
                attn_out = output[0]
            else:
                attn_out = output
            if attn_out.requires_grad:
                def bh(grad):
                    grad_captures[li_local] = grad.detach().cpu().float().numpy()
                attn_out.register_hook(bh)
        return fh

    handles = []
    for li in range(n_layers):
        handles.append(model.model.layers[li].self_attn.register_forward_hook(make_fh(li)))

    model.zero_grad()
    out = model(input_ids, use_cache=False)
    logits = out.logits
    targets = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        logits[:, :-1, :].contiguous().reshape(-1, logits.size(-1)).float(),
        targets.reshape(-1),
        reduction='mean'
    )
    loss.backward()

    for h in handles:
        h.remove()

    n_q_per_kv = n_q // n_kv
    g_table = np.zeros((n_layers, n_kv), dtype=np.float32)

    for li, grad in grad_captures.items():
        T = grad.shape[1]
        grad_reshaped = grad.reshape(1, T, n_q, head_dim)
        for hk in range(n_kv):
            q_slice = grad_reshaped[0, :, hk * n_q_per_kv : (hk+1) * n_q_per_kv, :]
            g_table[li, hk] = float(np.mean(np.sum(q_slice ** 2, axis=-1)))

    return g_table


# ----------------------------------------------------------------------
# Fisher metric + keys collection
# ----------------------------------------------------------------------

def collect_kqa(model, input_ids, n_layers):
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
# Water-filling allocation
# ----------------------------------------------------------------------

def water_filling_global(importance, total_budget, b_floor=2, b_max=6):
    n = len(importance)
    imp = np.array(importance, dtype=np.float64)
    imp = np.maximum(imp, 1e-12)

    bits = np.full(n, b_floor, dtype=int)
    spent = n * b_floor
    if spent > total_budget:
        return bits

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
# PCA-based L² Lloyd quantizer
# ----------------------------------------------------------------------

def fit_pca_l2_lloyd(K, bits):
    """
    Fit PCA + per-dim L² Lloyd in PCA basis.
    Returns {'K_mean', 'V', 'centroids'} where V is PCA eigenvectors (d, d).
    """
    K = K.astype(np.float32)
    K_mean = K.mean(axis=0)
    K_c = K - K_mean

    cov = (K_c.T @ K_c) / max(K.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    V = eigvecs[:, order]  # (d, d)

    K_pca = K_c @ V
    d = K.shape[1]
    centroids = np.zeros((d, 2 ** bits), dtype=np.float32)
    for j in range(d):
        centroids[j] = lloyd_max_1d_fit(K_pca[:, j], bits, n_iter=20).astype(np.float32)

    return {
        'K_mean': K_mean,
        'V': V.astype(np.float32),
        'centroids': centroids,
        'bits': bits,
    }


class PCAL2LloydHook:
    """PCA rotation + per-dim Lloyd + inverse rotation."""
    def __init__(self, head_quantizers, n_kv, head_dim):
        self.hq = head_quantizers
        self.n_kv = n_kv
        self.head_dim = head_dim

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x_bf = output.view(B, T, self.n_kv, self.head_dim)
        x_np = x_bf.float().cpu().numpy()
        x_q = np.zeros_like(x_np)

        for hk in range(self.n_kv):
            q = self.hq[hk]
            data = x_np[:, :, hk, :]
            shape = data.shape
            K_flat = data.reshape(-1, self.head_dim).astype(np.float32)

            # Center + PCA rotate
            K_c = K_flat - q['K_mean']
            K_pca = K_c @ q['V']

            # Per-dim Lloyd apply
            K_pca_q = np.zeros_like(K_pca)
            c = q['centroids']
            for j in range(self.head_dim):
                boundaries = (c[j, :-1] + c[j, 1:]) / 2
                idx = np.searchsorted(boundaries, K_pca[:, j])
                K_pca_q[:, j] = c[j, idx]

            # Inverse PCA + un-center
            K_recon = K_pca_q @ q['V'].T + q['K_mean']
            x_q[:, :, hk, :] = K_recon.reshape(shape)

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
    print("Next-9b: Cascade-Aware PCA + L² Lloyd (Gap 2 resolution, simplified)")
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

    # Baseline
    print("\n[Baseline] FP16 PPL...", flush=True)
    ppl_fp16, _ = compute_ppl(model, eval_ids)
    print(f"  FP16 PPL: {ppl_fp16:.4f}", flush=True)

    # Measure g
    print("\n[Phase 1a] Measuring cascade factor g_{l,h}...", flush=True)
    t0 = time.time()
    g_table = measure_cascade_factors(model, calib_ids, n_layers, n_kv, n_q, head_dim)
    print(f"  Measured in {time.time()-t0:.1f}s", flush=True)
    g_per_layer = g_table.sum(axis=1)
    top5_layers = np.argsort(g_per_layer)[::-1][:5]
    print(f"  Top 5 layers by Σ_h g: {top5_layers.tolist()} "
          f"(values: {[f'{g_per_layer[l]:.2e}' for l in top5_layers]})", flush=True)

    # Collect KQA for Fisher metric
    print("\n[Phase 1b] Collecting K/Q/attention...", flush=True)
    t0 = time.time()
    captured = collect_kqa(model, calib_ids, n_layers)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    # Compute M_{l,h}^avg and tr(M)
    print("\n[Phase 2] Computing Fisher metrics...", flush=True)
    n_q_per_kv = n_q // n_kv
    trace_table = np.zeros((n_layers, n_kv), dtype=np.float32)
    K_table = {}

    for li in range(n_layers):
        data = captured.get(li, {})
        if not all(k in data for k in ['k', 'q', 'attn']):
            continue
        T_c = data['k'].shape[1]
        K_all = data['k'].reshape(T_c, n_kv, head_dim).astype(np.float32)
        Q_all = data['q'].reshape(T_c, n_q, head_dim).astype(np.float32)
        attn_all = data['attn'][0].astype(np.float32)

        for hk in range(n_kv):
            K = K_all[:, hk, :]
            q_heads = list(range(hk * n_q_per_kv, (hk+1) * n_q_per_kv))
            Q = Q_all[:, q_heads, :].mean(axis=1)
            attn_mean = attn_all[q_heads, :, :].mean(axis=0)
            s_t = (attn_mean * (1.0 - attn_mean)).sum(axis=1)
            M = ((Q * s_t[:, None]).T @ Q) / max(T_c, 1)
            trace_table[li, hk] = float(np.trace(M))
            K_table[(li, hk)] = K

    print(f"  Fisher metrics computed for {len(K_table)} heads", flush=True)

    # Multiple budget tests
    results = {
        'model': MODEL_NAME,
        'ppl_fp16': ppl_fp16,
        'g_table': g_table.tolist(),
        'g_per_layer': g_per_layer.tolist(),
        'trace_table': trace_table.tolist(),
        'configs': {},
    }

    for avg_bits in AVG_BITS_TARGETS:
        cfg_name = f'cascade_pca_l2_avg{avg_bits}'
        print(f"\n[Config {cfg_name}] avg_bits={avg_bits}", flush=True)

        # Global budget allocation via Theorem B
        total_budget = int(round(n_layers * n_kv * avg_bits))
        print(f"  Total budget: {total_budget} bits", flush=True)

        # Normalize g per layer
        g_norm = np.zeros_like(g_table)
        for li in range(n_layers):
            max_g = g_table[li].max()
            g_norm[li] = g_table[li] / max_g if max_g > 0 else 1.0

        importance_flat = []
        index_map = []
        for li in range(n_layers):
            for hk in range(n_kv):
                imp = float(g_norm[li, hk]) * float(trace_table[li, hk])
                importance_flat.append(imp)
                index_map.append((li, hk))

        bits_flat = water_filling_global(
            np.array(importance_flat), total_budget, b_floor=B_FLOOR, b_max=B_MAX
        )

        bits_table = np.zeros((n_layers, n_kv), dtype=int)
        for k, (li, hk) in enumerate(index_map):
            bits_table[li, hk] = bits_flat[k]

        actual_avg = bits_table.mean()
        unique_bits, counts = np.unique(bits_table, return_counts=True)
        dist = dict(zip(unique_bits.tolist(), counts.tolist()))
        print(f"  Bit distribution: {dist} (actual avg: {actual_avg:.3f})", flush=True)

        # Top layers by total bits
        layer_bits_sum = bits_table.sum(axis=1)
        top5_bit_layers = np.argsort(layer_bits_sum)[::-1][:5]
        print(f"  Top 5 layers by total bits: {top5_bit_layers.tolist()} "
              f"(bits: {layer_bits_sum[top5_bit_layers].tolist()})", flush=True)

        # Fit PCA+L² Lloyd per head
        print(f"  Fitting quantizers...", flush=True)
        t_fit = time.time()
        per_layer_head_data = {li: [None] * n_kv for li in range(n_layers)}
        for (li, hk), K in K_table.items():
            b = int(bits_table[li, hk])
            try:
                hd = fit_pca_l2_lloyd(K, b)
                per_layer_head_data[li][hk] = hd
            except Exception as e:
                print(f"    L{li}/H{hk} fit failed: {e}", flush=True)
        print(f"  Fit done in {time.time()-t_fit:.1f}s", flush=True)

        # Fill missing with passthrough
        for li in range(n_layers):
            for hk in range(n_kv):
                if per_layer_head_data[li][hk] is None:
                    per_layer_head_data[li][hk] = {
                        'K_mean': np.zeros(head_dim, dtype=np.float32),
                        'V': np.eye(head_dim, dtype=np.float32),
                        'centroids': np.stack([
                            np.linspace(-3, 3, 2**B_FLOOR, dtype=np.float32)
                            for _ in range(head_dim)
                        ]),
                        'bits': B_FLOOR,
                    }

        # Install hooks and measure PPL
        handles = []
        for li in range(n_layers):
            hook = PCAL2LloydHook(per_layer_head_data[li], n_kv, head_dim)
            h = model.model.layers[li].self_attn.k_proj.register_forward_hook(hook)
            handles.append(h)

        t_ppl = time.time()
        try:
            ppl, loss = compute_ppl(model, eval_ids)
        except Exception as e:
            import traceback
            traceback.print_exc()
            ppl, loss = float('inf'), float('inf')
        finally:
            for h in handles:
                h.remove()

        delta = (ppl - ppl_fp16) / ppl_fp16 * 100
        print(f"  PPL: {ppl:.4f} (Δ vs FP16: {delta:+.2f}%) [eval: {time.time()-t_ppl:.1f}s]", flush=True)

        results['configs'][cfg_name] = {
            'avg_bits_target': avg_bits,
            'avg_bits_actual': float(actual_avg),
            'total_budget': total_budget,
            'ppl': ppl,
            'loss': loss,
            'delta_vs_fp16_pct': delta,
            'bit_distribution': {int(k): int(v) for k, v in dist.items()},
            'bits_table': bits_table.tolist(),
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — Cascade-aware PCA + L² Lloyd vs Next-4 reference")
    print("=" * 70)
    print(f"  FP16 baseline: {ppl_fp16:.4f}")
    print()
    print(f"{'Config':<30} | {'avg_bits':>10} | {'PPL':>10} | {'Δ vs FP16':>12}")
    print('-' * 75)

    # Reference Next-4 configs
    next4_ref = {
        'Next-4 B (Uniform 2b)': (2.000, 7.90),
        'Next-4 C (L² Lloyd all-2b)': (2.000, 9.12),
        'Next-4 E (L² outlier 3b @ 2-6)': (2.156, 6.95),
        'Next-4 F (L² outlier 4b @ 2-6)': (2.312, 7.09),
        'Next-4 G (L² layer2 only 4b)': (2.062, 8.53),
    }
    for name, (ab, ppl) in next4_ref.items():
        print(f"  {name:<28} | {ab:>10.3f} | {ppl:>10.4f} | {(ppl-ppl_fp16)/ppl_fp16*100:>+11.2f}%")

    for cfg_name, cfg_data in results['configs'].items():
        if 'ppl' in cfg_data:
            print(f"  {cfg_name:<28} | {cfg_data['avg_bits_actual']:>10.3f} | "
                  f"{cfg_data['ppl']:>10.4f} | {cfg_data['delta_vs_fp16_pct']:>+11.2f}%")

    # Best result comparison
    print()
    best_cfg = None
    best_ppl = float('inf')
    for cfg_name, cfg_data in results['configs'].items():
        if 'ppl' in cfg_data and cfg_data['ppl'] < best_ppl:
            best_ppl = cfg_data['ppl']
            best_cfg = cfg_name

    if best_cfg:
        print(f"Best Next-9b: {best_cfg} → {best_ppl:.4f}")
        print(f"  vs Next-4 E (6.95): {((best_ppl - 6.95) / 6.95 * 100):+.2f}%")
        print(f"  vs v3 Uniform 2b (6.46): {((best_ppl - 6.46) / 6.46 * 100):+.2f}%")

    results['runtime_sec'] = time.time() - t_start
    out_file = OUT_DIR / 'exp_next9b_cascade_pca_lloyd.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")
    print(f"Total runtime: {results['runtime_sec']:.1f}s ({results['runtime_sec']/60:.1f}m)")


if __name__ == '__main__':
    main()
