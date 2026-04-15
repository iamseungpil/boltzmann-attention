#!/usr/bin/env python3
"""
V2c: Full-Model Per-Dim WF at 2-bit Avg on Mistral
====================================================

Question: Does the per-dim WF bit allocation (that surgically fixed L1 H6 in
exp_v2 Test 4) generalize when applied to ALL (layer, kv_head) pairs?

Protocol:
  1. Calibrate: capture K for every (L,H), compute per-head PCA basis V, eigvals
  2. WF-allocate bits per dim for each head at total_budget = 2 * head_dim
  3. Fit Lloyd-Max per dim with allocated bits
  4. Forward with hook: quantize K on every k_proj output (all heads)
  5. Compare PPL:
       [A] FP16 baseline
       [B] Uniform PCA + 2-bit Lloyd (per-head PCA, all dims 2-bit) — v3 baseline
       [C] Per-head PCA + per-dim WF 2-bit Lloyd — the new method

Expected result:
  If WF generalizes: [C] PPL ≪ [B] PPL on Mistral (where [B] catastrophically fails)

GPU: 1 (exp_v2b uses GPU 0 in parallel)
"""
import json, time, gc, os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

MODEL_NAME = 'mistralai/Mistral-7B-v0.3'
DTYPE = torch.bfloat16
N_CALIB_TOKENS = 2048
N_EVAL_TOKENS = 2048
TARGET_AVG_BITS = 2.0
B_FLOOR = 1
B_MAX = 8

OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_texts(tok):
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in ds['text'] if len(t.strip()) > 100]
    return '\n\n'.join(texts[:300]), '\n\n'.join(texts[300:600])


def lloyd_1d(col, bits, n_iter=20):
    n_levels = 2 ** bits
    if n_levels <= 1:
        return np.array([float(col.mean())], dtype=np.float32)
    pcts = np.linspace(0, 100, n_levels + 2)[1:-1]
    centroids = np.sort(np.percentile(col, pcts)).astype(np.float64)
    for _ in range(n_iter):
        b = (centroids[:-1] + centroids[1:]) / 2
        idx = np.searchsorted(b, col)
        new_c = centroids.copy()
        for k in range(n_levels):
            m = idx == k
            if m.sum() > 0:
                new_c[k] = col[m].mean()
        if np.max(np.abs(new_c - centroids)) < 1e-6:
            break
        centroids = new_c
    return centroids.astype(np.float32)


def wf_alloc(sigma2, total_budget, b_floor=1, b_max=8):
    n = len(sigma2)
    s = np.maximum(sigma2, 1e-12)
    bits = np.zeros(n, dtype=int)
    spent = 0
    while spent < total_budget:
        best_g = -np.inf
        best = None
        for j in range(n):
            if bits[j] == 0:
                if spent + b_floor > total_budget:
                    continue
                g = s[j] * (1.0 - 4.0 ** (-b_floor)) / b_floor
                if g > best_g:
                    best_g, best = g, ('act', j)
            elif bits[j] < b_max:
                g = s[j] * (4.0 ** (-bits[j]) - 4.0 ** (-(bits[j] + 1)))
                if g > best_g:
                    best_g, best = g, ('add', j)
        if best is None:
            break
        op, j = best
        if op == 'act':
            bits[j] = b_floor
            spent += b_floor
        else:
            bits[j] += 1
            spent += 1
    return bits


def compute_ppl(model, input_ids):
    with torch.no_grad():
        out = model(input_ids, use_cache=False)
        logits = out.logits[:, :-1, :].contiguous()
        targets = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1), reduction='mean'
        )
        return float(torch.exp(loss).item()), float(loss.item())


class FullModelHook:
    """
    Per-head PCA + per-dim variable-bit Lloyd quantization.
    Installed on k_proj output for each layer.
    """
    def __init__(self, n_kv, head_dim, V_list, mean_list, centroids_list):
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.V_list = V_list            # list of (head_dim, head_dim) float32, length n_kv
        self.mean_list = mean_list      # list of (head_dim,) float32, length n_kv
        self.centroids_list = centroids_list  # list of list(head_dim) of 1d arrays, length n_kv

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x = output.view(B, T, self.n_kv, self.head_dim).float().cpu().numpy()
        for hk in range(self.n_kv):
            V = self.V_list[hk]
            m = self.mean_list[hk]
            cents = self.centroids_list[hk]
            data = x[:, :, hk, :].reshape(-1, self.head_dim)
            K_c = data - m
            K_pca = K_c @ V
            K_q = np.zeros_like(K_pca)
            for j in range(self.head_dim):
                cj = cents[j]
                if cj is None or len(cj) == 1:
                    K_q[:, j] = cj[0] if cj is not None else 0.0
                else:
                    b = (cj[:-1] + cj[1:]) / 2
                    idx = np.searchsorted(b, K_pca[:, j])
                    K_q[:, j] = cj[idx]
            K_rec = K_q @ V.T + m
            x[:, :, hk, :] = K_rec.reshape(data.shape[0], self.head_dim).reshape(B, T, self.head_dim)
        return torch.from_numpy(x).to(output.device).to(output.dtype).view(B, T, self.n_kv * self.head_dim)


def calibrate_all_heads(model, calib_ids, n_layers, n_kv, head_dim):
    """Capture K per (L,H), compute PCA basis + eigvals."""
    per_layer_K = {}

    def mk_hook(li):
        def h(m, i, o):
            per_layer_K[li] = o.detach().cpu().float().numpy()
        return h

    handles = []
    for li in range(n_layers):
        handles.append(model.model.layers[li].self_attn.k_proj.register_forward_hook(mk_hook(li)))
    with torch.no_grad():
        _ = model(calib_ids, use_cache=False)
    for h in handles:
        h.remove()

    basis = {}  # li -> list of (V, mean, eigvals_desc) per head
    for li in range(n_layers):
        K_all = per_layer_K[li].reshape(-1, n_kv, head_dim).astype(np.float32)
        per_head = []
        for hk in range(n_kv):
            K = K_all[:, hk, :]
            mean = K.mean(axis=0)
            Kc = K - mean
            cov = (Kc.T @ Kc) / max(K.shape[0] - 1, 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            V = eigvecs[:, order].astype(np.float32)
            ev = eigvals[order]
            per_head.append({
                'V': V, 'mean': mean.astype(np.float32),
                'eigvals': ev,
                'K_pca': Kc @ V,  # for Lloyd fitting
            })
        basis[li] = per_head
    return basis


def fit_centroids(basis, n_layers, n_kv, head_dim, scheme):
    """
    scheme: 'uniform2' or 'wf2'
    Returns: dict li -> list of list of centroid arrays, and bits_used stats.
    """
    out = {}
    all_bits = []
    for li in range(n_layers):
        per_head = []
        for hk in range(n_kv):
            bh = basis[li][hk]
            ev = bh['eigvals']
            K_pca = bh['K_pca']
            if scheme == 'uniform2':
                bits = np.full(head_dim, 2, dtype=int)
            elif scheme == 'wf2':
                bits = wf_alloc(ev, total_budget=int(TARGET_AVG_BITS * head_dim),
                                b_floor=B_FLOOR, b_max=B_MAX)
            else:
                raise ValueError(scheme)
            all_bits.append(bits.mean())
            cents = []
            for j in range(head_dim):
                b = int(bits[j])
                if b == 0:
                    cents.append(np.array([0.0], dtype=np.float32))
                else:
                    cents.append(lloyd_1d(K_pca[:, j], b, n_iter=15))
            per_head.append(cents)
        out[li] = per_head
    return out, float(np.mean(all_bits))


def install_full_hooks(model, basis, centroids_by_layer, n_layers, n_kv, head_dim):
    handles = []
    for li in range(n_layers):
        V_list = [basis[li][hk]['V'] for hk in range(n_kv)]
        mean_list = [basis[li][hk]['mean'] for hk in range(n_kv)]
        cents_list = centroids_by_layer[li]
        hook = FullModelHook(n_kv, head_dim, V_list, mean_list, cents_list)
        handles.append(model.model.layers[li].self_attn.k_proj.register_forward_hook(hook))
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def main():
    print("="*70)
    print("V2c: Full-Model Per-Dim WF at 2-bit on Mistral")
    print("="*70, flush=True)
    t_start = time.time()

    print("\nLoading Mistral-7B...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    print(f"  n_layers={n_layers}, n_kv={n_kv}, head_dim={head_dim}", flush=True)
    print(f"  Loaded in {time.time()-t_start:.1f}s", flush=True)

    calib_text, eval_text = get_texts(tok)
    calib_ids = tok(calib_text, return_tensors='pt', truncation=True,
                    max_length=N_CALIB_TOKENS)['input_ids'].to('cuda:0')
    eval_ids = tok(eval_text, return_tensors='pt', truncation=True,
                   max_length=N_EVAL_TOKENS)['input_ids'].to('cuda:0')

    # === [A] FP16 baseline ===
    ppl_fp16, _ = compute_ppl(model, eval_ids)
    print(f"\n[A] FP16 baseline: PPL = {ppl_fp16:.4f}", flush=True)

    # === Calibrate all heads ===
    print("\nCalibrating all heads (PCA + eigvals)...", flush=True)
    t0 = time.time()
    basis = calibrate_all_heads(model, calib_ids, n_layers, n_kv, head_dim)
    print(f"  Calibration done in {time.time()-t0:.1f}s", flush=True)

    # Analyze spectra to understand WF signal
    avg_kappa = []
    for li in range(n_layers):
        for hk in range(n_kv):
            ev = basis[li][hk]['eigvals']
            k = float(ev[0] / max(ev[-1], 1e-12))
            avg_kappa.append(k)
    print(f"  κ(Σ_K): median={np.median(avg_kappa):.1e}, "
          f"max={np.max(avg_kappa):.1e}, "
          f">1e4: {sum(1 for k in avg_kappa if k > 1e4)} heads", flush=True)

    # === [B] Uniform 2-bit per-head PCA + Lloyd ===
    print("\n[B] Fitting Uniform 2-bit centroids...", flush=True)
    t0 = time.time()
    cents_uniform, avg_b = fit_centroids(basis, n_layers, n_kv, head_dim, 'uniform2')
    print(f"  Fit in {time.time()-t0:.1f}s, avg_bits = {avg_b:.3f}", flush=True)

    handles = install_full_hooks(model, basis, cents_uniform, n_layers, n_kv, head_dim)
    t0 = time.time()
    ppl_uniform, _ = compute_ppl(model, eval_ids)
    remove_hooks(handles)
    print(f"  [B] Uniform 2-bit PPL = {ppl_uniform:.4f} "
          f"(Δ = {ppl_uniform - ppl_fp16:+.4f}) [{time.time()-t0:.1f}s eval]", flush=True)
    del cents_uniform
    gc.collect()

    # === [C] Per-dim WF 2-bit ===
    print("\n[C] Fitting WF-allocated 2-bit centroids...", flush=True)
    t0 = time.time()
    cents_wf, avg_b_wf = fit_centroids(basis, n_layers, n_kv, head_dim, 'wf2')
    print(f"  Fit in {time.time()-t0:.1f}s, avg_bits = {avg_b_wf:.3f}", flush=True)

    handles = install_full_hooks(model, basis, cents_wf, n_layers, n_kv, head_dim)
    t0 = time.time()
    ppl_wf, _ = compute_ppl(model, eval_ids)
    remove_hooks(handles)
    print(f"  [C] WF-allocated 2-bit PPL = {ppl_wf:.4f} "
          f"(Δ = {ppl_wf - ppl_fp16:+.4f}) [{time.time()-t0:.1f}s eval]", flush=True)

    # === Verdict ===
    print("\n" + "="*70)
    print("VERDICT — Full-Model Per-Dim WF Generalization")
    print("="*70)
    print(f"  [A] FP16                        : {ppl_fp16:.4f}")
    print(f"  [B] Per-head PCA + Uniform 2b   : {ppl_uniform:.4f}  (Δ = {ppl_uniform-ppl_fp16:+.4f})")
    print(f"  [C] Per-head PCA + WF 2b        : {ppl_wf:.4f}  (Δ = {ppl_wf-ppl_fp16:+.4f})")
    print(f"  Improvement [B]→[C]: {ppl_uniform - ppl_wf:+.4f} "
          f"({(1 - ppl_wf/ppl_uniform)*100:+.1f}% PPL reduction)")
    if ppl_wf < ppl_uniform * 0.9:
        print(f"  → ✅ WF generalizes: ≥10% PPL improvement at same 2-bit budget")
    elif ppl_wf < ppl_uniform:
        print(f"  → ⚠️  WF helps but improvement <10%")
    else:
        print(f"  → ❌ WF does NOT generalize at full-model level")

    results = {
        'model': MODEL_NAME,
        'n_layers': n_layers,
        'n_kv': n_kv,
        'head_dim': head_dim,
        'target_avg_bits': TARGET_AVG_BITS,
        'ppl_fp16': ppl_fp16,
        'ppl_uniform_2bit': ppl_uniform,
        'delta_uniform': ppl_uniform - ppl_fp16,
        'ppl_wf_2bit': ppl_wf,
        'delta_wf': ppl_wf - ppl_fp16,
        'improvement_pct': float((1 - ppl_wf/ppl_uniform) * 100),
        'kappa_stats': {
            'median': float(np.median(avg_kappa)),
            'max': float(np.max(avg_kappa)),
            'n_over_1e4': int(sum(1 for k in avg_kappa if k > 1e4)),
        },
        'runtime_sec': time.time() - t_start,
    }
    out = OUT_DIR / 'exp_v2c_full_model_wf.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Total runtime: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f}m)")


if __name__ == '__main__':
    main()
