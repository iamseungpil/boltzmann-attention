#!/usr/bin/env python3
"""
Next-1: Full-model Fisher-avg Mahalanobis Lloyd PPL on Mistral-7B
==================================================================

Apply Fisher-weighted Mahalanobis Lloyd-Max quantization to ALL layers
simultaneously and measure the resulting PPL.

Compare:
  (A) FP16 baseline
  (B) All layers: Uniform 2-bit
  (C) All layers: L² Lloyd 2-bit
  (D) All layers: Fisher-avg Mahalanobis Lloyd 2-bit  ← our method

Hypothesis: (D) beats (C) in PPL because it optimizes in the correct metric.

Fit phase:
  1. Single calibration forward with hooks on (k_proj, q_proj, attn_output)
  2. For each (layer, kv_head):
     - Compute averaged Fisher M = (1/T) Σ_t s_t q_t q_t^T  where s_t=Σ_j p(1-p)
     - Whiten K with M^{1/2} (eigendecomposition)
     - Fit Lloyd-Max per dim in whitened space
     - Save {K_mean, W_sqrt, W_inv_sqrt, centroids}

Eval phase:
  - Install hooks on all k_proj modules simultaneously
  - Hook applies: center → whiten → Lloyd dequantize → de-whiten → uncenter
  - Run forward on held-out text, compute PPL
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
BITS = 2

OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Lloyd-Max 1D (scalar quantizer)
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
# Quantizer hooks
# ----------------------------------------------------------------------

class UniformQuantHook:
    """Fit-less uniform quantizer per-dim (clip to 3σ)."""
    def __init__(self, n_kv, head_dim, bits=2):
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.n_levels = 2 ** bits
        # Per-head per-dim std is computed from the first batch
        self.stats = {}

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x = output.view(B, T, self.n_kv, self.head_dim)
        # Compute clip per (head, dim)
        std = x.std(dim=1, keepdim=True) + 1e-8  # (B, 1, H, d)
        clip = 3.0 * std
        x_clipped = torch.clip(x, -clip, clip)
        delta = 2.0 * clip / self.n_levels
        idx = torch.floor((x_clipped + clip) / delta).long()
        idx = torch.clip(idx, 0, self.n_levels - 1)
        recon = -clip + (idx.float() + 0.5) * delta
        return recon.view(B, T, self.n_kv * self.head_dim).to(output.dtype)


class L2LloydHook:
    """L² Lloyd-Max quantizer with pre-fitted centroids per (head, dim).
    FIX (2026-04-07): Apply K_mean centering matching the fit phase."""
    def __init__(self, head_dicts, n_kv, head_dim):
        # head_dicts: list of n_kv dicts with keys {'K_mean': (d,), 'centroids': (d, n_levels)}
        self.head_dicts = head_dicts
        self.n_kv = n_kv
        self.head_dim = head_dim

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x = output.view(B, T, self.n_kv, self.head_dim)
        x_np = x.float().cpu().numpy()
        x_q = np.zeros_like(x_np)
        for hk in range(self.n_kv):
            d = self.head_dicts[hk]
            K_mean = d['K_mean']
            c = d['centroids']  # (head_dim, n_levels)
            data = x_np[:, :, hk, :]  # (B, T, d)
            shape = data.shape
            data_flat = data.reshape(-1, self.head_dim)
            # Center (fit was on centered data)
            data_c = data_flat - K_mean
            data_q = np.zeros_like(data_c)
            for j in range(self.head_dim):
                boundaries = (c[j, :-1] + c[j, 1:]) / 2
                idx = np.searchsorted(boundaries, data_c[:, j])
                data_q[:, j] = c[j, idx]
            # Un-center
            data_q += K_mean
            x_q[:, :, hk, :] = data_q.reshape(shape)
        return torch.from_numpy(x_q).view(B, T, self.n_kv * self.head_dim).to(output.device).to(output.dtype)


class MahalanobisLloydHook:
    """Fisher-weighted Mahalanobis Lloyd quantizer.
    Per-head: center, whiten with M^{1/2}, Lloyd in whitened space, de-whiten.
    """
    def __init__(self, head_quantizers, n_kv, head_dim):
        # head_quantizers: list of n_kv dicts with keys:
        #   'K_mean': (d,), 'W_sqrt': (d, d), 'W_inv_sqrt': (d, d),
        #   'centroids': (d, n_levels)
        self.hq = head_quantizers
        self.n_kv = n_kv
        self.head_dim = head_dim

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x = output.view(B, T, self.n_kv, self.head_dim)
        x_np = x.float().cpu().numpy()
        x_q = np.zeros_like(x_np)
        for hk in range(self.n_kv):
            q = self.hq[hk]
            data = x_np[:, :, hk, :]   # (B, T, d)
            shape = data.shape
            K_flat = data.reshape(-1, self.head_dim)
            # Center
            K_c = K_flat - q['K_mean']
            # Whiten
            K_w = K_c @ q['W_sqrt']
            # Lloyd per dim
            K_wq = np.zeros_like(K_w)
            c = q['centroids']
            for j in range(self.head_dim):
                boundaries = (c[j, :-1] + c[j, 1:]) / 2
                idx = np.searchsorted(boundaries, K_w[:, j])
                K_wq[:, j] = c[j, idx]
            # De-whiten + uncenter
            K_dq = K_wq @ q['W_inv_sqrt'] + q['K_mean']
            x_q[:, :, hk, :] = K_dq.reshape(shape)
        return torch.from_numpy(x_q).view(B, T, self.n_kv * self.head_dim).to(output.device).to(output.dtype)


# ----------------------------------------------------------------------
# Calibration & fitting
# ----------------------------------------------------------------------

def get_texts(tok):
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        texts = [t for t in ds['text'] if len(t.strip()) > 100]
        calib = '\n\n'.join(texts[:300])
        eval_ = '\n\n'.join(texts[300:600])
    except Exception:
        calib = " ".join(["Calibration."] * 5000)
        eval_ = " ".join(["Evaluation."] * 5000)
    return calib, eval_


def fit_all_quantizers(model, tok, calib_text, n_layers, n_kv, n_q, head_dim, bits=BITS):
    """Single calibration pass, fit L² Lloyd + Mahalanobis Lloyd per (layer, head)."""
    print(f"  Calibration forward pass (T={N_CALIB_TOKENS})...", flush=True)
    enc = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB_TOKENS)
    input_ids = enc['input_ids'].to(DEVICE)
    T = input_ids.shape[1]

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
        attn_mod = model.model.layers[li].self_attn
        handles.append(attn_mod.k_proj.register_forward_hook(kh(li)))
        handles.append(attn_mod.q_proj.register_forward_hook(qh(li)))
        handles.append(attn_mod.register_forward_hook(ah(li)))

    t0 = time.time()
    with torch.no_grad():
        _ = model(input_ids, output_attentions=True, use_cache=False)
    print(f"  Calib forward done in {time.time()-t0:.1f}s", flush=True)
    for h in handles:
        h.remove()

    n_q_per_kv = n_q // n_kv
    all_l2 = {}
    all_mahal = {}

    print(f"  Fitting per-(layer, kv_head) quantizers...", flush=True)
    t0 = time.time()
    for li in range(n_layers):
        data = captured.get(li)
        if not data or any(k not in data for k in ['k', 'q', 'attn']):
            print(f"    layer {li}: missing data, skip")
            continue

        K_all = data['k'].reshape(T, n_kv, head_dim)
        Q_all = data['q'].reshape(T, n_q, head_dim)
        attn_all = data['attn'][0]  # (n_q, T, T)

        l2_list = []
        mahal_list = []
        for hk in range(n_kv):
            K = K_all[:, hk, :].astype(np.float32)
            K_mean = K.mean(axis=0, keepdims=False)
            K_centered = K - K_mean

            # === L² Lloyd (per-dim) ===
            centroids_l2 = np.zeros((head_dim, 2**bits))
            for j in range(head_dim):
                centroids_l2[j] = lloyd_max_1d_fit(K_centered[:, j], bits, n_iter=20)
            l2_list.append({'K_mean': K_mean, 'centroids': centroids_l2})

            # === Fisher-avg Mahalanobis Lloyd ===
            q_heads = list(range(hk * n_q_per_kv, (hk+1) * n_q_per_kv))
            Q = Q_all[:, q_heads, :].mean(axis=1).astype(np.float32)
            attn = attn_all[q_heads, :, :].mean(axis=0).astype(np.float32)
            s_t = (attn * (1.0 - attn)).sum(axis=1)   # (T,)
            M = ((Q * s_t[:, None]).T @ Q) / max(T, 1) + 1e-6 * np.eye(head_dim, dtype=np.float32)

            eigvals, eigvecs = np.linalg.eigh(M)
            eigvals = np.clip(eigvals, 1e-8, None)
            sqrt_eig = np.sqrt(eigvals)
            W_sqrt = (eigvecs * sqrt_eig) @ eigvecs.T
            W_inv_sqrt = (eigvecs * (1.0 / sqrt_eig)) @ eigvecs.T

            K_white = K_centered @ W_sqrt
            centroids_mh = np.zeros((head_dim, 2**bits))
            for j in range(head_dim):
                centroids_mh[j] = lloyd_max_1d_fit(K_white[:, j], bits, n_iter=20)

            mahal_list.append({
                'K_mean': K_mean,
                'W_sqrt': W_sqrt.astype(np.float32),
                'W_inv_sqrt': W_inv_sqrt.astype(np.float32),
                'centroids': centroids_mh.astype(np.float32),
            })

        all_l2[li] = l2_list
        all_mahal[li] = mahal_list

    print(f"  Fitting done in {time.time()-t0:.1f}s", flush=True)

    # Free captured tensors
    del captured
    gc.collect()

    return all_l2, all_mahal


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


def install_hooks(model, hook_class, args_per_layer, n_layers):
    """Install the same hook type across all layers with per-layer args."""
    handles = []
    for li in range(n_layers):
        hook_instance = hook_class(*args_per_layer[li])
        h = model.model.layers[li].self_attn.k_proj.register_forward_hook(hook_instance)
        handles.append(h)
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def main():
    print("=" * 60)
    print("Exp Next-1: Full Fisher-avg Mahalanobis Lloyd PPL")
    print("=" * 60, flush=True)
    t_start = time.time()

    print("Loading model...", flush=True)
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
    eval_enc = tok(eval_text, return_tensors='pt', truncation=True, max_length=N_EVAL_TOKENS)
    eval_ids = eval_enc['input_ids'].to(DEVICE)
    T_eval = eval_ids.shape[1]
    print(f"  Eval tokens: {T_eval}", flush=True)

    # (A) Baseline FP16
    print("\n[A] Baseline FP16 PPL...", flush=True)
    ppl_fp16, loss_fp16 = compute_ppl(model, eval_ids)
    print(f"  FP16 PPL: {ppl_fp16:.4f}", flush=True)

    # Fit quantizers
    print("\n[Fit] Fitting L² Lloyd + Mahalanobis Lloyd across all layers...", flush=True)
    all_l2, all_mahal = fit_all_quantizers(
        model, tok, calib_text, n_layers, n_kv, n_q, head_dim, bits=BITS
    )

    # (B) Uniform 2-bit (no fitting needed)
    print("\n[B] Uniform 2-bit PPL...", flush=True)
    uniform_args = [(n_kv, head_dim, BITS)] * n_layers
    handles = install_hooks(model, UniformQuantHook, uniform_args, n_layers)
    t0 = time.time()
    ppl_uniform, loss_uniform = compute_ppl(model, eval_ids)
    print(f"  Uniform 2b PPL: {ppl_uniform:.4f} ({time.time()-t0:.1f}s)", flush=True)
    remove_hooks(handles)

    # (C) L² Lloyd 2-bit
    print("\n[C] L² Lloyd 2-bit PPL...", flush=True)
    l2_args = [([h['centroids'] for h in all_l2[li]], n_kv, head_dim) for li in range(n_layers)]
    handles = install_hooks(model, L2LloydHook, l2_args, n_layers)
    t0 = time.time()
    ppl_l2, loss_l2 = compute_ppl(model, eval_ids)
    print(f"  L² Lloyd 2b PPL: {ppl_l2:.4f} ({time.time()-t0:.1f}s)", flush=True)
    remove_hooks(handles)

    # (D) Fisher-avg Mahalanobis Lloyd 2-bit
    print("\n[D] Fisher-avg Mahalanobis Lloyd 2-bit PPL...", flush=True)
    mh_args = [(all_mahal[li], n_kv, head_dim) for li in range(n_layers)]
    handles = install_hooks(model, MahalanobisLloydHook, mh_args, n_layers)
    t0 = time.time()
    ppl_mahal, loss_mahal = compute_ppl(model, eval_ids)
    print(f"  Mahalanobis Lloyd 2b PPL: {ppl_mahal:.4f} ({time.time()-t0:.1f}s)", flush=True)
    remove_hooks(handles)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  FP16 baseline:                      {ppl_fp16:8.4f}")
    print(f"  Uniform 2-bit:                      {ppl_uniform:8.4f}  (+{(ppl_uniform-ppl_fp16)/ppl_fp16*100:+.2f}%)")
    print(f"  L² Lloyd 2-bit:                     {ppl_l2:8.4f}  (+{(ppl_l2-ppl_fp16)/ppl_fp16*100:+.2f}%)")
    print(f"  Fisher-avg Mahalanobis Lloyd 2-bit: {ppl_mahal:8.4f}  (+{(ppl_mahal-ppl_fp16)/ppl_fp16*100:+.2f}%)")
    print()
    print(f"  Mahalanobis vs L² Lloyd: {(ppl_mahal-ppl_l2)/ppl_l2*100:+.2f}% "
          f"({'BETTER' if ppl_mahal < ppl_l2 else 'WORSE'})")
    print(f"  Mahalanobis vs Uniform:  {(ppl_mahal-ppl_uniform)/ppl_uniform*100:+.2f}% "
          f"({'BETTER' if ppl_mahal < ppl_uniform else 'WORSE'})")

    results = {
        'model': MODEL_NAME,
        'bits': BITS,
        'n_calib_tokens': N_CALIB_TOKENS,
        'n_eval_tokens': T_eval,
        'n_layers': n_layers,
        'n_kv': n_kv,
        'n_q': n_q,
        'head_dim': head_dim,
        'ppl_fp16': ppl_fp16,
        'ppl_uniform_2b': ppl_uniform,
        'ppl_l2_lloyd_2b': ppl_l2,
        'ppl_fisher_mahalanobis_lloyd_2b': ppl_mahal,
        'loss_fp16': loss_fp16,
        'loss_uniform': loss_uniform,
        'loss_l2': loss_l2,
        'loss_mahalanobis': loss_mahal,
        'mahalanobis_vs_l2_pct': (ppl_mahal - ppl_l2) / ppl_l2 * 100,
        'mahalanobis_vs_uniform_pct': (ppl_mahal - ppl_uniform) / ppl_uniform * 100,
        'runtime_sec': time.time() - t_start,
    }
    out_file = OUT_DIR / 'exp_next1_full_fisher_lloyd_ppl.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")
    print(f"Total runtime: {results['runtime_sec']:.1f}s ({results['runtime_sec']/60:.1f}m)")


if __name__ == '__main__':
    main()
