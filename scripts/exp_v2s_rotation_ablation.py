#!/usr/bin/env python3
"""
V2s: Rotation Ablation — Does per-head PCA matter after sink + Lloyd?
=======================================================================

v2p showed: Mistral + per-head PCA + Lloyd + sink_k=1 ≈ near-lossless at 2-bit.
Open question: is the per-head PCA rotation actually contributing, or is it
redundant once sink protection + Lloyd (or uniform grid) are in place?

This experiment crosses 4 axes on Mistral-7B:
  Rotation:  {Identity, Per-head PCA}
  Quantizer: {Lloyd per-dim, Uniform grid per-dim}
  Sink:      {0, 1}
  EvalLen:   {2048, 32768}

= 2 × 2 × 2 × 2 = 16 configs.

Expected outcomes:
  H_pca_critical: Per-head PCA is essential. Identity + anything performs poorly.
  H_pca_redundant: Sink + Lloyd is enough. PCA gives marginal ≤0.1 PPL improvement.

For the paper, we need to know whether Theorem 6.16.3 (PCA optimality) is
load-bearing for the method, or whether the whole story collapses to "sink +
Lloyd" without needing the rotation.
"""
import json, os, time, gc
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

MODEL = 'mistralai/Mistral-7B-v0.3'
DTYPE = torch.bfloat16
N_CALIB = 2048
EVAL_LENGTHS = [2048, 32768]
OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')


def lloyd_1d(col, bits, n_iter=15):
    n_levels = 2 ** bits
    if n_levels <= 1:
        return np.array([float(col.mean())], dtype=np.float32)
    pcts = np.linspace(0, 100, n_levels + 2)[1:-1]
    c = np.sort(np.percentile(col, pcts)).astype(np.float64)
    for _ in range(n_iter):
        b = (c[:-1] + c[1:]) / 2
        idx = np.searchsorted(b, col)
        new_c = c.copy()
        for k in range(n_levels):
            m = idx == k
            if m.sum() > 0:
                new_c[k] = col[m].mean()
        if np.max(np.abs(new_c - c)) < 1e-6:
            break
        c = new_c
    return c.astype(np.float32)


def uniform_grid_1d(col, bits):
    n_levels = 2 ** bits
    if n_levels <= 1:
        return np.array([float(col.mean())], dtype=np.float32)
    r = float(np.max(np.abs(col)))
    if r < 1e-12:
        return np.array([0.0] * n_levels, dtype=np.float32)
    return np.linspace(-r, r, n_levels).astype(np.float32)


def compute_ppl(model, ids):
    with torch.no_grad():
        out = model(ids, use_cache=False)
        logits = out.logits[:, :-1].contiguous()
        tgt = ids[:, 1:].contiguous()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                tgt.reshape(-1), reduction='mean')
        return float(torch.exp(loss).item())


class RotQuantHook:
    def __init__(self, n_kv, head_dim, rotation, V_list, mean_list, cents_list, sink_k):
        self.n_kv = n_kv; self.head_dim = head_dim
        self.rotation = rotation
        self.V_list = V_list; self.mean_list = mean_list
        self.cents_list = cents_list; self.sink_k = sink_k

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x_orig = output.view(B, T, self.n_kv, self.head_dim).float().cpu().numpy()
        x = x_orig.copy()
        for hk in range(self.n_kv):
            m = self.mean_list[hk]
            cents = self.cents_list[hk]
            data = x[:, :, hk, :].reshape(-1, self.head_dim)
            K_c = data - m
            if self.rotation == 'pca':
                V = self.V_list[hk]
                K_pca = K_c @ V
            else:
                K_pca = K_c
            K_q = K_pca.copy()
            for j in range(self.head_dim):
                cj = cents[j]
                if cj is None or len(cj) == 1:
                    K_q[:, j] = cj[0] if cj is not None else 0.0
                else:
                    bnd = (cj[:-1] + cj[1:]) / 2
                    idx = np.searchsorted(bnd, K_pca[:, j])
                    K_q[:, j] = cj[idx]
            if self.rotation == 'pca':
                K_rec = K_q @ self.V_list[hk].T + m
            else:
                K_rec = K_q + m
            x[:, :, hk, :] = K_rec.reshape(B, T, self.head_dim)
        if self.sink_k > 0:
            k = min(self.sink_k, T)
            x[:, :k, :, :] = x_orig[:, :k, :, :]
        return torch.from_numpy(x).to(output.device).to(output.dtype).view(B, T, self.n_kv * self.head_dim)


def calibrate(model, ids, n_layers, n_kv, head_dim):
    pl = {}
    def mk(li):
        def h(m, i, o): pl[li] = o.detach().cpu().float().numpy()
        return h
    handles = [model.model.layers[li].self_attn.k_proj.register_forward_hook(mk(li)) for li in range(n_layers)]
    with torch.no_grad():
        _ = model(ids, use_cache=False)
    for h in handles: h.remove()

    basis = {}
    for li in range(n_layers):
        K_all = pl[li].reshape(-1, n_kv, head_dim).astype(np.float32)
        ph = []
        for hk in range(n_kv):
            K = K_all[:, hk, :]; mean = K.mean(axis=0); Kc = K - mean
            cov = (Kc.T @ Kc) / max(K.shape[0]-1, 1)
            ev, vv = np.linalg.eigh(cov)
            order = np.argsort(ev)[::-1]
            V = vv[:, order].astype(np.float32)
            ph.append({'V': V, 'mean': mean.astype(np.float32),
                       'eigvals': ev[order], 'K_pca': Kc @ V, 'K_c': Kc})
        basis[li] = ph
    return basis


def fit_cents(basis, rotation, qtype, n_layers, n_kv, head_dim, bits=2):
    out = {}
    for li in range(n_layers):
        per = []
        for hk in range(n_kv):
            data = basis[li][hk]['K_pca'] if rotation == 'pca' else basis[li][hk]['K_c']
            cents = []
            for j in range(head_dim):
                if qtype == 'lloyd':
                    cents.append(lloyd_1d(data[:, j], bits, 15))
                else:
                    cents.append(uniform_grid_1d(data[:, j], bits))
            per.append(cents)
        out[li] = per
    return out


def install(model, basis, cents, rotation, sink_k, n_layers, n_kv, head_dim):
    handles = []
    for li in range(n_layers):
        V_list = [basis[li][hk]['V'] for hk in range(n_kv)]
        mean_list = [basis[li][hk]['mean'] for hk in range(n_kv)]
        hook = RotQuantHook(n_kv, head_dim, rotation, V_list, mean_list, cents[li], sink_k)
        handles.append(model.model.layers[li].self_attn.k_proj.register_forward_hook(hook))
    return handles


def main():
    print("="*70)
    print("V2s: Rotation Ablation (Mistral)")
    print("="*70, flush=True)
    t_start = time.time()

    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='sdpa', low_cpu_mem_usage=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, 'head_dim', None) or (model.config.hidden_size // model.config.num_attention_heads)
    print(f"  n_layers={n_layers}, n_kv={n_kv}, head_dim={head_dim}", flush=True)

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in ds['text'] if len(t.strip()) > 100]
    calib_text = '\n\n'.join(texts[:300])
    eval_text = '\n\n'.join(texts[300:3000])
    calib_ids = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB)['input_ids'].to('cuda:0')

    fp16 = {}
    for L in EVAL_LENGTHS:
        eval_ids = tok(eval_text, return_tensors='pt', truncation=True, max_length=L)['input_ids'].to('cuda:0')
        fp16[L] = compute_ppl(model, eval_ids)
        print(f"  FP16 @ L={L}: {fp16[L]:.4f}", flush=True)

    basis = calibrate(model, calib_ids, n_layers, n_kv, head_dim)
    print(f"  Calibrated in {time.time()-t_start:.1f}s", flush=True)

    # Pre-fit all 4 centroid sets
    cents = {}
    for rotation in ['identity', 'pca']:
        for qtype in ['lloyd', 'grid']:
            print(f"  Fitting {rotation} + {qtype}...", flush=True)
            cents[(rotation, qtype)] = fit_cents(basis, rotation, qtype, n_layers, n_kv, head_dim)

    results = {'model': MODEL, 'fp16': fp16, 'configs': {}}

    for L in EVAL_LENGTHS:
        eval_ids = tok(eval_text, return_tensors='pt', truncation=True, max_length=L)['input_ids'].to('cuda:0')
        for rotation in ['identity', 'pca']:
            for qtype in ['lloyd', 'grid']:
                for sink_k in [0, 1]:
                    handles = install(model, basis, cents[(rotation, qtype)], rotation, sink_k, n_layers, n_kv, head_dim)
                    ppl = compute_ppl(model, eval_ids)
                    for h in handles: h.remove()
                    key = f'L{L}_{rotation}_{qtype}_sink{sink_k}'
                    results['configs'][key] = ppl
                    print(f"  [L={L:>5} {rotation:<8} {qtype:<5} sink={sink_k}] PPL = {ppl:.4f}  "
                          f"(Δ = {ppl-fp16[L]:+.4f})", flush=True)

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY — Rotation × Quantizer × Sink × Length")
    print("="*70)
    for L in EVAL_LENGTHS:
        print(f"\n  L = {L} (FP16 = {fp16[L]:.4f}):")
        print(f"  {'rotation':<10}|{'quant':<6}|{'sink=0':>10}|{'sink=1':>10}")
        for rot in ['identity', 'pca']:
            for qt in ['lloyd', 'grid']:
                k0 = f'L{L}_{rot}_{qt}_sink0'
                k1 = f'L{L}_{rot}_{qt}_sink1'
                print(f"  {rot:<10}|{qt:<6}|{results['configs'][k0]:>10.4f}|{results['configs'][k1]:>10.4f}")

    out = OUT_DIR / 'exp_v2s_rotation_ablation.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Runtime: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f}m)")


if __name__ == '__main__':
    main()
