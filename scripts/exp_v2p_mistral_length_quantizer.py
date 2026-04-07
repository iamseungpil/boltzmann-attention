#!/usr/bin/env python3
"""
V2p: Mistral Eval-Length Sweep × Quantizer Type × Sink Protection
===================================================================

Purpose: Reproduce (or refute) the coworker's observation that "Lloyd 2-bit
catastrophically fails (PPL 180) even with sink_k=4 protection (PPL 162)"
on Mistral-7B at 49K eval tokens.

Our v2h result on 2048 eval tokens: Lloyd-per-dim + sink_k=1 → 5.97 (uniform2)
vs 7.08 no sink, which is a clean win. Coworker's result suggests at 49K the
picture is different.

Two hypotheses for the discrepancy:
  H1 (eval length): Longer eval surfaces more sink-like tokens (multiple BOS-
      like delimiters in 49K tokens) that sink_k=1 doesn't protect.
  H2 (quantizer type): Coworker's "Uniform" might be a plain scale-and-grid
      quantizer (bounded L∞), whereas our Lloyd-Max places centroids at mass
      centers leaving extreme values poorly reconstructed. Uniform grid should
      beat Lloyd on heavy-tailed distributions.

This experiment crosses both:
  Eval length:     {2048, 8192, 32768}
  Quantizer type:  {Lloyd per-dim, Uniform grid per-dim}
  Sink protection: {0, 1, 4}
= 3 × 2 × 3 = 18 configs on Mistral-7B-v0.3 with per-head PCA rotation.

Both quantizers allocate 2 bits per dim uniformly.
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
EVAL_LENGTHS = [2048, 8192, 32768]
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
    """
    Symmetric max-based uniform grid quantizer.
    Scale r = max(|col|) over calibration data (can clip percentile if desired).
    Levels = linspace(-r, r, n_levels).
    """
    n_levels = 2 ** bits
    if n_levels <= 1:
        return np.array([float(col.mean())], dtype=np.float32)
    r = float(np.max(np.abs(col)))
    if r < 1e-12:
        return np.array([0.0] * n_levels, dtype=np.float32)
    levels = np.linspace(-r, r, n_levels).astype(np.float32)
    return levels


def compute_ppl(model, ids):
    with torch.no_grad():
        out = model(ids, use_cache=False)
        logits = out.logits[:, :-1].contiguous()
        tgt = ids[:, 1:].contiguous()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                tgt.reshape(-1), reduction='mean')
        return float(torch.exp(loss).item())


class QuantHook:
    """Per-head PCA + per-dim scalar quantizer (Lloyd or uniform grid) + sink FP16."""
    def __init__(self, n_kv, head_dim, V_list, mean_list, cents_list, sink_k):
        self.n_kv = n_kv; self.head_dim = head_dim
        self.V_list = V_list; self.mean_list = mean_list
        self.cents_list = cents_list; self.sink_k = sink_k

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x_orig = output.view(B, T, self.n_kv, self.head_dim).float().cpu().numpy()
        x = x_orig.copy()
        for hk in range(self.n_kv):
            V = self.V_list[hk]; m = self.mean_list[hk]; cents = self.cents_list[hk]
            data = x[:, :, hk, :].reshape(-1, self.head_dim)
            K_c = data - m
            K_pca = K_c @ V
            K_q = K_pca.copy()
            for j in range(self.head_dim):
                cj = cents[j]
                if cj is None or len(cj) == 1:
                    K_q[:, j] = cj[0] if cj is not None else 0.0
                else:
                    bnd = (cj[:-1] + cj[1:]) / 2
                    idx = np.searchsorted(bnd, K_pca[:, j])
                    K_q[:, j] = cj[idx]
            K_rec = K_q @ V.T + m
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
                       'eigvals': ev[order], 'K_pca': Kc @ V})
        basis[li] = ph
    return basis


def fit_cents(basis, qtype, n_layers, n_kv, head_dim, bits=2):
    """qtype: 'lloyd' or 'uniform_grid'"""
    out = {}
    for li in range(n_layers):
        per = []
        for hk in range(n_kv):
            K_pca = basis[li][hk]['K_pca']
            cents = []
            for j in range(head_dim):
                if qtype == 'lloyd':
                    cents.append(lloyd_1d(K_pca[:, j], bits, 15))
                elif qtype == 'uniform_grid':
                    cents.append(uniform_grid_1d(K_pca[:, j], bits))
                else:
                    raise ValueError(qtype)
            per.append(cents)
        out[li] = per
    return out


def install(model, basis, cents, sink_k, n_layers, n_kv, head_dim):
    handles = []
    for li in range(n_layers):
        V_list = [basis[li][hk]['V'] for hk in range(n_kv)]
        mean_list = [basis[li][hk]['mean'] for hk in range(n_kv)]
        hook = QuantHook(n_kv, head_dim, V_list, mean_list, cents[li], sink_k)
        handles.append(model.model.layers[li].self_attn.k_proj.register_forward_hook(hook))
    return handles


def main():
    print("="*70)
    print("V2p: Mistral Eval-Length × Quantizer × Sink")
    print("="*70, flush=True)
    t_start = time.time()

    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, 'head_dim', None) or (model.config.hidden_size // model.config.num_attention_heads)
    print(f"  n_layers={n_layers}, n_kv={n_kv}, head_dim={head_dim}", flush=True)

    # Data
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in ds['text'] if len(t.strip()) > 100]
    calib_text = '\n\n'.join(texts[:300])
    # Use a LOT of text for eval so we can truncate at long lengths
    eval_text = '\n\n'.join(texts[300:3000])  # ~2000 texts, should give 50K+ tokens
    print(f"  Eval text length: {len(eval_text)} chars", flush=True)

    calib_ids = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB)['input_ids'].to('cuda:0')
    print(f"  Calib tokens: {calib_ids.shape[1]}", flush=True)

    # FP16 baselines at each eval length
    fp16_ppls = {}
    for L in EVAL_LENGTHS:
        eval_ids = tok(eval_text, return_tensors='pt', truncation=True, max_length=L)['input_ids'].to('cuda:0')
        actual_T = eval_ids.shape[1]
        ppl = compute_ppl(model, eval_ids)
        fp16_ppls[L] = {'target_len': L, 'actual_len': actual_T, 'ppl_fp16': ppl}
        print(f"  FP16 @ L={L} (actual {actual_T}): {ppl:.4f}", flush=True)

    # Calibrate once (on 2048 cal tokens, reused for all eval lengths)
    basis = calibrate(model, calib_ids, n_layers, n_kv, head_dim)
    print(f"  Calibrated in {time.time()-t_start:.1f}s", flush=True)

    # Fit both quantizer types once
    print("\n  Fitting Lloyd centroids...", flush=True)
    cents_lloyd = fit_cents(basis, 'lloyd', n_layers, n_kv, head_dim)
    print("  Fitting Uniform grid levels...", flush=True)
    cents_grid = fit_cents(basis, 'uniform_grid', n_layers, n_kv, head_dim)

    results = {'model': MODEL, 'fp16_per_length': fp16_ppls, 'configs': {}}

    for L in EVAL_LENGTHS:
        eval_ids = tok(eval_text, return_tensors='pt', truncation=True, max_length=L)['input_ids'].to('cuda:0')
        actual_T = eval_ids.shape[1]
        ppl_fp16 = fp16_ppls[L]['ppl_fp16']
        for qtype, cents in [('lloyd', cents_lloyd), ('grid', cents_grid)]:
            for sink_k in [0, 1, 4]:
                handles = install(model, basis, cents, sink_k, n_layers, n_kv, head_dim)
                ppl = compute_ppl(model, eval_ids)
                for h in handles: h.remove()
                key = f'L{L}_{qtype}_sink{sink_k}'
                results['configs'][key] = {
                    'eval_len': actual_T,
                    'quantizer': qtype,
                    'sink_k': sink_k,
                    'ppl': ppl,
                    'delta_fp16': ppl - ppl_fp16,
                }
                print(f"  [L={L:>5} {qtype:<5} sink={sink_k}] PPL = {ppl:.4f}  (Δ = {ppl-ppl_fp16:+.4f})", flush=True)

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY — PPL by (eval length, quantizer, sink)")
    print("="*70)
    print(f"  {'EvalLen':<8}|{'FP16':>8}|{'Ll-k0':>9}|{'Ll-k1':>9}|{'Ll-k4':>9}|{'Gr-k0':>9}|{'Gr-k1':>9}|{'Gr-k4':>9}")
    for L in EVAL_LENGTHS:
        fp = fp16_ppls[L]['ppl_fp16']
        row = f"  {L:<8}|{fp:>8.3f}|"
        for qtype in ['lloyd', 'grid']:
            for sink_k in [0, 1, 4]:
                k = f'L{L}_{qtype}_sink{sink_k}'
                ppl = results['configs'][k]['ppl']
                row += f"{ppl:>9.3f}|"
        print(row, flush=True)

    out = OUT_DIR / 'exp_v2p_length_quantizer.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Runtime: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f}m)")


if __name__ == '__main__':
    main()
