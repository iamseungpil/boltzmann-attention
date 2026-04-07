#!/usr/bin/env python3
"""
V2i: Full Ablation Matrix (Rotation × Quantizer × Sink)
=========================================================

Completes the ablation that reviewers will demand. We've already measured:
  Per-head PCA + Uniform 2-bit + no sink  (v2c)
  Per-head PCA + Uniform 2-bit + sink=1   (v2h)
  Per-head PCA + WF 2-bit + no sink       (v2c)
  Per-head PCA + WF 2-bit + sink=1        (v2h)

This script adds the Identity rotation row to complete a 2×2×2 cube:
  Rotation: {Identity, Per-head PCA}
  Quantizer: {Uniform 2-bit, WF 2-bit}
  Sink: {sink_k=0, sink_k=1}

8 configs × 3 models = 24 cells. Expected runtime: ~20 min.

The purpose: show that each of (per-head PCA), (WF), and (sink protection)
contributes a measurable and non-redundant improvement, and that their
combination is necessary for near-lossless 2-bit.
"""
import json, os, time, gc
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

DTYPE = torch.bfloat16
N_CALIB = 2048
N_EVAL = 2048
OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')

MODELS = [
    'mistralai/Mistral-7B-v0.3',
    'mistralai/Mistral-Nemo-Base-2407',
    'Qwen/Qwen2.5-7B',
]


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


def wf_alloc(sigma2, total_budget, b_floor=1, b_max=8):
    n = len(sigma2); s = np.maximum(sigma2, 1e-12)
    bits = np.zeros(n, dtype=int); spent = 0
    while spent < total_budget:
        bg = -np.inf; best = None
        for j in range(n):
            if bits[j] == 0:
                if spent + b_floor > total_budget: continue
                g = s[j] * (1.0 - 4.0**(-b_floor)) / b_floor
                if g > bg: bg, best = g, ('act', j)
            elif bits[j] < b_max:
                g = s[j] * (4.0**(-bits[j]) - 4.0**(-(bits[j]+1)))
                if g > bg: bg, best = g, ('add', j)
        if best is None: break
        op, j = best
        if op == 'act': bits[j] = b_floor; spent += b_floor
        else: bits[j] += 1; spent += 1
    return bits


def compute_ppl(model, ids):
    with torch.no_grad():
        out = model(ids, use_cache=False)
        logits = out.logits[:, :-1].contiguous()
        tgt = ids[:, 1:].contiguous()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                tgt.reshape(-1), reduction='mean')
        return float(torch.exp(loss).item())


class AblationHook:
    """
    rotation: 'identity' or 'pca' (per-head)
    scheme: 'uniform2' or 'wf2'
    sink_k: int
    """
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
            else:  # identity
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
            # Also store identity-basis per-dim variance (diag of cov), for identity-WF
            diag_var = np.maximum(np.diag(cov), 1e-12)
            ph.append({'V': V, 'mean': mean.astype(np.float32),
                       'eigvals': ev[order], 'K_pca': Kc @ V,
                       'K_c': Kc, 'diag_var': diag_var})
        basis[li] = ph
    return basis


def fit_cents(basis, rotation, scheme, n_layers, n_kv, head_dim, target_avg_bits=2.0):
    out = {}
    for li in range(n_layers):
        per = []
        for hk in range(n_kv):
            bh = basis[li][hk]
            if rotation == 'pca':
                sigma2 = bh['eigvals']; data = bh['K_pca']
            else:  # identity
                sigma2 = bh['diag_var']; data = bh['K_c']
            if scheme == 'uniform2':
                bits = np.full(head_dim, 2, dtype=int)
            elif scheme == 'wf2':
                bits = wf_alloc(sigma2, int(target_avg_bits * head_dim), 1, 8)
            else:
                raise ValueError(scheme)
            cents = []
            for j in range(head_dim):
                b = int(bits[j])
                cents.append(lloyd_1d(data[:, j], b, 15) if b > 0 else np.array([0.0], dtype=np.float32))
            per.append(cents)
        out[li] = per
    return out


def install(model, basis, cents, rotation, sink_k, n_layers, n_kv, head_dim):
    handles = []
    for li in range(n_layers):
        V_list = [basis[li][hk]['V'] for hk in range(n_kv)]
        mean_list = [basis[li][hk]['mean'] for hk in range(n_kv)]
        hook = AblationHook(n_kv, head_dim, rotation, V_list, mean_list, cents[li], sink_k)
        handles.append(model.model.layers[li].self_attn.k_proj.register_forward_hook(hook))
    return handles


def run_model(model_id):
    print(f"\n{'='*70}\n  {model_id}\n{'='*70}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, 'head_dim', None) or (model.config.hidden_size // model.config.num_attention_heads)
    print(f"  n_layers={n_layers}, n_kv={n_kv}, head_dim={head_dim}, loaded in {time.time()-t0:.1f}s", flush=True)

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in ds['text'] if len(t.strip()) > 100]
    calib = '\n\n'.join(texts[:300]); eval_t = '\n\n'.join(texts[300:600])
    calib_ids = tok(calib, return_tensors='pt', truncation=True, max_length=N_CALIB)['input_ids'].to('cuda:0')
    eval_ids = tok(eval_t, return_tensors='pt', truncation=True, max_length=N_EVAL)['input_ids'].to('cuda:0')

    ppl_fp16 = compute_ppl(model, eval_ids)
    print(f"  FP16: {ppl_fp16:.4f}", flush=True)

    basis = calibrate(model, calib_ids, n_layers, n_kv, head_dim)

    result = {'model': model_id, 'ppl_fp16': ppl_fp16, 'configs': {}}

    # 8 configs: rotation × scheme × sink
    for rotation in ['identity', 'pca']:
        for scheme in ['uniform2', 'wf2']:
            cents = fit_cents(basis, rotation, scheme, n_layers, n_kv, head_dim)
            for sink_k in [0, 1]:
                handles = install(model, basis, cents, rotation, sink_k, n_layers, n_kv, head_dim)
                ppl = compute_ppl(model, eval_ids)
                for h in handles: h.remove()
                key = f'{rotation}_{scheme}_sink{sink_k}'
                result['configs'][key] = ppl
                print(f"  [{rotation:<8} {scheme:<8} sink={sink_k}] PPL = {ppl:.4f}  (Δ = {ppl-ppl_fp16:+.4f})", flush=True)
            del cents; gc.collect()

    del model, tok, basis
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    print("="*70)
    print("V2i: Full Ablation Matrix (Rotation × Quantizer × Sink)")
    print("="*70, flush=True)
    t_start = time.time()

    results = {}
    for mid in MODELS:
        try:
            sn = mid.split('/')[-1].lower()
            results[sn] = run_model(mid)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY — 2-bit PPL by rotation × quantizer × sink")
    print("="*70)
    cols = [
        ('identity_uniform2_sink0', 'Id+U+k0'),
        ('identity_uniform2_sink1', 'Id+U+k1'),
        ('identity_wf2_sink0',      'Id+W+k0'),
        ('identity_wf2_sink1',      'Id+W+k1'),
        ('pca_uniform2_sink0',      'PCA+U+k0'),
        ('pca_uniform2_sink1',      'PCA+U+k1'),
        ('pca_wf2_sink0',           'PCA+W+k0'),
        ('pca_wf2_sink1',           'PCA+W+k1'),
    ]
    header = f"  {'model':<22}|{'FP16':>8}|" + "|".join(f"{label:>9}" for _, label in cols)
    print(header, flush=True)
    for sn, r in results.items():
        row = f"  {sn:<22}|{r['ppl_fp16']:>8.3f}|"
        for k, _ in cols:
            row += f"{r['configs'].get(k, float('nan')):>9.3f}|"
        print(row, flush=True)

    out = OUT_DIR / 'exp_v2i_ablation_matrix.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Runtime: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f}m)")


if __name__ == '__main__':
    main()
