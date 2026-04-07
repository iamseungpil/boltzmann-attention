#!/usr/bin/env python3
"""
V2ag: Full-Model Residual Stream Error vs PPL Ratio
======================================================

v2af showed that per-layer ||Δo||² (single-layer attention output error)
predicts PPL direction correctly on 3/4 models but FAILS on Qwen-1.5B
(r_exact = 0.76 vs r_ppl = 1.41). This suggests cross-layer error
cascading dominates on small models.

This experiment measures the FULL-MODEL residual stream perturbation
by running the model with quantization applied at EVERY layer, capturing
h_ℓ (the residual stream after layer ℓ) and comparing to the FP16 trace.

Protocol per model:
  1. Calibrate PCA + per-dim Lloyd / Grid centroids for every layer/head.
  2. Run FP16 forward on eval, capture h_ℓ after each layer.
  3. Install Lloyd hooks on every k_proj → forward → capture h_ℓ^Lloyd.
  4. Install Grid  hooks on every k_proj → forward → capture h_ℓ^Grid.
  5. Per layer ℓ: compute ||h_ℓ^Lloyd - h_ℓ^FP16||^2 and ||h_ℓ^Grid - h_ℓ^FP16||^2.
  6. Aggregate: per-layer ratios, final layer ratio, cumulative ratio.
  7. Compare final-layer ratio against r_ppl from v2p/v2u/v2aa.

If r_final correctly predicts r_ppl sign on all 4 models (including
Qwen-1.5B), cross-layer cascading is the explanation for v2af's failure
and the full-model metric is sufficient.

Runtime: ~5 minutes (3 forwards per model × 4 models).
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
    ('mistralai/Mistral-7B-v0.3', 'mistral-7b'),
    ('mistralai/Mistral-Nemo-Base-2407', 'nemo-12b'),
    ('Qwen/Qwen2.5-7B', 'qwen-7b'),
    ('Qwen/Qwen2.5-1.5B', 'qwen-1.5b'),
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


def uniform_grid_1d(col, bits):
    n_levels = 2 ** bits
    if n_levels <= 1:
        return np.array([float(col.mean())], dtype=np.float32)
    r = float(np.max(np.abs(col)))
    if r < 1e-12:
        return np.array([0.0] * n_levels, dtype=np.float32)
    return np.linspace(-r, r, n_levels).astype(np.float32)


class QuantHook:
    """Per-head PCA + per-dim scalar quantizer (Lloyd or grid)."""
    def __init__(self, n_kv, head_dim, V_list, mean_list, cents_list):
        self.n_kv = n_kv; self.head_dim = head_dim
        self.V_list = V_list; self.mean_list = mean_list
        self.cents_list = cents_list
    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x = output.view(B, T, self.n_kv, self.head_dim).float().cpu().numpy()
        for hk in range(self.n_kv):
            V = self.V_list[hk]; m = self.mean_list[hk]; cents = self.cents_list[hk]
            data = x[:, :, hk, :].reshape(-1, self.head_dim)
            K_c = data - m; K_pca = K_c @ V
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


def fit_cents(basis, kind, n_layers, n_kv, head_dim, bits=2):
    out = {}
    for li in range(n_layers):
        per = []
        for hk in range(n_kv):
            K_pca = basis[li][hk]['K_pca']
            cents = []
            for j in range(head_dim):
                if kind == 'lloyd':
                    cents.append(lloyd_1d(K_pca[:, j], bits, 15))
                else:
                    cents.append(uniform_grid_1d(K_pca[:, j], bits))
            per.append(cents)
        out[li] = per
    return out


def capture_residuals(model, ids, n_layers):
    """Forward pass, capture residual stream (decoder layer output) per layer."""
    captured = {}
    def mk(li):
        def h(m, i, o):
            # HuggingFace decoder layer output is a tuple: (hidden_states, ...)
            a = o[0] if isinstance(o, tuple) else o
            captured[li] = a.detach().clone()
        return h
    handles = [model.model.layers[li].register_forward_hook(mk(li)) for li in range(n_layers)]
    with torch.no_grad():
        _ = model(ids, use_cache=False)
    for h in handles: h.remove()
    return captured


def install_quant_hooks(model, basis, cents, n_layers, n_kv, head_dim):
    handles = []
    for li in range(n_layers):
        V_list = [basis[li][hk]['V'] for hk in range(n_kv)]
        mean_list = [basis[li][hk]['mean'] for hk in range(n_kv)]
        hook = QuantHook(n_kv, head_dim, V_list, mean_list, cents[li])
        handles.append(model.model.layers[li].self_attn.k_proj.register_forward_hook(hook))
    return handles


def analyze_model(model_id, sn):
    print(f"\n{'='*70}\n  {sn}: {model_id}\n{'='*70}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='sdpa', low_cpu_mem_usage=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, 'head_dim', None) or (model.config.hidden_size // model.config.num_attention_heads)
    print(f"  n_layers={n_layers} n_kv={n_kv} head_dim={head_dim} loaded in {time.time()-t0:.1f}s", flush=True)

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in ds['text'] if len(t.strip()) > 100]
    calib_text = '\n\n'.join(texts[:300])
    eval_text  = '\n\n'.join(texts[300:600])
    calib_ids = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB)['input_ids'].to('cuda:0')
    eval_ids  = tok(eval_text,  return_tensors='pt', truncation=True, max_length=N_EVAL)['input_ids'].to('cuda:0')

    # ---- Calibration ----
    basis = calibrate(model, calib_ids, n_layers, n_kv, head_dim)
    cents_lloyd = fit_cents(basis, 'lloyd', n_layers, n_kv, head_dim)
    cents_grid  = fit_cents(basis, 'grid',  n_layers, n_kv, head_dim)
    print(f"  Calibrated in {time.time()-t0:.1f}s", flush=True)

    # ---- FP16 residual stream ----
    h_fp = capture_residuals(model, eval_ids, n_layers)
    print(f"  FP16 forward done at {time.time()-t0:.1f}s", flush=True)

    # ---- Lloyd all-layer quantized residual stream ----
    handles = install_quant_hooks(model, basis, cents_lloyd, n_layers, n_kv, head_dim)
    h_ll = capture_residuals(model, eval_ids, n_layers)
    for h in handles: h.remove()
    print(f"  Lloyd forward done at {time.time()-t0:.1f}s", flush=True)

    # ---- Grid all-layer quantized residual stream ----
    handles = install_quant_hooks(model, basis, cents_grid, n_layers, n_kv, head_dim)
    h_gr = capture_residuals(model, eval_ids, n_layers)
    for h in handles: h.remove()
    print(f"  Grid  forward done at {time.time()-t0:.1f}s", flush=True)

    del model, tok
    gc.collect(); torch.cuda.empty_cache()

    # ---- Compute per-layer residual error norms ----
    per_layer = []
    for li in range(n_layers):
        delta_ll = (h_ll[li] - h_fp[li]).float()
        delta_gr = (h_gr[li] - h_fp[li]).float()
        # ||Δh||² averaged over positions, averaged over features
        norm_ll = float(delta_ll.pow(2).sum(dim=-1).mean().item())  # avg per token
        norm_gr = float(delta_gr.pow(2).sum(dim=-1).mean().item())
        # Also relative to FP16 residual magnitude
        base_norm = float(h_fp[li].float().pow(2).sum(dim=-1).mean().item())
        rel_ll = norm_ll / max(base_norm, 1e-12)
        rel_gr = norm_gr / max(base_norm, 1e-12)
        per_layer.append({
            'layer': li,
            'delta_lloyd': norm_ll, 'delta_grid': norm_gr,
            'base_norm': base_norm,
            'rel_lloyd': rel_ll, 'rel_grid': rel_gr,
            'ratio_lloyd_over_grid': norm_ll / max(norm_gr, 1e-12),
        })

    # Aggregate metrics
    delta_ll_final = per_layer[-1]['delta_lloyd']
    delta_gr_final = per_layer[-1]['delta_grid']
    delta_ll_sum = sum(p['delta_lloyd'] for p in per_layer)
    delta_gr_sum = sum(p['delta_grid']  for p in per_layer)
    delta_ll_max = max(p['delta_lloyd'] for p in per_layer)
    delta_gr_max = max(p['delta_grid']  for p in per_layer)

    r_final = delta_ll_final / max(delta_gr_final, 1e-12)
    r_sum   = delta_ll_sum   / max(delta_gr_sum,   1e-12)
    r_max   = delta_ll_max   / max(delta_gr_max,   1e-12)

    print(f"\n  Per-layer Δh² (last 5 layers):")
    for p in per_layer[-5:]:
        print(f"    L{p['layer']:<3} Ll={p['delta_lloyd']:.3e}  Gr={p['delta_grid']:.3e}  "
              f"ratio Ll/Gr={p['ratio_lloyd_over_grid']:.3f}  rel={p['rel_lloyd']*100:.2f}%/{p['rel_grid']*100:.2f}%")
    print(f"\n  Aggregates:")
    print(f"    r_final (last layer)  = {r_final:.4f}")
    print(f"    r_sum   (sum layers)  = {r_sum:.4f}")
    print(f"    r_max   (max layer)   = {r_max:.4f}", flush=True)

    return {
        'model': model_id, 'short_name': sn,
        'n_layers': n_layers,
        'per_layer': per_layer,
        'ratios': {
            'r_final': r_final,
            'r_sum':   r_sum,
            'r_max':   r_max,
        },
    }


def main():
    print("="*70)
    print("V2ag: Full-Model Residual Stream Error")
    print("="*70, flush=True)
    t_start = time.time()

    results = {}
    for mid, sn in MODELS:
        try:
            results[sn] = analyze_model(mid, sn)
        except Exception as e:
            print(f"ERROR on {sn}: {e}")
            import traceback; traceback.print_exc()

    ref_ppl = {
        'mistral-7b': 9.9644  / 6.4343,
        'nemo-12b':   8.3734  / 7.0115,
        'qwen-7b':    7.3914  / 7.8364,
        'qwen-1.5b':  21.4728 / 15.2580,
    }

    print("\n" + "="*70)
    print("VALIDATION — r_final predicts r_ppl direction?")
    print("="*70)
    print(f"  {'model':<14}|{'r_final':>10}|{'r_sum':>10}|{'r_max':>10}|{'r_ppl':>10}"
          f"|{'final OK':>10}|{'sum OK':>9}")
    for sn, r in results.items():
        rr = r['ratios']
        rp = ref_ppl.get(sn, float('nan'))
        ok_f = '✓' if (rr['r_final'] > 1) == (rp > 1) else '✗'
        ok_s = '✓' if (rr['r_sum']   > 1) == (rp > 1) else '✗'
        print(f"  {sn:<14}|{rr['r_final']:>10.3f}|{rr['r_sum']:>10.3f}|{rr['r_max']:>10.3f}|"
              f"{rp:>10.3f}|{ok_f:>10}|{ok_s:>9}")

    out = OUT_DIR / 'exp_v2ag_full_model_residual.json'
    with open(out, 'w') as f:
        json.dump({
            'reference_ppl_Lloyd_over_Grid': ref_ppl,
            'results': results,
        }, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Runtime: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f}m)")


if __name__ == '__main__':
    main()
