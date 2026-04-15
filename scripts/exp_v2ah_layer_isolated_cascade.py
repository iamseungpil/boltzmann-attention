#!/usr/bin/env python3
"""
V2ah: Layer-Isolated Cascade Contribution
============================================

v2ag showed that full-model residual error r_final correctly predicts r_ppl
on all 4 models, with model-specific "cascade factors" (Mistral 2.7×,
Nemo 1.5×, Qwen-7B 1.4×, Qwen-1.5B 2.0×) by which the per-layer attention
error gets amplified into the final residual.

This experiment dissects the cascade by quantizing ONE layer at a time
and measuring the resulting final-layer residual error. This decomposes
the full cascade into per-layer isolated contributions:

  ‖Δh_L^{ℓ-only}‖² = forward Jacobian × Δo_ℓ × Jacobian^T applied
                    along the path layer ℓ → L

Protocol per model:
  1. Calibrate PCA + Lloyd/Grid centroids at every layer.
  2. FP16 forward, capture h_L (final layer residual).
  3. For each selected layer ℓ ∈ sampled set:
       a. Install quant hook ONLY at layer ℓ (Lloyd, then Grid)
       b. Forward, capture h_L^{ℓ-only,Lloyd} and h_L^{ℓ-only,Grid}
       c. Compute ‖h_L^{ℓ-only} - h_L^FP16‖²
  4. Compare across layers to identify cascade hot-spots.

We sample ~8 layers per model to keep runtime tractable.

Expected interpretation:
  - Mistral: Layer 0/1 (sink layers) should have huge isolated contribution
    because their Lloyd error projects onto the BOS direction, and the
    sink mechanism propagates that error through every subsequent layer's
    attention.
  - Nemo: contribution should be more uniformly spread across layers
    (many delimiter sinks at different layers).
  - Qwen-1.5B: cascade should be late-layer dominated (small model, errors
    accumulate in last few layers because there's less compensating
    capacity).
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
N_LAYERS_SAMPLE = 8
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


def capture_final_residual(model, ids, n_layers):
    """Capture final-layer residual after forward."""
    captured = {}
    def h(m, i, o):
        a = o[0] if isinstance(o, tuple) else o
        captured['h_final'] = a.detach().clone()
    handle = model.model.layers[n_layers - 1].register_forward_hook(h)
    with torch.no_grad():
        _ = model(ids, use_cache=False)
    handle.remove()
    return captured['h_final']


def install_single_layer_hook(model, layer_idx, basis, cents, n_kv, head_dim):
    V_list = [basis[layer_idx][hk]['V'] for hk in range(n_kv)]
    mean_list = [basis[layer_idx][hk]['mean'] for hk in range(n_kv)]
    hook = QuantHook(n_kv, head_dim, V_list, mean_list, cents[layer_idx])
    handle = model.model.layers[layer_idx].self_attn.k_proj.register_forward_hook(hook)
    return handle


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
    print(f"  n_layers={n_layers} loaded in {time.time()-t0:.1f}s", flush=True)

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in ds['text'] if len(t.strip()) > 100]
    calib_text = '\n\n'.join(texts[:300])
    eval_text  = '\n\n'.join(texts[300:600])
    calib_ids = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB)['input_ids'].to('cuda:0')
    eval_ids  = tok(eval_text,  return_tensors='pt', truncation=True, max_length=N_EVAL)['input_ids'].to('cuda:0')

    basis = calibrate(model, calib_ids, n_layers, n_kv, head_dim)
    cents_lloyd = fit_cents(basis, 'lloyd', n_layers, n_kv, head_dim)
    cents_grid  = fit_cents(basis, 'grid',  n_layers, n_kv, head_dim)
    print(f"  Calibrated in {time.time()-t0:.1f}s", flush=True)

    h_fp = capture_final_residual(model, eval_ids, n_layers)
    base_norm = float(h_fp.float().pow(2).sum(dim=-1).mean().item())
    print(f"  FP16 final residual norm² (per token avg) = {base_norm:.3e}", flush=True)

    # Sample layers to test (always include 0, 1, last; spread rest)
    if n_layers <= N_LAYERS_SAMPLE + 2:
        sample_layers = list(range(n_layers))
    else:
        # Force include 0, 1
        sample_layers = [0, 1]
        step = (n_layers - 3) // (N_LAYERS_SAMPLE - 3)
        for i in range(N_LAYERS_SAMPLE - 3):
            sample_layers.append(2 + i * step)
        sample_layers.append(n_layers - 1)
        sample_layers = sorted(set(sample_layers))

    print(f"  Sampled layers: {sample_layers}", flush=True)

    per_layer = []
    for li in sample_layers:
        # Lloyd at this layer only
        h_ll_handle = install_single_layer_hook(model, li, basis, cents_lloyd, n_kv, head_dim)
        h_ll = capture_final_residual(model, eval_ids, n_layers)
        h_ll_handle.remove()
        d_ll = (h_ll - h_fp).float().pow(2).sum(dim=-1).mean().item()

        # Grid at this layer only
        h_gr_handle = install_single_layer_hook(model, li, basis, cents_grid, n_kv, head_dim)
        h_gr = capture_final_residual(model, eval_ids, n_layers)
        h_gr_handle.remove()
        d_gr = (h_gr - h_fp).float().pow(2).sum(dim=-1).mean().item()

        per_layer.append({
            'layer': li,
            'final_delta_lloyd': float(d_ll),
            'final_delta_grid': float(d_gr),
            'rel_lloyd': float(d_ll / max(base_norm, 1e-12)),
            'rel_grid':  float(d_gr / max(base_norm, 1e-12)),
            'ratio_ll_over_gr': float(d_ll / max(d_gr, 1e-12)),
        })

    # Print
    print(f"\n  Per-layer isolated cascade (Lloyd vs Grid at layer ℓ → final residual):")
    print(f"  {'layer':<6}|{'Δh²_Ll':>12}|{'Δh²_Gr':>12}|{'rel_Ll':>10}|{'rel_Gr':>10}|{'Ll/Gr':>10}")
    for p in per_layer:
        print(f"  {p['layer']:<6}|{p['final_delta_lloyd']:>12.3e}|{p['final_delta_grid']:>12.3e}|"
              f"{p['rel_lloyd']*100:>9.2f}%|{p['rel_grid']*100:>9.2f}%|{p['ratio_ll_over_gr']:>10.3f}")

    # Find dominant layer
    dom_layer = max(per_layer, key=lambda p: p['final_delta_lloyd'])
    print(f"\n  Dominant single-layer Lloyd contribution: L{dom_layer['layer']} "
          f"(rel {dom_layer['rel_lloyd']*100:.2f}%)", flush=True)

    del model, tok
    gc.collect(); torch.cuda.empty_cache()

    return {
        'model': model_id, 'short_name': sn,
        'n_layers': n_layers,
        'base_norm': base_norm,
        'sampled_layers': sample_layers,
        'per_layer': per_layer,
        'dominant_layer': dom_layer['layer'],
    }


def main():
    print("="*70)
    print("V2ah: Layer-Isolated Cascade Contribution")
    print("="*70, flush=True)
    t_start = time.time()

    results = {}
    for mid, sn in MODELS:
        try:
            results[sn] = analyze_model(mid, sn)
        except Exception as e:
            print(f"ERROR on {sn}: {e}")
            import traceback; traceback.print_exc()

    # Summary: which layers dominate per model
    print("\n" + "="*70)
    print("CASCADE PROFILE SUMMARY")
    print("="*70)
    for sn, r in results.items():
        print(f"\n  {sn} (n_layers={r['n_layers']}, dominant L{r['dominant_layer']}):")
        per = r['per_layer']
        # Total Lloyd & Grid
        tot_ll = sum(p['final_delta_lloyd'] for p in per)
        tot_gr = sum(p['final_delta_grid']  for p in per)
        print(f"    Sum-of-isolated  Ll={tot_ll:.3e}  Gr={tot_gr:.3e}  ratio={tot_ll/max(tot_gr,1e-12):.3f}")
        # Top-3 Lloyd contributors
        top3 = sorted(per, key=lambda p: -p['final_delta_lloyd'])[:3]
        for rank, p in enumerate(top3):
            frac = p['final_delta_lloyd'] / max(tot_ll, 1e-12) * 100
            print(f"    Rank {rank+1}: L{p['layer']} → {frac:.1f}% of total Lloyd cascade")

    out = OUT_DIR / 'exp_v2ah_layer_cascade.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Runtime: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f}m)")


if __name__ == '__main__':
    main()
