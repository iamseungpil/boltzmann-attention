#!/usr/bin/env python3
"""
exp_next9: Llama-3.1-8B full mode + PPL characterization (5th data point)
==========================================================================

Goal: add Llama-3.1-8B to the 4-model 3-mode classification table from
V2_THEORY_AND_3MODES_2026-04-08.md as the 5th held-out data point.

Measures:
  1. pos0_attn  — mean attention mass on position 0 across top-32 high-κ heads
  2. κ_max      — max condition number of per-head Pre-RoPE key covariance
  3. PPL @ 2-bit Pre-RoPE per-head PCA × {Lloyd, Grid} × {sink_k=0, sink_k=1}
     at L ∈ {2048, 8192}.

Then classifies into Mode A / B / C using the published thresholds:
  Mode A: pos0_attn > 0.40 AND κ_max > 1e6
  Mode B: pos0_attn < 0.20 AND κ_max > 1e6
  Mode C: κ_max  < 1e5

Output: reports/axis2_theoretical_verification/exp_next9_llama_full.json
"""
import json, os, time, gc
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

DTYPE = torch.bfloat16
N_CALIB = 2048
EVAL_LENGTHS = [2048, 8192]
MODEL_ID = os.environ.get('LLAMA_MODEL_ID', 'NousResearch/Meta-Llama-3.1-8B')
SHORT_NAME = 'llama-3.1-8b'
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
        B, Tm1, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        tgt_flat = tgt.reshape(-1)
        total_loss = 0.0
        total_n = 0
        CHUNK = 4096
        for s in range(0, logits_flat.size(0), CHUNK):
            e = min(s + CHUNK, logits_flat.size(0))
            l = logits_flat[s:e].float()
            t = tgt_flat[s:e]
            loss = F.cross_entropy(l, t, reduction='sum')
            total_loss += float(loss.item())
            total_n += (e - s)
            del l
        del logits, logits_flat
        return float(np.exp(total_loss / total_n))


class QuantHook:
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
            ev_sorted = ev[order]
            V = vv[:, order].astype(np.float32)
            kappa = float(ev_sorted[0] / max(ev_sorted[-1], 1e-12))
            ph.append({
                'V': V, 'mean': mean.astype(np.float32),
                'eigvals': ev_sorted, 'K_pca': Kc @ V, 'kappa': kappa,
            })
        basis[li] = ph
    return basis


def fit_cents(basis, qtype, n_layers, n_kv, head_dim, bits=2):
    out = {}
    for li in range(n_layers):
        per = []
        for hk in range(n_kv):
            K_pca = basis[li][hk]['K_pca']
            cents = []
            for j in range(head_dim):
                if qtype == 'lloyd':
                    cents.append(lloyd_1d(K_pca[:, j], bits, 15))
                else:
                    cents.append(uniform_grid_1d(K_pca[:, j], bits))
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


def measure_attention_signature(model, ids, n_layers, n_kv, n_q, head_dim, basis):
    """Capture attention weights, compute pos0_attn over top-32 high-κ heads."""
    q_per_kv = n_q // n_kv
    with torch.no_grad():
        out = model(ids, use_cache=False, output_attentions=True)
    attn_weights = out.attentions  # tuple of (B, n_q, T, T)

    head_stats = []
    for li in range(n_layers):
        for hk in range(n_kv):
            head_stats.append({'layer': li, 'kv_head': hk, 'kappa': basis[li][hk]['kappa']})
    head_stats.sort(key=lambda x: -x['kappa'])
    top32 = head_stats[:32]

    for rec in top32:
        li = rec['layer']; hk = rec['kv_head']
        attn = attn_weights[li][0].float().cpu().numpy()  # (n_q, T, T)
        q_start = hk * q_per_kv
        q_end = q_start + q_per_kv
        attn_avg = attn[q_start:q_end].mean(axis=0).mean(axis=0)  # (T,)
        rec['attn_pos0'] = float(attn_avg[0])
        rec['attn_first4'] = float(attn_avg[:4].sum())

    pos0_attn = float(np.mean([r['attn_pos0'] for r in top32]))
    first4_attn = float(np.mean([r['attn_first4'] for r in top32]))
    kappa_max = float(max(r['kappa'] for r in head_stats))
    kappa_median = float(np.median([r['kappa'] for r in head_stats]))
    return {
        'pos0_attn': pos0_attn,
        'first4_attn': first4_attn,
        'kappa_max': kappa_max,
        'kappa_median': kappa_median,
        'top32_heads': top32,
        'n_total_heads': len(head_stats),
    }


def classify_mode(pos0_attn, kappa_max):
    if pos0_attn > 0.40 and kappa_max > 1e6:
        return 'A (positional sink)'
    if pos0_attn < 0.20 and kappa_max > 1e6:
        return 'B (distributed tail)'
    if kappa_max < 1e5:
        return 'C (bulk-tail)'
    return 'BOUNDARY (unclassified)'


def main():
    print("=" * 70)
    print("exp_next9: Llama-3.1-8B full mode + PPL (5th data point)")
    print("=" * 70, flush=True)
    t_start = time.time()

    print(f"\nLoading {MODEL_ID} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    # eager attention so output_attentions works for sig measurement
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_q = model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, 'head_dim', None) or (model.config.hidden_size // n_q)
    print(f"  loaded n_layers={n_layers} n_q={n_q} n_kv={n_kv} head_dim={head_dim} in {time.time()-t0:.1f}s", flush=True)

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in ds['text'] if len(t.strip()) > 100]
    calib = '\n\n'.join(texts[:300])
    eval_t = '\n\n'.join(texts[300:3000])
    calib_ids = tok(calib, return_tensors='pt', truncation=True, max_length=N_CALIB)['input_ids'].to('cuda:0')
    print(f"  calib tokens: {calib_ids.shape[1]}", flush=True)

    # Step 1: calibrate per-head PCA basis
    print("\n[1/4] Calibrating per-head Pre-RoPE PCA ...", flush=True)
    basis = calibrate(model, calib_ids, n_layers, n_kv, head_dim)
    n_heads = n_layers * n_kv
    print(f"  calibrated {n_heads} heads", flush=True)

    # Step 2: attention signature (uses output_attentions, length 2048)
    print("\n[2/4] Measuring attention signature (top-32 high-κ heads) ...", flush=True)
    sig = measure_attention_signature(model, calib_ids, n_layers, n_kv, n_q, head_dim, basis)
    mode = classify_mode(sig['pos0_attn'], sig['kappa_max'])
    print(f"  pos0_attn  = {sig['pos0_attn']*100:.2f}%", flush=True)
    print(f"  first4_attn= {sig['first4_attn']*100:.2f}%", flush=True)
    print(f"  κ_max      = {sig['kappa_max']:.2e}", flush=True)
    print(f"  κ_median   = {sig['kappa_median']:.2e}", flush=True)
    print(f"  → MODE     = {mode}", flush=True)

    # Step 3: fit centroids
    print("\n[3/4] Fitting Lloyd + Grid centroids ...", flush=True)
    cents_lloyd = fit_cents(basis, 'lloyd', n_layers, n_kv, head_dim, bits=2)
    cents_grid = fit_cents(basis, 'grid', n_layers, n_kv, head_dim, bits=2)
    print("  done", flush=True)

    # Step 4: PPL evaluation
    print("\n[4/4] Evaluating PPL @ FP16 + Lloyd/Grid × sink_k {0,1} ...", flush=True)
    fp16 = {}
    configs = {}
    for L in EVAL_LENGTHS:
        eval_ids = tok(eval_t, return_tensors='pt', truncation=True, max_length=L)['input_ids'].to('cuda:0')
        fp16[L] = compute_ppl(model, eval_ids)
        print(f"  FP16 @ L={L}: {fp16[L]:.4f}", flush=True)
        for qtype, cents in [('lloyd', cents_lloyd), ('grid', cents_grid)]:
            for sink_k in [0, 1]:
                handles = install(model, basis, cents, sink_k, n_layers, n_kv, head_dim)
                ppl = compute_ppl(model, eval_ids)
                for h in handles: h.remove()
                key = f'L{L}_{qtype}_sink{sink_k}'
                configs[key] = ppl
                print(f"  [L={L:>5} {qtype:<5} sink={sink_k}] PPL = {ppl:.4f}  (Δ = {ppl - fp16[L]:+.4f})", flush=True)

    # Save
    sig_serial = {k: v for k, v in sig.items() if k != 'top32_heads'}
    sig_serial['top32_heads'] = [
        {kk: float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else vv
         for kk, vv in r.items()} for r in sig['top32_heads']
    ]
    result = {
        'model': MODEL_ID,
        'short_name': SHORT_NAME,
        'n_layers': n_layers, 'n_q': n_q, 'n_kv': n_kv, 'head_dim': head_dim,
        'signature': sig_serial,
        'mode': mode,
        'fp16': fp16,
        'configs': configs,
        'thresholds': {'pos0_high': 0.40, 'pos0_low': 0.20,
                        'kappa_high': 1e6, 'kappa_low': 1e5},
    }
    out = OUT_DIR / 'exp_next9_llama_full.json'
    with open(out, 'w') as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nSaved: {out}", flush=True)

    # Final summary
    print("\n" + "=" * 70)
    print("LLAMA-3.1-8B SUMMARY (5th data point)")
    print("=" * 70)
    print(f"  pos0_attn = {sig['pos0_attn']*100:.2f}%   κ_max = {sig['kappa_max']:.2e}")
    print(f"  → MODE: {mode}")
    print(f"\n  PPL table:")
    print(f"  {'L':>6} | {'FP16':>7} | {'Lloyd':>7} | {'Lloyd+s1':>9} | {'Grid':>7} | {'Grid+s1':>8}")
    for L in EVAL_LENGTHS:
        print(f"  {L:>6} | {fp16[L]:>7.3f} | "
              f"{configs[f'L{L}_lloyd_sink0']:>7.3f} | "
              f"{configs[f'L{L}_lloyd_sink1']:>9.3f} | "
              f"{configs[f'L{L}_grid_sink0']:>7.3f} | "
              f"{configs[f'L{L}_grid_sink1']:>8.3f}")
    print(f"\nRuntime: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f}m)")


if __name__ == '__main__':
    main()
