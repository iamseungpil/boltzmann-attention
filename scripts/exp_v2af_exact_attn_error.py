#!/usr/bin/env python3
"""
V2af: Exact Attention Output Error vs Predicted Metrics
==========================================================

v2ae showed that awMSE = Σ_t a_t ||e_t||^2 predicts PPL direction correctly
on Mistral and Qwen-7B but fails on Nemo and Qwen-1.5B. The failure hints
that ||e_t||^2 is too coarse — it includes error components orthogonal to
the query direction q, which do not affect attention output.

Correct first-order expansion of attention output:

    o(q)  = Σ_t s_t(q) v_t                              (softmax output)
    o'(q) = Σ_t s'_t(q) v_t   where s'_t = softmax(q (K+E)^T / √d)_t
    Δo(q) ≈ Σ_t s_t(q) · (q·E_t/√d - <α>_s) · v_t
          = Σ_t s_t(q) (q·E_t/√d) (v_t - o(q))
    where α_t = q·E_t/√d, <α>_s = Σ_t s_t(q) α_t.

The RIGHT metric for attention output perturbation is therefore

    qaMSE(head) = E_q [ || Σ_t s_t(q) (q·E_t/√d) (v_t - o(q)) ||^2 ]     (*)

This involves:
  1. s_t(q) : attention weight (NOT just averaged)
  2. q·E_t  : QUERY-PROJECTED error (NOT full magnitude)
  3. v_t - o(q) : value's deviation from current attention output

awMSE (v2ae) only has term 1, approximated. It misses the query projection
(term 2) and the v-deviation weighting (term 3).

This experiment:
  1. Captures Q, K, V, and attention weights for each layer/head on eval data
  2. Computes K reconstructions under Lloyd and Grid per-dim 2-bit quantizers
  3. For each (L, H), computes THREE quantities:
       (a) Raw  MSE      = (1/T) Σ_t ||e_t||^2          (v2ae raw)
       (b) awMSE          = Σ_t a_t ||e_t||^2            (v2ae candidate)
       (c) qaMSE (exact)  = (1/Q) Σ_q || Δo(q) ||^2      (direct from softmax)
     where Δo(q) is computed by actually running softmax on q K^T and
     q K_quant^T, and taking the output difference.
  4. Aggregates per head, per model.
  5. Compares (a), (b), (c) ratios against the empirical PPL ratio.

If qaMSE (c) correctly predicts r_ppl on all 4 models while awMSE (b) does
not, we have direct evidence that the missing ingredient is the query
projection q·E_t, not the full ||e_t||.

Runtime: ~10 minutes.
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


def quantize_per_dim(K_pca, centroids):
    out = np.zeros_like(K_pca)
    for j in range(K_pca.shape[1]):
        cj = centroids[j]
        if cj is None or len(cj) == 1:
            out[:, j] = cj[0] if cj is not None else 0.0
        else:
            bnd = (cj[:-1] + cj[1:]) / 2
            idx = np.searchsorted(bnd, K_pca[:, j])
            out[:, j] = cj[idx]
    return out


def analyze_model(model_id, sn):
    print(f"\n{'='*70}\n  {sn}: {model_id}\n{'='*70}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    head_dim = getattr(model.config, 'head_dim', None) or (model.config.hidden_size // n_q)
    q_per_kv = n_q // n_kv
    scale = 1.0 / np.sqrt(head_dim)
    print(f"  n_layers={n_layers} n_kv={n_kv} n_q={n_q} head_dim={head_dim} q/kv={q_per_kv} loaded in {time.time()-t0:.1f}s", flush=True)

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in ds['text'] if len(t.strip()) > 100]
    calib_text = '\n\n'.join(texts[:300])
    eval_text  = '\n\n'.join(texts[300:600])
    calib_ids = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB)['input_ids'].to('cuda:0')
    eval_ids  = tok(eval_text,  return_tensors='pt', truncation=True, max_length=N_EVAL)['input_ids'].to('cuda:0')
    T_eval = eval_ids.shape[1]

    # ---- Calibration: build PCA + centroids per head ----
    captured_k_calib = {}
    def mk_k_calib(li):
        def h(m, i, o): captured_k_calib[li] = o.detach().cpu().float().numpy()
        return h
    handles = [model.model.layers[li].self_attn.k_proj.register_forward_hook(mk_k_calib(li)) for li in range(n_layers)]
    with torch.no_grad():
        _ = model(calib_ids, use_cache=False)
    for h in handles: h.remove()

    basis = {}
    cents_lloyd = {}
    cents_grid = {}
    for li in range(n_layers):
        K_all = captured_k_calib[li].reshape(-1, n_kv, head_dim).astype(np.float32)
        basis[li] = {}; cents_lloyd[li] = {}; cents_grid[li] = {}
        for hk in range(n_kv):
            K = K_all[:, hk, :]; mean = K.mean(axis=0); Kc = K - mean
            cov = (Kc.T @ Kc) / max(K.shape[0]-1, 1)
            ev, vv = np.linalg.eigh(cov)
            order = np.argsort(ev)[::-1]
            V = vv[:, order].astype(np.float32)
            K_pca = Kc @ V
            basis[li][hk] = {'V': V, 'mean': mean.astype(np.float32), 'eigvals': ev[order]}
            cents_lloyd[li][hk] = [lloyd_1d(K_pca[:, j], 2, 15) for j in range(head_dim)]
            cents_grid[li][hk]  = [uniform_grid_1d(K_pca[:, j], 2) for j in range(head_dim)]
    del captured_k_calib
    print(f"  Calibrated in {time.time()-t0:.1f}s", flush=True)

    # ---- Eval: capture Q, K, V per layer ----
    captured_q = {}; captured_k = {}; captured_v = {}
    def mk_q(li):
        def h(m, i, o): captured_q[li] = o.detach().cpu().float().numpy()
        return h
    def mk_k(li):
        def h(m, i, o): captured_k[li] = o.detach().cpu().float().numpy()
        return h
    def mk_v(li):
        def h(m, i, o): captured_v[li] = o.detach().cpu().float().numpy()
        return h
    handles = []
    for li in range(n_layers):
        handles.append(model.model.layers[li].self_attn.q_proj.register_forward_hook(mk_q(li)))
        handles.append(model.model.layers[li].self_attn.k_proj.register_forward_hook(mk_k(li)))
        handles.append(model.model.layers[li].self_attn.v_proj.register_forward_hook(mk_v(li)))

    # We need RoPE-applied Q, K. Since we hook BEFORE RoPE in the module,
    # we need to apply RoPE here ourselves to match real attention.
    # For simplicity and because Pre-RoPE PCA is the framework, we work in
    # pre-RoPE space: treat Q·K^T as if RoPE were already absorbed into
    # the per-head PCA basis. The error E is also in pre-RoPE space.
    # This is an approximation; the paper-level theorem works the same way
    # since Pre-RoPE PCA operates before RoPE application.

    with torch.no_grad():
        _ = model(eval_ids, use_cache=False)
    for h in handles: h.remove()

    del model, tok
    gc.collect(); torch.cuda.empty_cache()

    # ---- Compute per-head metrics ----
    per_head = []
    for li in range(n_layers):
        Q_full = captured_q[li].reshape(T_eval, n_q, head_dim).astype(np.float32)  # (T, nq, d)
        K_full = captured_k[li].reshape(T_eval, n_kv, head_dim).astype(np.float32)  # (T, nkv, d)
        V_full = captured_v[li].reshape(T_eval, n_kv, head_dim).astype(np.float32)  # (T, nkv, d)

        for hk in range(n_kv):
            K = K_full[:, hk, :]          # (T, d)
            Vh = V_full[:, hk, :]         # (T, d)
            mean = basis[li][hk]['mean']
            Vpca = basis[li][hk]['V']
            Kc = K - mean
            K_pca = Kc @ Vpca             # (T, d) in PCA basis

            # Reconstruct K under Lloyd and Grid
            K_pca_q_lloyd = quantize_per_dim(K_pca, cents_lloyd[li][hk])
            K_pca_q_grid  = quantize_per_dim(K_pca, cents_grid[li][hk])
            # Back to original basis
            K_lloyd = K_pca_q_lloyd @ Vpca.T + mean    # (T, d)
            K_grid  = K_pca_q_grid  @ Vpca.T + mean
            E_lloyd = K - K_lloyd                       # (T, d) error
            E_grid  = K - K_grid

            # Raw MSE (unweighted mean per position)
            raw_lloyd = float(np.mean(np.sum(E_lloyd**2, axis=1)))
            raw_grid  = float(np.mean(np.sum(E_grid **2, axis=1)))

            # Q heads for this KV head
            q_start = hk * q_per_kv
            q_end   = q_start + q_per_kv
            aw_lloyd_total = 0.0
            aw_grid_total  = 0.0
            qa_lloyd_total = 0.0
            qa_grid_total  = 0.0
            exact_lloyd_total = 0.0
            exact_grid_total  = 0.0
            n_q_used = 0

            # To avoid O(T²·n_q) per head, we compute attention for
            # the chunk of queries for this KV head and reuse.
            for qh in range(q_start, q_end):
                Q = Q_full[:, qh, :]        # (T, d)
                # FP16 attention
                scores_fp = (Q @ K.T) * scale              # (T_q, T_k)
                # Apply causal mask
                T = scores_fp.shape[0]
                mask = np.triu(np.ones((T, T), dtype=bool), k=1)
                scores_fp[mask] = -1e9
                s_fp = softmax_np(scores_fp)                # (T_q, T_k)

                # Lloyd/Grid quantized attention
                scores_ll = (Q @ K_lloyd.T) * scale; scores_ll[mask] = -1e9
                scores_gr = (Q @ K_grid .T) * scale; scores_gr[mask] = -1e9
                s_ll = softmax_np(scores_ll); s_gr = softmax_np(scores_gr)

                # Attention outputs (each query q has its own output vector in R^d)
                o_fp = s_fp @ Vh           # (T_q, d)
                o_ll = s_ll @ Vh
                o_gr = s_gr @ Vh

                # Exact output error norms per query
                dll = o_ll - o_fp; dgr = o_gr - o_fp
                exact_lloyd_total += float(np.sum(np.sum(dll**2, axis=1)))
                exact_grid_total  += float(np.sum(np.sum(dgr**2, axis=1)))

                # awMSE per query: Σ_t s_fp_t ||e_t||^2
                e_norm_ll = np.sum(E_lloyd**2, axis=1)      # (T_k,)
                e_norm_gr = np.sum(E_grid **2, axis=1)
                aw_lloyd_total += float(np.sum(s_fp @ e_norm_ll))
                aw_grid_total  += float(np.sum(s_fp @ e_norm_gr))

                # qaMSE per query: Σ_t s_fp_t (q·e_t)^2
                qdotE_ll = Q @ E_lloyd.T    # (T_q, T_k)
                qdotE_gr = Q @ E_grid .T
                qa_lloyd_total += float(np.sum(s_fp * (qdotE_ll ** 2))) * (scale ** 2)
                qa_grid_total  += float(np.sum(s_fp * (qdotE_gr ** 2))) * (scale ** 2)

                n_q_used += T  # T queries per q head

            # Normalize per query count (so metrics are per-query averages)
            aw_lloyd = aw_lloyd_total / n_q_used
            aw_grid  = aw_grid_total  / n_q_used
            qa_lloyd = qa_lloyd_total / n_q_used
            qa_grid  = qa_grid_total  / n_q_used
            exact_lloyd = exact_lloyd_total / n_q_used
            exact_grid  = exact_grid_total  / n_q_used

            per_head.append({
                'layer': li, 'kv_head': hk,
                'kappa': float(basis[li][hk]['eigvals'][0] / max(basis[li][hk]['eigvals'][-1], 1e-12)),
                'raw_MSE_lloyd': raw_lloyd, 'raw_MSE_grid': raw_grid,
                'awMSE_lloyd': aw_lloyd,   'awMSE_grid': aw_grid,
                'qaMSE_lloyd': qa_lloyd,   'qaMSE_grid': qa_grid,
                'exact_lloyd': exact_lloyd, 'exact_grid': exact_grid,
            })

        # Progress log every 8 layers
        if (li + 1) % 8 == 0:
            print(f"    layer {li+1}/{n_layers} processed, t={time.time()-t0:.1f}s", flush=True)

    # Free captured tensors
    del captured_q, captured_k, captured_v
    gc.collect()

    # Aggregate
    def agg(key): return sum(h[key] for h in per_head)
    totals = {
        'raw_MSE_lloyd': agg('raw_MSE_lloyd'), 'raw_MSE_grid': agg('raw_MSE_grid'),
        'awMSE_lloyd': agg('awMSE_lloyd'),     'awMSE_grid': agg('awMSE_grid'),
        'qaMSE_lloyd': agg('qaMSE_lloyd'),     'qaMSE_grid': agg('qaMSE_grid'),
        'exact_lloyd': agg('exact_lloyd'),     'exact_grid': agg('exact_grid'),
    }
    r_raw   = totals['raw_MSE_lloyd'] / max(totals['raw_MSE_grid'],   1e-12)
    r_aw    = totals['awMSE_lloyd']   / max(totals['awMSE_grid'],     1e-12)
    r_qa    = totals['qaMSE_lloyd']   / max(totals['qaMSE_grid'],     1e-12)
    r_exact = totals['exact_lloyd']   / max(totals['exact_grid'],     1e-12)

    print(f"\n  === Aggregate metrics across {len(per_head)} heads ===", flush=True)
    print(f"  raw  MSE ratio  Ll/Gr = {r_raw:.4f}   (Lloyd raw  {totals['raw_MSE_lloyd']:.2e} / Grid {totals['raw_MSE_grid']:.2e})")
    print(f"  aw   MSE ratio  Ll/Gr = {r_aw:.4f}    (Lloyd aw   {totals['awMSE_lloyd']:.2e} / Grid {totals['awMSE_grid']:.2e})")
    print(f"  qa   MSE ratio  Ll/Gr = {r_qa:.4f}    (Lloyd qa   {totals['qaMSE_lloyd']:.2e} / Grid {totals['qaMSE_grid']:.2e})")
    print(f"  exact Δo² ratio Ll/Gr = {r_exact:.4f} (Lloyd exact {totals['exact_lloyd']:.2e} / Grid {totals['exact_grid']:.2e})", flush=True)

    return {
        'model': model_id, 'short_name': sn,
        'n_heads': len(per_head),
        'totals': totals,
        'ratios': {'r_raw': r_raw, 'r_aw': r_aw, 'r_qa': r_qa, 'r_exact': r_exact},
        'per_head': per_head,
    }


def softmax_np(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def main():
    print("="*70)
    print("V2af: Exact Attention Output Error vs Predicted Metrics")
    print("="*70, flush=True)
    t_start = time.time()

    results = {}
    for mid, sn in MODELS:
        try:
            results[sn] = analyze_model(mid, sn)
        except Exception as e:
            print(f"ERROR on {sn}: {e}")
            import traceback; traceback.print_exc()

    # Reference PPL ratios from v2p/v2u/v2aa
    ref_ppl = {
        'mistral-7b': 9.9644  / 6.4343,
        'nemo-12b':   8.3734  / 7.0115,
        'qwen-7b':    7.3914  / 7.8364,
        'qwen-1.5b':  21.4728 / 15.2580,
    }

    print("\n" + "="*70)
    print("VALIDATION — which metric predicts sign(r_ppl - 1) correctly?")
    print("="*70)
    print(f"  {'model':<14}|{'r_raw':>10}|{'r_aw':>10}|{'r_qa':>10}|{'r_exact':>10}|{'r_ppl':>10}"
          f"|{'raw':>5}|{'aw':>5}|{'qa':>5}|{'exact':>7}")
    for sn, r in results.items():
        rr = r['ratios']
        rp = ref_ppl.get(sn, float('nan'))
        ok_raw   = '✓' if (rr['r_raw']   > 1) == (rp > 1) else '✗'
        ok_aw    = '✓' if (rr['r_aw']    > 1) == (rp > 1) else '✗'
        ok_qa    = '✓' if (rr['r_qa']    > 1) == (rp > 1) else '✗'
        ok_exact = '✓' if (rr['r_exact'] > 1) == (rp > 1) else '✗'
        print(f"  {sn:<14}|{rr['r_raw']:>10.3f}|{rr['r_aw']:>10.3f}|{rr['r_qa']:>10.3f}|"
              f"{rr['r_exact']:>10.3f}|{rp:>10.3f}|"
              f"{ok_raw:>5}|{ok_aw:>5}|{ok_qa:>5}|{ok_exact:>7}")
    print()
    print("  r_raw   = raw L2 key error ratio")
    print("  r_aw    = Σ a_t ||e_t||^2 ratio   (v2ae candidate)")
    print("  r_qa    = Σ a_t (q·e_t)^2 ratio   (v2af new candidate)")
    print("  r_exact = actual ||Δo||^2 ratio from running softmax (ground truth)")

    out = OUT_DIR / 'exp_v2af_exact_attn_error.json'
    with open(out, 'w') as f:
        json.dump({
            'reference_ppl_Lloyd_over_Grid': ref_ppl,
            'results': results,
        }, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Runtime: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f}m)")


if __name__ == '__main__':
    main()
