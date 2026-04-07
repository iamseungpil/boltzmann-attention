#!/usr/bin/env python3
"""
V2af (GPU): Exact Attention Output Error vs Predicted Metrics — torch/GPU rewrite.

Identical scientific protocol to exp_v2af_exact_attn_error.py but uses
torch on GPU for the attention matmul loop. 50x speedup.

Computes per (layer, kv_head) on each model:
  raw_MSE  : (1/T) Σ_t ||e_t||^2
  awMSE    : Σ_t a_bar_t ||e_t||^2   where a_bar_t = mean_q s_t(q)
  qaMSE    : (1/Q) Σ_q Σ_t s_t(q) (q·e_t)^2         (query-projected, v2af new)
  exact    : (1/Q) Σ_q ||Δo(q)||^2                  (ground truth via running softmax)

Compares Lloyd/Grid ratio for each against reference PPL ratio from v2p/v2u.
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


def quantize_per_dim_torch(K_pca: torch.Tensor, centroids_list) -> torch.Tensor:
    """K_pca: (T, d) on GPU. centroids_list: list of 1D arrays (numpy).
    Returns (T, d) on GPU."""
    device = K_pca.device
    dtype = K_pca.dtype
    T, d = K_pca.shape
    out = torch.zeros_like(K_pca)
    for j in range(d):
        cj = centroids_list[j]
        if cj is None or len(cj) == 1:
            val = float(cj[0]) if cj is not None else 0.0
            out[:, j] = val
        else:
            cj_t = torch.tensor(cj, device=device, dtype=dtype)
            bnd = (cj_t[:-1] + cj_t[1:]) / 2
            idx = torch.searchsorted(bnd, K_pca[:, j].contiguous())
            out[:, j] = cj_t[idx]
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
    scale = 1.0 / (head_dim ** 0.5)
    print(f"  n_layers={n_layers} n_kv={n_kv} n_q={n_q} head_dim={head_dim} q/kv={q_per_kv} loaded in {time.time()-t0:.1f}s", flush=True)

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in ds['text'] if len(t.strip()) > 100]
    calib_text = '\n\n'.join(texts[:300])
    eval_text  = '\n\n'.join(texts[300:600])
    calib_ids = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB)['input_ids'].to('cuda:0')
    eval_ids  = tok(eval_text,  return_tensors='pt', truncation=True, max_length=N_EVAL)['input_ids'].to('cuda:0')
    T_eval = eval_ids.shape[1]

    # ---- Calibration: PCA + Lloyd/Grid ----
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
            basis[li][hk] = {'V': V, 'mean': mean.astype(np.float32),
                              'eigvals': ev[order]}
            cents_lloyd[li][hk] = [lloyd_1d(K_pca[:, j], 2, 15) for j in range(head_dim)]
            cents_grid[li][hk]  = [uniform_grid_1d(K_pca[:, j], 2) for j in range(head_dim)]
    del captured_k_calib
    print(f"  Calibrated in {time.time()-t0:.1f}s", flush=True)

    # ---- Eval: capture Q, K, V tensors directly as GPU tensors ----
    captured_q = {}; captured_k = {}; captured_v = {}
    def mk_q(li):
        def h(m, i, o): captured_q[li] = o.detach().clone()  # (1, T, n_q*d)
        return h
    def mk_k(li):
        def h(m, i, o): captured_k[li] = o.detach().clone()
        return h
    def mk_v(li):
        def h(m, i, o): captured_v[li] = o.detach().clone()
        return h
    handles = []
    for li in range(n_layers):
        handles.append(model.model.layers[li].self_attn.q_proj.register_forward_hook(mk_q(li)))
        handles.append(model.model.layers[li].self_attn.k_proj.register_forward_hook(mk_k(li)))
        handles.append(model.model.layers[li].self_attn.v_proj.register_forward_hook(mk_v(li)))
    with torch.no_grad():
        _ = model(eval_ids, use_cache=False)
    for h in handles: h.remove()

    del model, tok
    gc.collect(); torch.cuda.empty_cache()

    device = 'cuda:0'
    # Build causal mask on GPU (do it once)
    mask = torch.triu(torch.ones(T_eval, T_eval, dtype=torch.bool, device=device), diagonal=1)

    per_head = []
    for li in range(n_layers):
        Q_full = captured_q[li].view(T_eval, n_q, head_dim).to(torch.float32)    # (T, nq, d)
        K_full = captured_k[li].view(T_eval, n_kv, head_dim).to(torch.float32)   # (T, nkv, d)
        V_full = captured_v[li].view(T_eval, n_kv, head_dim).to(torch.float32)   # (T, nkv, d)

        for hk in range(n_kv):
            K = K_full[:, hk, :]      # (T, d) torch GPU
            Vh = V_full[:, hk, :]     # (T, d) torch GPU

            # Build K_pca on GPU then quantize
            V_pca = torch.from_numpy(basis[li][hk]['V']).to(device)              # (d, d)
            mean = torch.from_numpy(basis[li][hk]['mean']).to(device)            # (d,)
            Kc = K - mean                                                        # (T, d)
            K_pca = Kc @ V_pca                                                   # (T, d)

            # Quantize per-dim
            K_pca_q_lloyd = quantize_per_dim_torch(K_pca, cents_lloyd[li][hk])
            K_pca_q_grid  = quantize_per_dim_torch(K_pca, cents_grid[li][hk])

            K_lloyd = K_pca_q_lloyd @ V_pca.T + mean
            K_grid  = K_pca_q_grid  @ V_pca.T + mean
            E_lloyd = K - K_lloyd                                                # (T, d)
            E_grid  = K - K_grid

            # Raw MSE
            e_norm_ll = torch.sum(E_lloyd**2, dim=1)                             # (T,)
            e_norm_gr = torch.sum(E_grid **2, dim=1)
            raw_lloyd = float(e_norm_ll.mean().item())
            raw_grid  = float(e_norm_gr.mean().item())

            # Q heads for this KV head: batch all q_per_kv queries together
            q_start = hk * q_per_kv
            Q_block = Q_full[:, q_start:q_start+q_per_kv, :]                     # (T, q_per_kv, d)
            # Flatten to (q_per_kv*T, d) as the "query set"
            Q_batch = Q_block.permute(1, 0, 2).reshape(-1, head_dim)             # (q_per_kv*T, d)

            # Attention scores for FP16 K
            scores_fp = Q_batch @ K.T * scale                                    # (q_per_kv*T, T)
            scores_ll = Q_batch @ K_lloyd.T * scale
            scores_gr = Q_batch @ K_grid .T * scale

            # Apply causal mask: we need the mask to repeat for each q-head
            # Each Q_batch row i corresponds to (q_head qh, query position t);
            # the mask should be: key position k > t → masked.
            # Build a causal mask over (q_per_kv*T, T)
            causal = mask.unsqueeze(0).expand(q_per_kv, T_eval, T_eval).reshape(-1, T_eval)  # (q_per_kv*T, T)
            scores_fp = scores_fp.masked_fill(causal, -1e9)
            scores_ll = scores_ll.masked_fill(causal, -1e9)
            scores_gr = scores_gr.masked_fill(causal, -1e9)

            s_fp = torch.softmax(scores_fp, dim=-1)                              # (q_per_kv*T, T)
            s_ll = torch.softmax(scores_ll, dim=-1)
            s_gr = torch.softmax(scores_gr, dim=-1)

            # Attention outputs (each query row becomes a d_v vector)
            o_fp = s_fp @ Vh                                                      # (q_per_kv*T, d)
            o_ll = s_ll @ Vh
            o_gr = s_gr @ Vh

            exact_lloyd = float(torch.sum((o_ll - o_fp)**2).item() / s_fp.shape[0])
            exact_grid  = float(torch.sum((o_gr - o_fp)**2).item() / s_fp.shape[0])

            # awMSE: Σ_t s_fp_t ||e_t||^2 averaged over queries
            # = (s_fp @ e_norm) mean
            aw_lloyd = float((s_fp @ e_norm_ll).mean().item())
            aw_grid  = float((s_fp @ e_norm_gr).mean().item())

            # qaMSE: (1/Q) Σ_q Σ_t s_t(q) (q·e_t)^2 * (1/d)
            # q·e_t as (q_per_kv*T, T): Q_batch @ E_x.T
            qdotE_ll = Q_batch @ E_lloyd.T * scale   # apply 1/√d
            qdotE_gr = Q_batch @ E_grid .T * scale
            qa_lloyd = float((s_fp * (qdotE_ll ** 2)).sum(dim=-1).mean().item())
            qa_grid  = float((s_fp * (qdotE_gr ** 2)).sum(dim=-1).mean().item())

            per_head.append({
                'layer': li, 'kv_head': hk,
                'kappa': float(basis[li][hk]['eigvals'][0] / max(basis[li][hk]['eigvals'][-1], 1e-12)),
                'raw_MSE_lloyd': raw_lloyd, 'raw_MSE_grid': raw_grid,
                'awMSE_lloyd': aw_lloyd,   'awMSE_grid': aw_grid,
                'qaMSE_lloyd': qa_lloyd,   'qaMSE_grid': qa_grid,
                'exact_lloyd': exact_lloyd, 'exact_grid': exact_grid,
            })

            del scores_fp, scores_ll, scores_gr, s_fp, s_ll, s_gr
            del o_fp, o_ll, o_gr, qdotE_ll, qdotE_gr
            torch.cuda.empty_cache()

        # Clean up captured tensors for this layer
        del captured_q[li], captured_k[li], captured_v[li]
        if (li + 1) % 8 == 0:
            print(f"    layer {li+1}/{n_layers} processed, t={time.time()-t0:.1f}s", flush=True)

    # Aggregate
    def agg(k): return sum(h[k] for h in per_head)
    totals = {
        'raw_MSE_lloyd': agg('raw_MSE_lloyd'), 'raw_MSE_grid': agg('raw_MSE_grid'),
        'awMSE_lloyd': agg('awMSE_lloyd'),     'awMSE_grid': agg('awMSE_grid'),
        'qaMSE_lloyd': agg('qaMSE_lloyd'),     'qaMSE_grid': agg('qaMSE_grid'),
        'exact_lloyd': agg('exact_lloyd'),     'exact_grid': agg('exact_grid'),
    }
    r_raw   = totals['raw_MSE_lloyd'] / max(totals['raw_MSE_grid'], 1e-12)
    r_aw    = totals['awMSE_lloyd']   / max(totals['awMSE_grid'],   1e-12)
    r_qa    = totals['qaMSE_lloyd']   / max(totals['qaMSE_grid'],   1e-12)
    r_exact = totals['exact_lloyd']   / max(totals['exact_grid'],   1e-12)

    print(f"\n  === Aggregate over {len(per_head)} heads ===")
    print(f"  r_raw   = {r_raw:.4f}  (Ll {totals['raw_MSE_lloyd']:.2e} / Gr {totals['raw_MSE_grid']:.2e})")
    print(f"  r_aw    = {r_aw:.4f}   (Ll {totals['awMSE_lloyd']:.2e} / Gr {totals['awMSE_grid']:.2e})")
    print(f"  r_qa    = {r_qa:.4f}   (Ll {totals['qaMSE_lloyd']:.2e} / Gr {totals['qaMSE_grid']:.2e})")
    print(f"  r_exact = {r_exact:.4f} (Ll {totals['exact_lloyd']:.2e} / Gr {totals['exact_grid']:.2e})", flush=True)

    return {
        'model': model_id, 'short_name': sn,
        'n_heads': len(per_head),
        'totals': totals,
        'ratios': {'r_raw': r_raw, 'r_aw': r_aw, 'r_qa': r_qa, 'r_exact': r_exact},
        'per_head': per_head,
    }


def main():
    print("="*70)
    print("V2af (GPU): Exact Attention Output Error vs Predicted Metrics")
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
    print("VALIDATION")
    print("="*70)
    print(f"  {'model':<14}|{'r_raw':>10}|{'r_aw':>10}|{'r_qa':>10}|{'r_exact':>10}|{'r_ppl':>10}"
          f"|{'raw':>5}|{'aw':>5}|{'qa':>5}|{'exact':>7}")
    for sn, r in results.items():
        rr = r['ratios']
        rp = ref_ppl.get(sn, float('nan'))
        ok = lambda m: '✓' if (rr[m] > 1) == (rp > 1) else '✗'
        print(f"  {sn:<14}|{rr['r_raw']:>10.3f}|{rr['r_aw']:>10.3f}|{rr['r_qa']:>10.3f}|"
              f"{rr['r_exact']:>10.3f}|{rp:>10.3f}|"
              f"{ok('r_raw'):>5}|{ok('r_aw'):>5}|{ok('r_qa'):>5}|{ok('r_exact'):>7}")

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
