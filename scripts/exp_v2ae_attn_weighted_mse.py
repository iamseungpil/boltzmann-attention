#!/usr/bin/env python3
"""
V2ae: Attention-Weighted Reconstruction Error — Validates the Candidate Theorem
================================================================================

Tests whether attention-weighted MSE (awMSE) predicts PPL better than raw MSE
across the three failure modes identified in v2ad.

Candidate theorem (from V2_THEORY_AND_3MODES report, Section 3.2):

    PPL degradation ~  E_q [ sum_t a_t(q) * ||e_t||^2 ]
                     =  E[||e_t||^2]  +  Cov_t(a_t, ||e_t||^2)
                        (raw MSE)        (attention-error coupling)

Claim: the covariance term is what makes Lloyd fail on Mode A/B at 2-bit
despite being raw-MSE-optimal. Grid has higher raw MSE but lower Cov because
its L∞ bound caps ||e_t||^2 uniformly.

Protocol per model:
  1. Calibrate PCA basis + fit Lloyd/Grid per-dim centroids at 2 bits.
  2. Forward on eval data (L=2048) with attention output.
  3. For each (layer, head), compute per-position:
       e_t^{Lloyd}  = || k_t - reconstruct_Lloyd(k_t) ||^2
       e_t^{Grid}   = || k_t - reconstruct_Grid (k_t) ||^2
       a_t          = avg attention mass received, across queries
  4. Compute four scalar metrics per (layer, head):
       raw_MSE_X     = mean_t e_t^X
       awMSE_X       = sum_t a_t * e_t^X  (note: sum_t a_t is roughly T*1/T = 1)
       max_ae_X      = max_t (a_t * e_t^X)
       cov_X         = Cov_t(a_t, e_t^X)
  5. Aggregate by sum/median across all (L, H).
  6. Compare ratios Lloyd/Grid against the empirical PPL ratio from v2p/v2u.

  r_raw = raw_MSE_Lloyd / raw_MSE_Grid
  r_aw  = awMSE_Lloyd  / awMSE_Grid
  r_ppl = PPL_Lloyd    / PPL_Grid

Prediction (from candidate theorem):
  Mode A/B:  r_raw < 1 (Lloyd raw-better) but r_ppl > 1 (Lloyd worse).
             r_aw should track r_ppl (r_aw > 1) → awMSE correctly predicts.
  Mode C:    All three ratios should be consistent (≈ 1 or all < 1 or all > 1).

If the prediction holds across all 4 models, the candidate theorem is
empirically validated.

GPU: 0 (uses eager attention; single model at a time).
"""
import json, os, time, gc
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
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


def quantize_per_dim(K_pca, centroids_per_dim):
    """Apply per-dim scalar quantizer. K_pca: (T, d). centroids_per_dim: list of 1D arrays."""
    out = np.zeros_like(K_pca)
    for j in range(K_pca.shape[1]):
        cj = centroids_per_dim[j]
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
        output_attentions=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    head_dim = getattr(model.config, 'head_dim', None) or (model.config.hidden_size // n_q)
    q_per_kv = n_q // n_kv
    print(f"  n_layers={n_layers} n_kv={n_kv} n_q={n_q} head_dim={head_dim} q/kv={q_per_kv} loaded in {time.time()-t0:.1f}s", flush=True)

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in ds['text'] if len(t.strip()) > 100]
    calib_text = '\n\n'.join(texts[:300])
    eval_text = '\n\n'.join(texts[300:600])
    calib_ids = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB)['input_ids'].to('cuda:0')
    eval_ids = tok(eval_text, return_tensors='pt', truncation=True, max_length=N_EVAL)['input_ids'].to('cuda:0')
    T_eval = eval_ids.shape[1]
    print(f"  Calib T={calib_ids.shape[1]}, Eval T={T_eval}", flush=True)

    # --- Calibration: capture K to build PCA + Lloyd/Grid centroids ---
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
        basis[li] = {}
        cents_lloyd[li] = {}
        cents_grid[li] = {}
        for hk in range(n_kv):
            K = K_all[:, hk, :]
            mean = K.mean(axis=0); Kc = K - mean
            cov = (Kc.T @ Kc) / max(K.shape[0]-1, 1)
            ev, vv = np.linalg.eigh(cov)
            order = np.argsort(ev)[::-1]
            V = vv[:, order].astype(np.float32)
            K_pca = Kc @ V
            basis[li][hk] = {'V': V, 'mean': mean.astype(np.float32),
                              'eigvals': ev[order]}
            # Per-dim 2-bit centroids
            cl = [lloyd_1d(K_pca[:, j], 2, 15) for j in range(head_dim)]
            cg = [uniform_grid_1d(K_pca[:, j], 2) for j in range(head_dim)]
            cents_lloyd[li][hk] = cl
            cents_grid[li][hk] = cg
    del captured_k_calib
    print(f"  Calibrated in {time.time()-t0:.1f}s", flush=True)

    # --- Eval: capture K + attention weights ---
    captured_k_eval = {}
    def mk_k_eval(li):
        def h(m, i, o): captured_k_eval[li] = o.detach().cpu().float().numpy()
        return h
    handles = [model.model.layers[li].self_attn.k_proj.register_forward_hook(mk_k_eval(li)) for li in range(n_layers)]
    with torch.no_grad():
        out = model(eval_ids, use_cache=False, output_attentions=True)
    attn_weights = out.attentions  # tuple of (1, n_q, T, T) per layer
    for h in handles: h.remove()

    del model, tok
    gc.collect(); torch.cuda.empty_cache()

    # --- Compute per-head metrics ---
    per_head = []
    for li in range(n_layers):
        K_all = captured_k_eval[li].reshape(-1, n_kv, head_dim).astype(np.float32)
        attn_full = attn_weights[li][0].float().cpu().numpy()  # (n_q, T, T)
        for hk in range(n_kv):
            K = K_all[:, hk, :]
            V = basis[li][hk]['V']; mean = basis[li][hk]['mean']
            Kc = K - mean
            K_pca = Kc @ V  # (T, d)

            # Lloyd reconstruction in PCA space
            K_pca_q_lloyd = quantize_per_dim(K_pca, cents_lloyd[li][hk])
            err_pca_lloyd = K_pca - K_pca_q_lloyd  # (T, d)
            # Error magnitude squared per position (in PCA space == L2 norm since V is orthogonal)
            e_t_lloyd = np.sum(err_pca_lloyd ** 2, axis=1)  # (T,)

            # Grid reconstruction
            K_pca_q_grid = quantize_per_dim(K_pca, cents_grid[li][hk])
            err_pca_grid = K_pca - K_pca_q_grid
            e_t_grid = np.sum(err_pca_grid ** 2, axis=1)  # (T,)

            # Attention mass received at each position (averaged over queries for this KV head)
            q_start = hk * q_per_kv
            q_end = q_start + q_per_kv
            attn_avg = attn_full[q_start:q_end].mean(axis=0)  # (T, T) query → key
            a_t = attn_avg.mean(axis=0)  # (T,) average attention received per key position
            # (sum of a_t over t should be ≈ 1 since each query row sums to 1)

            T = a_t.shape[0]

            # Metrics
            raw_MSE_lloyd = float(np.mean(e_t_lloyd))
            raw_MSE_grid  = float(np.mean(e_t_grid))
            awMSE_lloyd   = float(np.sum(a_t * e_t_lloyd))
            awMSE_grid    = float(np.sum(a_t * e_t_grid))
            max_ae_lloyd  = float(np.max(a_t * e_t_lloyd))
            max_ae_grid   = float(np.max(a_t * e_t_grid))
            cov_lloyd     = float(np.mean((a_t - a_t.mean()) * (e_t_lloyd - e_t_lloyd.mean())))
            cov_grid      = float(np.mean((a_t - a_t.mean()) * (e_t_grid  - e_t_grid.mean())))

            per_head.append({
                'layer': li, 'kv_head': hk,
                'kappa_top_over_min': float(basis[li][hk]['eigvals'][0] / max(basis[li][hk]['eigvals'][-1], 1e-12)),
                'raw_MSE_lloyd': raw_MSE_lloyd, 'raw_MSE_grid': raw_MSE_grid,
                'awMSE_lloyd': awMSE_lloyd,    'awMSE_grid': awMSE_grid,
                'max_ae_lloyd': max_ae_lloyd,  'max_ae_grid': max_ae_grid,
                'cov_lloyd': cov_lloyd,         'cov_grid': cov_grid,
                'pos0_attn': float(a_t[0]),
            })

    # --- Aggregate model-level metrics ---
    total_raw_lloyd = sum(h['raw_MSE_lloyd'] for h in per_head)
    total_raw_grid  = sum(h['raw_MSE_grid']  for h in per_head)
    total_aw_lloyd  = sum(h['awMSE_lloyd']   for h in per_head)
    total_aw_grid   = sum(h['awMSE_grid']    for h in per_head)
    total_max_ae_lloyd = sum(h['max_ae_lloyd'] for h in per_head)
    total_max_ae_grid  = sum(h['max_ae_grid']  for h in per_head)
    total_cov_lloyd = sum(h['cov_lloyd'] for h in per_head)
    total_cov_grid  = sum(h['cov_grid']  for h in per_head)

    r_raw = total_raw_lloyd / max(total_raw_grid, 1e-12)
    r_aw  = total_aw_lloyd  / max(total_aw_grid,  1e-12)
    r_max_ae = total_max_ae_lloyd / max(total_max_ae_grid, 1e-12)
    r_cov = total_cov_lloyd / max(total_cov_grid, 1e-12) if total_cov_grid > 1e-12 else float('nan')

    print(f"\n  === Aggregate metrics across {len(per_head)} heads ===", flush=True)
    print(f"  Total raw MSE        : Lloyd={total_raw_lloyd:.3e}  Grid={total_raw_grid:.3e}  ratio Ll/Gr = {r_raw:.3f}")
    print(f"  Total awMSE          : Lloyd={total_aw_lloyd:.3e}  Grid={total_aw_grid:.3e}  ratio Ll/Gr = {r_aw:.3f}")
    print(f"  Total max (a·e²)     : Lloyd={total_max_ae_lloyd:.3e}  Grid={total_max_ae_grid:.3e}  ratio Ll/Gr = {r_max_ae:.3f}")
    print(f"  Total Cov(a, e²)     : Lloyd={total_cov_lloyd:.3e}  Grid={total_cov_grid:.3e}", flush=True)

    # --- Top heads by awMSE_Lloyd vs awMSE_Grid difference ---
    per_head_sorted = sorted(per_head, key=lambda h: -(h['awMSE_lloyd'] - h['awMSE_grid']))
    print(f"\n  Top-10 heads where Lloyd is MUCH worse than Grid (awMSE):", flush=True)
    print(f"  {'L':<4}{'H':<3}{'κ':>10}{'pos0':>10}{'raw_Ll':>12}{'raw_Gr':>12}{'aw_Ll':>12}{'aw_Gr':>12}", flush=True)
    for h in per_head_sorted[:10]:
        print(f"  {h['layer']:<4}{h['kv_head']:<3}{h['kappa_top_over_min']:>10.1e}"
              f"{h['pos0_attn']:>10.4f}{h['raw_MSE_lloyd']:>12.3e}{h['raw_MSE_grid']:>12.3e}"
              f"{h['awMSE_lloyd']:>12.3e}{h['awMSE_grid']:>12.3e}", flush=True)

    return {
        'model': model_id, 'short_name': sn,
        'n_heads': len(per_head),
        'totals': {
            'raw_MSE_lloyd': total_raw_lloyd, 'raw_MSE_grid': total_raw_grid,
            'awMSE_lloyd': total_aw_lloyd, 'awMSE_grid': total_aw_grid,
            'max_ae_lloyd': total_max_ae_lloyd, 'max_ae_grid': total_max_ae_grid,
            'cov_lloyd': total_cov_lloyd, 'cov_grid': total_cov_grid,
        },
        'ratios': {
            'r_raw': r_raw, 'r_aw': r_aw, 'r_max_ae': r_max_ae, 'r_cov': r_cov,
        },
        'per_head': per_head,
    }


def main():
    print("="*70)
    print("V2ae: Attention-Weighted Reconstruction Error")
    print("="*70, flush=True)
    t_start = time.time()

    results = {}
    for mid, sn in MODELS:
        try:
            results[sn] = analyze_model(mid, sn)
        except Exception as e:
            print(f"ERROR on {sn}: {e}")
            import traceback; traceback.print_exc()

    # --- Reference PPL ratios from v2p/v2u (at L=2048) ---
    ref_ppl_ratios = {
        'mistral-7b':  9.9644  / 6.4343,   # Lloyd_nosink / Grid_nosink from v2p
        'nemo-12b':    8.3734  / 7.0115,   # v2u
        'qwen-7b':     7.3914  / 7.8364,   # v2u
        'qwen-1.5b':   21.4728 / 15.2580,  # v2aa
    }

    print("\n" + "="*70)
    print("VALIDATION TABLE — Does awMSE predict PPL direction better than raw MSE?")
    print("="*70)
    print(f"  {'model':<14}|{'r_raw':>10}|{'r_aw':>10}|{'r_max_ae':>10}|{'r_ppl':>10}|{'raw OK?':>9}|{'aw OK?':>9}")
    for sn, r in results.items():
        rr = r['ratios']
        r_ppl = ref_ppl_ratios.get(sn, float('nan'))
        # A metric "correctly predicts" if sign(r_metric - 1) == sign(r_ppl - 1)
        raw_ok = (rr['r_raw'] > 1) == (r_ppl > 1)
        aw_ok = (rr['r_aw'] > 1) == (r_ppl > 1)
        print(f"  {sn:<14}|{rr['r_raw']:>10.3f}|{rr['r_aw']:>10.3f}|"
              f"{rr['r_max_ae']:>10.3f}|{r_ppl:>10.3f}|"
              f"{'✓' if raw_ok else '✗':>9}|{'✓' if aw_ok else '✗':>9}")
    print()
    print("  (r > 1 means Lloyd is worse than Grid on that metric)")
    print("  r_raw is misleading when Lloyd is raw-MSE-optimal but PPL worse")
    print("  r_aw should track r_ppl if the candidate theorem holds")

    out = OUT_DIR / 'exp_v2ae_attn_weighted_mse.json'
    with open(out, 'w') as f:
        json.dump({
            'reference_ppl_ratios_Lloyd_over_Grid': ref_ppl_ratios,
            'results': results,
        }, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Runtime: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f}m)")


if __name__ == '__main__':
    main()
