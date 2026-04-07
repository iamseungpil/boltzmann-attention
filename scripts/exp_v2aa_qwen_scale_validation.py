#!/usr/bin/env python3
"""
V2aa: Decision Rule Validation on Qwen Scale Variants
=======================================================

v2w+v2v2 derived a calibration-only decision rule:
  - Mode A (sink): pos-0 attention mass on top-κ heads > 40% → use Lloyd+sink_k=1
  - Mode B (tail): pos-0 attention mass on top-κ heads < 20% → use Uniform Grid

Currently only verified on 3 models (Mistral-7B, Nemo-12B, Qwen-7B). This
experiment adds 2 more scale variants — Qwen2.5-1.5B and Qwen2.5-14B — to
test whether the rule generalizes within a model family.

Per model:
  1. Calibrate (2048 tokens) and compute per-head κ
  2. Measure pos-0 attention mass on top-32 high-κ heads
  3. Run PPL test at L=2048 with 4 configs:
     - Lloyd sink=0, Lloyd sink=1, Grid sink=0, Grid sink=1
  4. Compare rule prediction vs empirical best

Goal: Confirm the rule generalizes across a 10× scale range (1.5B to 14B).
"""
import json, os, time, gc
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
    'Qwen/Qwen2.5-1.5B',
    'Qwen/Qwen2.5-14B',
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


def compute_ppl(model, ids):
    with torch.no_grad():
        out = model(ids, use_cache=False)
        logits = out.logits[:, :-1].contiguous()
        tgt = ids[:, 1:].contiguous()
        B, Tm1, V = logits.shape
        lf = logits.reshape(-1, V); tf = tgt.reshape(-1)
        total_loss = 0.0; total_n = 0
        CHUNK = 4096
        for s in range(0, lf.size(0), CHUNK):
            e = min(s + CHUNK, lf.size(0))
            l = lf[s:e].float()
            loss = F.cross_entropy(l, tf[s:e], reduction='sum')
            total_loss += float(loss.item())
            total_n += (e - s)
            del l
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
            ph.append({'V': V, 'mean': mean.astype(np.float32), 'eigvals': ev[order], 'K_pca': Kc @ V})
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


def run_model(model_id):
    print(f"\n{'='*70}\n  {model_id}\n{'='*70}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # Load TWO model instances: one with eager (for attention), one with sdpa (for PPL)
    # Actually, eager works for both — just slower. Use eager for simplicity.
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
    print(f"  n_layers={n_layers} n_kv={n_kv} n_q={n_q} head_dim={head_dim} q/kv={q_per_kv} loaded in {time.time()-t0:.1f}s")

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in ds['text'] if len(t.strip()) > 100]
    calib_text = '\n\n'.join(texts[:300])
    eval_text = '\n\n'.join(texts[300:600])
    calib_ids = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB)['input_ids'].to('cuda:0')
    eval_ids = tok(eval_text, return_tensors='pt', truncation=True, max_length=N_EVAL)['input_ids'].to('cuda:0')

    # Calibrate + capture attention in one pass
    pl = {}
    def mk(li):
        def h(m, i, o): pl[li] = o.detach().cpu().float().numpy()
        return h
    handles = [model.model.layers[li].self_attn.k_proj.register_forward_hook(mk(li)) for li in range(n_layers)]
    with torch.no_grad():
        out = model(calib_ids, use_cache=False, output_attentions=True)
    attn_weights = out.attentions
    for h in handles: h.remove()

    # Build basis
    basis = {}
    head_kappa = []
    for li in range(n_layers):
        K_all = pl[li].reshape(-1, n_kv, head_dim).astype(np.float32)
        ph = []
        for hk in range(n_kv):
            K = K_all[:, hk, :]; mean = K.mean(axis=0); Kc = K - mean
            cov = (Kc.T @ Kc) / max(K.shape[0]-1, 1)
            ev, vv = np.linalg.eigh(cov)
            order = np.argsort(ev)[::-1]
            V = vv[:, order].astype(np.float32)
            ev = ev[order]
            kappa = float(ev[0] / max(ev[-1], 1e-12))
            head_kappa.append({'layer': li, 'kv_head': hk, 'kappa': kappa})
            ph.append({'V': V, 'mean': mean.astype(np.float32), 'eigvals': ev, 'K_pca': Kc @ V})
        basis[li] = ph

    head_kappa.sort(key=lambda x: -x['kappa'])
    top32 = head_kappa[:min(32, len(head_kappa))]
    max_kappa = head_kappa[0]['kappa']

    # Measure pos0 attention mass on top-32 heads
    for rec in top32:
        li = rec['layer']; hk = rec['kv_head']
        attn = attn_weights[li][0].float().cpu().numpy()
        q_start = hk * q_per_kv
        q_end = q_start + q_per_kv
        attn_avg = attn[q_start:q_end].mean(axis=0).mean(axis=0)
        rec['attn_pos0'] = float(attn_avg[0])

    mean_pos0 = float(np.mean([r['attn_pos0'] for r in top32]))
    print(f"\n  Top-32 pos0 attention mass: mean={mean_pos0*100:.1f}%")
    print(f"  Max κ: {max_kappa:.1e}")

    # Apply decision rule
    if mean_pos0 > 0.4:
        predicted_mode = 'Mode-A (Lloyd+sink)'
    elif mean_pos0 > 0.2:
        predicted_mode = 'Mode-A-mild (Lloyd+sink)'
    else:
        predicted_mode = 'Mode-B (Grid)'
    print(f"  Predicted: {predicted_mode}", flush=True)

    # FP16 + 4 configs
    ppl_fp16 = compute_ppl(model, eval_ids)
    print(f"  FP16: {ppl_fp16:.4f}", flush=True)

    cents_lloyd = fit_cents(basis, 'lloyd', n_layers, n_kv, head_dim)
    cents_grid = fit_cents(basis, 'grid', n_layers, n_kv, head_dim)
    configs_ppl = {}
    for qtype, cents in [('lloyd', cents_lloyd), ('grid', cents_grid)]:
        for sink_k in [0, 1]:
            handles = install(model, basis, cents, sink_k, n_layers, n_kv, head_dim)
            ppl = compute_ppl(model, eval_ids)
            for h in handles: h.remove()
            key = f'{qtype}_sink{sink_k}'
            configs_ppl[key] = ppl
            print(f"  [{qtype:<5} sink={sink_k}] PPL={ppl:.4f}  Δ={ppl-ppl_fp16:+.4f}", flush=True)

    # Empirical best
    best_key = min(configs_ppl, key=lambda k: configs_ppl[k])
    if 'lloyd' in best_key:
        empirical_mode = 'Mode-A (Lloyd+sink)' if 'sink1' in best_key else 'Mode-A (Lloyd no sink)'
    else:
        empirical_mode = 'Mode-B (Grid)'

    prediction_correct = (('A' in predicted_mode) == ('A' in empirical_mode))
    print(f"\n  Empirical best: {best_key} → {empirical_mode}")
    print(f"  Prediction {'✓' if prediction_correct else '✗'}")

    del model, tok
    gc.collect(); torch.cuda.empty_cache()

    return {
        'model': model_id,
        'n_layers': n_layers,
        'n_kv': n_kv,
        'max_kappa': max_kappa,
        'mean_pos0_attn_top32': mean_pos0,
        'predicted_mode': predicted_mode,
        'empirical_best_config': best_key,
        'empirical_mode': empirical_mode,
        'prediction_correct': prediction_correct,
        'ppl_fp16': ppl_fp16,
        'ppl_configs': configs_ppl,
    }


def main():
    print("="*70)
    print("V2aa: Decision Rule Validation on Qwen Scale Variants")
    print("="*70, flush=True)
    t_start = time.time()

    results = {}
    for mid in MODELS:
        try:
            sn = mid.split('/')[-1].lower()
            results[sn] = run_model(mid)
        except Exception as e:
            print(f"ERROR on {mid}: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"  {'model':<20}|{'max_κ':>10}|{'pos0%':>8}|{'predicted':<22}|{'empirical':<22}|{'OK':>3}")
    for sn, r in results.items():
        print(f"  {sn:<20}|{r['max_kappa']:>10.1e}|{r['mean_pos0_attn_top32']*100:>7.1f}%|"
              f"{r['predicted_mode']:<22}|{r['empirical_mode']:<22}|{'✓' if r['prediction_correct'] else '✗':>3}")

    out = OUT_DIR / 'exp_v2aa_qwen_validation.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Runtime: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f}m)")


if __name__ == '__main__':
    main()
