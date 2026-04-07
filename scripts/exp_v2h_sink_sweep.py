#!/usr/bin/env python3
"""
V2h: Sink Protection Sweep — How Many Tokens, and Do We Still Need WF?
========================================================================

v2f [E] finding: FP16 first-4 tokens + WF 2-bit → PPL = 5.533 (+0.145 vs FP16).

This almost closes the 2-bit gap. Open questions:

Q1: Is WF even needed, or is uniform 2-bit + sink protection enough?
    → Test [F] = FP16 first-4 + uniform 2-bit
Q2: How many sink tokens are needed? 1, 2, 4, 8, 16?
    → Sweep K ∈ {1, 2, 4, 8, 16} with both uniform and WF
Q3: Does sink protection work on other models?
    → Test Mistral-Nemo (has enrichment 6.12×) and Qwen (has 0 aligned heads)
    → Expected: Nemo needs sinks too; Qwen uniform 2-bit already fine
Q4: What if we protect sinks WITHOUT per-head PCA (back to KVTC-like flat
    quantization)?
    → Test simple channel-level sink protection

Runtime: ~10 min total
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


class SinkHook:
    """Per-head PCA + per-dim Lloyd + FP16 sink token protection."""
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
        # Sink protection
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


def fit_cents(basis, scheme, n_layers, n_kv, head_dim):
    out = {}
    for li in range(n_layers):
        per = []
        for hk in range(n_kv):
            bh = basis[li][hk]
            ev = bh['eigvals']; K_pca = bh['K_pca']
            if scheme == 'uniform2':
                bits = np.full(head_dim, 2, dtype=int)
            elif scheme == 'wf2':
                bits = wf_alloc(ev, 2*head_dim, 1, 8)
            else:
                raise ValueError(scheme)
            cents = []
            for j in range(head_dim):
                b = int(bits[j])
                cents.append(lloyd_1d(K_pca[:, j], b, 15) if b > 0 else np.array([0.0], dtype=np.float32))
            per.append(cents)
        out[li] = per
    return out


def install(model, basis, cents, sink_k, n_layers, n_kv, head_dim):
    handles = []
    for li in range(n_layers):
        V_list = [basis[li][hk]['V'] for hk in range(n_kv)]
        mean_list = [basis[li][hk]['mean'] for hk in range(n_kv)]
        hook = SinkHook(n_kv, head_dim, V_list, mean_list, cents[li], sink_k)
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
    head_dim = model.config.hidden_size // model.config.num_attention_heads
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

    result = {'model': model_id, 'ppl_fp16': ppl_fp16, 'sweeps': {}}

    # Pre-fit centroids for both schemes (reused across sink-k)
    print("  Fitting uniform2 cents...", flush=True)
    cents_u = fit_cents(basis, 'uniform2', n_layers, n_kv, head_dim)
    print("  Fitting wf2 cents...", flush=True)
    cents_w = fit_cents(basis, 'wf2', n_layers, n_kv, head_dim)

    # Sink-k sweep
    for sink_k in [0, 1, 2, 4, 8, 16]:
        for scheme, cents in [('uniform', cents_u), ('wf', cents_w)]:
            handles = install(model, basis, cents, sink_k, n_layers, n_kv, head_dim)
            ppl = compute_ppl(model, eval_ids)
            for h in handles: h.remove()
            result['sweeps'][f'{scheme}_sink{sink_k}'] = ppl
            print(f"  [{scheme:<7} sink_k={sink_k:>2}] PPL = {ppl:.4f}  (Δ = {ppl-ppl_fp16:+.4f})", flush=True)

    # Cleanup
    del model, tok, basis, cents_u, cents_w
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    print("="*70)
    print("V2h: Sink Protection Sweep + Cross-Model")
    print("="*70, flush=True)
    t_start = time.time()

    results = {}
    for mid in MODELS:
        try:
            sn = mid.split('/')[-1].replace('Base-', '').lower()
            results[sn] = run_model(mid)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  {'model':<25}|{'FP16':>8}|{'uni-k0':>9}|{'uni-k1':>9}|{'uni-k4':>9}|{'wf-k0':>9}|{'wf-k4':>9}", flush=True)
    for sn, r in results.items():
        sw = r['sweeps']
        print(f"  {sn:<25}|{r['ppl_fp16']:>8.4f}|"
              f"{sw.get('uniform_sink0', 0):>9.4f}|"
              f"{sw.get('uniform_sink1', 0):>9.4f}|"
              f"{sw.get('uniform_sink4', 0):>9.4f}|"
              f"{sw.get('wf_sink0', 0):>9.4f}|"
              f"{sw.get('wf_sink4', 0):>9.4f}", flush=True)

    out = OUT_DIR / 'exp_v2h_sink_sweep.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Runtime: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f}m)")


if __name__ == '__main__':
    main()
