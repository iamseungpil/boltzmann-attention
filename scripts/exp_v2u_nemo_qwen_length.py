#!/usr/bin/env python3
"""
V2u: Length Sweep on Mistral-Nemo and Qwen2.5 — Universal Lloyd Catastrophe?
==============================================================================

v2p showed Mistral Lloyd catastrophe scales with eval length:
  L=2048: 9.96, L=8192: 14.53, L=32768: 20.12
and sink_k=1 fully resolves it at all lengths.

This experiment checks whether the same pattern holds on the other two
architectures in our cross-model set, and whether the two types of Qwen
high-κ heads (sink-type and delimiter-type from v2q) show different behavior.

Per model: length ∈ {2048, 8192, 32768} × quantizer ∈ {Lloyd, Grid} × sink_k ∈ {0, 1}

Expected outcomes:
  - Nemo: similar to Mistral (same family; expect Lloyd catastrophe scaling)
  - Qwen: milder Lloyd catastrophe (fewer true sink heads) but possibly a
          length-dependent component from delimiter-driven Type A heads that
          sink_k=1 doesn't fix
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
EVAL_LENGTHS = [2048, 8192, 32768]
OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')

MODELS = [
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


def uniform_grid_1d(col, bits):
    n_levels = 2 ** bits
    if n_levels <= 1:
        return np.array([float(col.mean())], dtype=np.float32)
    r = float(np.max(np.abs(col)))
    if r < 1e-12:
        return np.array([0.0] * n_levels, dtype=np.float32)
    return np.linspace(-r, r, n_levels).astype(np.float32)


def compute_ppl(model, ids):
    # Chunked cross-entropy to avoid OOM at long eval (avoids materializing
    # full logits.float() tensor which is T x vocab x 4 bytes)
    with torch.no_grad():
        out = model(ids, use_cache=False)
        logits = out.logits[:, :-1].contiguous()  # (B, T-1, V) in bf16
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
        mean_loss = total_loss / total_n
        return float(np.exp(mean_loss))


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
            ph.append({'V': V, 'mean': mean.astype(np.float32),
                       'eigvals': ev[order], 'K_pca': Kc @ V})
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
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='sdpa', low_cpu_mem_usage=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, 'head_dim', None) or (model.config.hidden_size // model.config.num_attention_heads)
    print(f"  n_layers={n_layers}, n_kv={n_kv}, head_dim={head_dim}, loaded in {time.time()-t0:.1f}s", flush=True)

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in ds['text'] if len(t.strip()) > 100]
    calib = '\n\n'.join(texts[:300])
    eval_t = '\n\n'.join(texts[300:3000])
    calib_ids = tok(calib, return_tensors='pt', truncation=True, max_length=N_CALIB)['input_ids'].to('cuda:0')

    fp16 = {}
    for L in EVAL_LENGTHS:
        eval_ids = tok(eval_t, return_tensors='pt', truncation=True, max_length=L)['input_ids'].to('cuda:0')
        fp16[L] = compute_ppl(model, eval_ids)
        print(f"  FP16 @ L={L}: {fp16[L]:.4f}", flush=True)

    basis = calibrate(model, calib_ids, n_layers, n_kv, head_dim)

    cents_lloyd = fit_cents(basis, 'lloyd', n_layers, n_kv, head_dim)
    cents_grid = fit_cents(basis, 'grid', n_layers, n_kv, head_dim)

    result = {'model': model_id, 'fp16': fp16, 'configs': {}}
    for L in EVAL_LENGTHS:
        eval_ids = tok(eval_t, return_tensors='pt', truncation=True, max_length=L)['input_ids'].to('cuda:0')
        for qtype, cents in [('lloyd', cents_lloyd), ('grid', cents_grid)]:
            for sink_k in [0, 1, 4]:
                handles = install(model, basis, cents, sink_k, n_layers, n_kv, head_dim)
                ppl = compute_ppl(model, eval_ids)
                for h in handles: h.remove()
                key = f'L{L}_{qtype}_sink{sink_k}'
                result['configs'][key] = ppl
                print(f"  [L={L:>5} {qtype:<5} sink={sink_k}] PPL = {ppl:.4f}  (Δ = {ppl-fp16[L]:+.4f})", flush=True)

    del model, tok, basis, cents_lloyd, cents_grid
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    print("="*70)
    print("V2u: Nemo + Qwen length × quantizer × sink")
    print("="*70, flush=True)
    t_start = time.time()

    results = {}
    for mid in MODELS:
        try:
            sn = mid.split('/')[-1].lower()
            results[sn] = run_model(mid)
        except Exception as e:
            print(f"ERROR on {mid}: {e}", flush=True)
            import traceback; traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for sn, r in results.items():
        print(f"\n  {sn}:")
        for L in EVAL_LENGTHS:
            fp = r['fp16'][L]
            row = f"  L={L:<5} FP16={fp:.3f} |"
            for qt in ['lloyd', 'grid']:
                for s in [0, 1, 4]:
                    p = r['configs'][f'L{L}_{qt}_sink{s}']
                    row += f" {qt[0]}{s}={p:.3f}"
            print(row)

    out = OUT_DIR / 'exp_v2u_nemo_qwen_length.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Runtime: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f}m)")


if __name__ == '__main__':
    main()
