#!/usr/bin/env python3
"""
V2f: FP16 Ceiling & Selective Protection — Upper Bound for Per-Head WF
========================================================================

v2c showed: FP16=5.388, Uniform-2bit=9.953, WF-2bit=7.084 (+1.696 residual gap)

This experiment bounds what any further per-dim allocation scheme can achieve
by selectively upgrading specific dims to FP16:

  [A] FP16 dim-0 only on 32 high-κ heads (κ>1e4) + rest 2-bit uniform
  [B] FP16 dim-0 only on 32 high-κ heads (κ>1e4) + rest 2-bit WF
  [C] FP16 top-3 dims on 32 high-κ heads + 2-bit WF on rest
  [D] FP16 dim-0 on ALL 256 heads + 2-bit WF on rest (full top-dim protection)
  [E] FP16 first-4 TOKEN positions everywhere (sink protection) + 2-bit WF

If [A] or [B] get PPL near FP16 (5.4), then dim-0 of high-κ heads is the
whole story. If they still show substantial gap, the residual lives elsewhere.

[E] tests the sink hypothesis directly: if protecting the first 4 token
positions closes most of the gap, then the problem was attention sinks all
along and per-dim WF was fixing a symptom.

GPU: 0 (exp_v2e uses GPU 1)
"""
import json, os, time, gc
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

MODEL = 'mistralai/Mistral-7B-v0.3'
DTYPE = torch.bfloat16
N_CALIB = 2048
N_EVAL = 2048
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


def wf_alloc(sigma2, total_budget, b_floor=1, b_max=8):
    n = len(sigma2); s = np.maximum(sigma2, 1e-12)
    bits = np.zeros(n, dtype=int); spent = 0
    while spent < total_budget:
        bg = -np.inf; best = None
        for j in range(n):
            if bits[j] == 0:
                if spent + b_floor > total_budget:
                    continue
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


class SelectiveHook:
    """
    Per-head PCA quantization with selective FP16 protection.
    - fp16_dims[hk]: set of PCA dim indices to keep in FP16 (not quantized)
    - fp16_positions: set of TOKEN positions to keep in FP16 (sink protection)
    """
    def __init__(self, n_kv, head_dim, V_list, mean_list, centroids_list,
                 fp16_dims=None, fp16_positions=None):
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.V_list = V_list
        self.mean_list = mean_list
        self.centroids_list = centroids_list
        self.fp16_dims = fp16_dims or {hk: set() for hk in range(n_kv)}
        self.fp16_positions = fp16_positions  # list/set of token idxs or None

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x_orig = output.view(B, T, self.n_kv, self.head_dim).float().cpu().numpy()
        x = x_orig.copy()
        for hk in range(self.n_kv):
            V = self.V_list[hk]
            m = self.mean_list[hk]
            cents = self.centroids_list[hk]
            data = x[:, :, hk, :].reshape(-1, self.head_dim)
            K_c = data - m
            K_pca = K_c @ V
            K_q = K_pca.copy()
            fp_dims = self.fp16_dims.get(hk, set())
            for j in range(self.head_dim):
                if j in fp_dims:
                    continue  # keep FP16
                cj = cents[j]
                if cj is None or len(cj) == 1:
                    K_q[:, j] = cj[0] if cj is not None else 0.0
                else:
                    bnd = (cj[:-1] + cj[1:]) / 2
                    idx = np.searchsorted(bnd, K_pca[:, j])
                    K_q[:, j] = cj[idx]
            K_rec = K_q @ V.T + m
            x[:, :, hk, :] = K_rec.reshape(B, T, self.head_dim)
        # Sink protection: restore original K for first-k token positions
        if self.fp16_positions is not None:
            for p in self.fp16_positions:
                if p < T:
                    x[:, p, :, :] = x_orig[:, p, :, :]
        return torch.from_numpy(x).to(output.device).to(output.dtype).view(B, T, self.n_kv * self.head_dim)


def calibrate(model, ids, n_layers, n_kv, head_dim):
    pl = {}
    def mk(li):
        def h(m, i, o):
            pl[li] = o.detach().cpu().float().numpy()
        return h
    handles = [model.model.layers[li].self_attn.k_proj.register_forward_hook(mk(li)) for li in range(n_layers)]
    with torch.no_grad():
        _ = model(ids, use_cache=False)
    for h in handles: h.remove()

    basis = {}
    for li in range(n_layers):
        K_all = pl[li].reshape(-1, n_kv, head_dim).astype(np.float32)
        per_head = []
        for hk in range(n_kv):
            K = K_all[:, hk, :]
            mean = K.mean(axis=0)
            Kc = K - mean
            cov = (Kc.T @ Kc) / max(K.shape[0]-1, 1)
            ev, vv = np.linalg.eigh(cov)
            order = np.argsort(ev)[::-1]
            V = vv[:, order].astype(np.float32)
            per_head.append({'V': V, 'mean': mean.astype(np.float32),
                             'eigvals': ev[order], 'K_pca': Kc @ V})
        basis[li] = per_head
    return basis


def fit_cents(basis, scheme, n_layers, n_kv, head_dim, target_avg_bits=2.0,
              fp16_dims_per_head=None):
    """
    scheme: 'uniform2' or 'wf2'
    fp16_dims_per_head: dict (li, hk) -> set of PCA dims that are FP16 (exclude from bit budget)
    Returns: centroids_by_layer dict, total_bits_used
    """
    out = {}
    total_bits = 0
    for li in range(n_layers):
        per = []
        for hk in range(n_kv):
            bh = basis[li][hk]
            ev = bh['eigvals']; K_pca = bh['K_pca']
            fp_dims = set()
            if fp16_dims_per_head is not None:
                fp_dims = fp16_dims_per_head.get((li, hk), set())
            # Budget ONLY over non-FP16 dims
            free_dims = [j for j in range(head_dim) if j not in fp_dims]
            n_free = len(free_dims)
            budget = int(target_avg_bits * head_dim)  # keep same total bits on non-FP16
            if scheme == 'uniform2':
                bits_sub = np.full(n_free, 2, dtype=int)
            elif scheme == 'wf2':
                ev_sub = np.array([ev[j] for j in free_dims])
                bits_sub = wf_alloc(ev_sub, total_budget=budget, b_floor=1, b_max=8)
            else:
                raise ValueError(scheme)
            bits = np.zeros(head_dim, dtype=int)
            for idx, j in enumerate(free_dims):
                bits[j] = bits_sub[idx]
            total_bits += int(bits.sum()) + 16 * len(fp_dims)
            cents = []
            for j in range(head_dim):
                if j in fp_dims:
                    cents.append(None)  # FP16, skip
                    continue
                b = int(bits[j])
                if b == 0:
                    cents.append(np.array([0.0], dtype=np.float32))
                else:
                    cents.append(lloyd_1d(K_pca[:, j], b, n_iter=15))
            per.append(cents)
        out[li] = per
    return out, total_bits


def install_hook(model, basis, cents, fp16_dims_per_layer_head, fp16_positions, n_layers, n_kv, head_dim):
    handles = []
    for li in range(n_layers):
        V_list = [basis[li][hk]['V'] for hk in range(n_kv)]
        mean_list = [basis[li][hk]['mean'] for hk in range(n_kv)]
        per_hk_fp = {hk: fp16_dims_per_layer_head.get((li, hk), set()) for hk in range(n_kv)}
        hook = SelectiveHook(n_kv, head_dim, V_list, mean_list, cents[li],
                             fp16_dims=per_hk_fp, fp16_positions=fp16_positions)
        handles.append(model.model.layers[li].self_attn.k_proj.register_forward_hook(hook))
    return handles


def main():
    print("="*70)
    print("V2f: FP16 Ceiling & Selective Protection")
    print("="*70, flush=True)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in ds['text'] if len(t.strip()) > 100]
    calib_text = '\n\n'.join(texts[:300]); eval_text = '\n\n'.join(texts[300:600])
    calib_ids = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB)['input_ids'].to('cuda:0')
    eval_ids = tok(eval_text, return_tensors='pt', truncation=True, max_length=N_EVAL)['input_ids'].to('cuda:0')

    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    # === FP16 baseline ===
    ppl_fp16 = compute_ppl(model, eval_ids)
    print(f"\n[baseline] FP16: {ppl_fp16:.4f}", flush=True)

    # === Calibrate ===
    t1 = time.time()
    basis = calibrate(model, calib_ids, n_layers, n_kv, head_dim)
    print(f"  Calibrated in {time.time()-t1:.1f}s", flush=True)

    # Identify high-κ heads from basis
    high_kappa_heads = []
    for li in range(n_layers):
        for hk in range(n_kv):
            ev = basis[li][hk]['eigvals']
            k = float(ev[0] / max(ev[-1], 1e-12))
            if k > 1e4:
                high_kappa_heads.append((li, hk))
    print(f"  {len(high_kappa_heads)} heads with κ>1e4", flush=True)

    results = {'ppl_fp16': ppl_fp16}

    # === [A] FP16 dim-0 high-κ + uniform-2 elsewhere ===
    fp_A = {(li, hk): {0} for (li, hk) in high_kappa_heads}
    print("\n[A] FP16 dim-0 high-κ + Uniform-2bit elsewhere", flush=True)
    cA, _ = fit_cents(basis, 'uniform2', n_layers, n_kv, head_dim, fp16_dims_per_head=fp_A)
    handles = install_hook(model, basis, cA, fp_A, None, n_layers, n_kv, head_dim)
    pA = compute_ppl(model, eval_ids)
    for h in handles: h.remove()
    print(f"  [A] PPL = {pA:.4f}  (Δ = {pA-ppl_fp16:+.4f})", flush=True)
    results['A_fp16_d0_highk_uniform'] = pA
    del cA; gc.collect()

    # === [B] FP16 dim-0 high-κ + WF-2 elsewhere ===
    print("\n[B] FP16 dim-0 high-κ + WF-2bit elsewhere", flush=True)
    cB, _ = fit_cents(basis, 'wf2', n_layers, n_kv, head_dim, fp16_dims_per_head=fp_A)
    handles = install_hook(model, basis, cB, fp_A, None, n_layers, n_kv, head_dim)
    pB = compute_ppl(model, eval_ids)
    for h in handles: h.remove()
    print(f"  [B] PPL = {pB:.4f}  (Δ = {pB-ppl_fp16:+.4f})", flush=True)
    results['B_fp16_d0_highk_wf'] = pB
    del cB; gc.collect()

    # === [C] FP16 top-3 dims high-κ + WF-2 elsewhere ===
    fp_C = {(li, hk): {0, 1, 2} for (li, hk) in high_kappa_heads}
    print("\n[C] FP16 top-3 dims high-κ + WF-2bit elsewhere", flush=True)
    cC, _ = fit_cents(basis, 'wf2', n_layers, n_kv, head_dim, fp16_dims_per_head=fp_C)
    handles = install_hook(model, basis, cC, fp_C, None, n_layers, n_kv, head_dim)
    pC = compute_ppl(model, eval_ids)
    for h in handles: h.remove()
    print(f"  [C] PPL = {pC:.4f}  (Δ = {pC-ppl_fp16:+.4f})", flush=True)
    results['C_fp16_top3_highk_wf'] = pC
    del cC; gc.collect()

    # === [D] FP16 dim-0 ALL heads + WF-2 elsewhere ===
    fp_D = {(li, hk): {0} for li in range(n_layers) for hk in range(n_kv)}
    print("\n[D] FP16 dim-0 ALL heads + WF-2bit elsewhere", flush=True)
    cD, _ = fit_cents(basis, 'wf2', n_layers, n_kv, head_dim, fp16_dims_per_head=fp_D)
    handles = install_hook(model, basis, cD, fp_D, None, n_layers, n_kv, head_dim)
    pD = compute_ppl(model, eval_ids)
    for h in handles: h.remove()
    print(f"  [D] PPL = {pD:.4f}  (Δ = {pD-ppl_fp16:+.4f})", flush=True)
    results['D_fp16_d0_all_wf'] = pD
    del cD; gc.collect()

    # === [E] Sink protection: FP16 first-4 tokens + WF-2 on rest ===
    fp_E = {}  # no FP16 dims
    print("\n[E] FP16 first-4 token positions (sinks) + WF-2bit rest", flush=True)
    cE, _ = fit_cents(basis, 'wf2', n_layers, n_kv, head_dim, fp16_dims_per_head=fp_E)
    handles = install_hook(model, basis, cE, fp_E, [0, 1, 2, 3], n_layers, n_kv, head_dim)
    pE = compute_ppl(model, eval_ids)
    for h in handles: h.remove()
    print(f"  [E] PPL = {pE:.4f}  (Δ = {pE-ppl_fp16:+.4f})", flush=True)
    results['E_sink_fp16_wf'] = pE

    # === Summary ===
    print("\n" + "="*70)
    print("SUMMARY — Bounding the residual +1.696 PPL gap of v2c WF")
    print("="*70)
    print(f"  v2c reference:")
    print(f"    FP16             : {ppl_fp16:.4f}")
    print(f"    Uniform 2-bit    : 9.9529 (+4.565)")
    print(f"    WF 2-bit (v2c)   : 7.0840 (+1.696)")
    print(f"\n  This experiment:")
    print(f"    [A] FP16 d0 high-κ + Uniform : {pA:.4f}  (Δ = {pA-ppl_fp16:+.4f})")
    print(f"    [B] FP16 d0 high-κ + WF      : {pB:.4f}  (Δ = {pB-ppl_fp16:+.4f})")
    print(f"    [C] FP16 top3 high-κ + WF    : {pC:.4f}  (Δ = {pC-ppl_fp16:+.4f})")
    print(f"    [D] FP16 d0 ALL + WF         : {pD:.4f}  (Δ = {pD-ppl_fp16:+.4f})")
    print(f"    [E] FP16 first-4 tokens + WF : {pE:.4f}  (Δ = {pE-ppl_fp16:+.4f})")

    out = OUT_DIR / 'exp_v2f_fp16_ceiling.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Runtime: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
