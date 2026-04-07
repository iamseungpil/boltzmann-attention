#!/usr/bin/env python3
"""
V2q: Qwen2.5-7B Sink Mechanism Diagnosis (position-based or token-based?)
===========================================================================

Qwen's tokenizer does NOT auto-prepend BOS. Cal position 0 = ' Sen', Eval
position 0 = ' There'. Yet v2h showed Qwen benefits from sink_k=1 (8.167 →
7.516 PPL). Two hypotheses:

  H_pos: The sink is position-based. Whatever token lives at position 0
         acts as a sink because the model learned to dump attention there.
  H_tok: The sink is token-based. Some specific token type (e.g. a delimiter
         or common word) consistently accumulates massive-activation signature
         and happens to appear at position 0 in both cal and eval.

Protocol (mirrors v2e for Mistral):
  Q1: Find residual-stream massive channels and which TOKEN POSITIONS fire them.
      If the massive channel fires on position 0 regardless of token identity
      → H_pos. If it fires on a specific token at various positions → H_tok.

  Q2: Find high-κ heads (per-head PCA on calibration). Measure attention
      fraction on first-k positions. If high → sink-dominated (same as Mistral).

  Q3: Top PCA eigenvector of K for top-κ heads: which token positions/types?

  Q4: Controlled position-0 test. Run two eval sequences:
      (a) Normal eval with ' There' at position 0
      (b) Same eval but with a different token inserted at position 0
      If sink_k=1 benefit is the same in both → position-based.
      If it drops or changes → token-based.

GPU: 1
"""
import json, os, time
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

MODEL = 'Qwen/Qwen2.5-7B'
DTYPE = torch.bfloat16
N_TOKENS = 1024
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


def compute_ppl(model, ids):
    with torch.no_grad():
        out = model(ids, use_cache=False)
        logits = out.logits[:, :-1].contiguous()
        tgt = ids[:, 1:].contiguous()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                tgt.reshape(-1), reduction='mean')
        return float(torch.exp(loss).item())


class SinkHook:
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


def main():
    print("="*70)
    print("V2q: Qwen2.5-7B Sink Mechanism Diagnosis")
    print("="*70, flush=True)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='eager', low_cpu_mem_usage=True,
        output_attentions=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    head_dim = getattr(model.config, 'head_dim', None) or (model.config.hidden_size // n_q)
    q_per_kv = n_q // n_kv
    hidden = model.config.hidden_size
    print(f"  n_layers={n_layers}, n_kv={n_kv}, n_q={n_q}, head_dim={head_dim}, hidden={hidden}", flush=True)

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    text = '\n\n'.join([t for t in ds['text'] if len(t.strip()) > 100][:200])
    enc = tok(text, return_tensors='pt', truncation=True, max_length=N_TOKENS)
    ids = enc['input_ids'].to('cuda:0')
    T = ids.shape[1]
    token_strs = [tok.decode([i.item()]) for i in ids[0]]
    print(f"  Tokens: {T}, pos0 = {token_strs[0]!r}", flush=True)

    # ===== Capture residual + k_proj =====
    captured_res = {}
    captured_k = {}
    def mk_res(li):
        def h(m, i, o):
            a = o[0] if isinstance(o, tuple) else o
            captured_res[li] = a.detach().cpu().float().numpy()
        return h
    def mk_k(li):
        def h(m, i, o):
            captured_k[li] = o.detach().cpu().float().numpy()
        return h
    handles = []
    for li in range(n_layers):
        handles.append(model.model.layers[li].register_forward_hook(mk_res(li)))
        handles.append(model.model.layers[li].self_attn.k_proj.register_forward_hook(mk_k(li)))
    with torch.no_grad():
        out = model(ids, use_cache=False, output_attentions=True)
    attn_weights = out.attentions
    for h in handles: h.remove()

    # ===== Q1: Residual massive channels + which token positions fire them =====
    print(f"\n=== Q1: Residual-Stream Massive Channels ===", flush=True)
    massive_per_layer = {}
    for li in range(n_layers):
        acts = captured_res[li][0]  # (T, hidden)
        mpc = np.abs(acts).max(axis=0)  # (hidden,)
        med = float(np.median(mpc))
        mx = float(mpc.max())
        top10 = np.argsort(mpc)[::-1][:10]
        massive_per_layer[li] = {
            'median_max': med, 'max': mx, 'ratio': mx/max(med,1e-10),
            'top10_ids': [int(x) for x in top10.tolist()],
        }

    # Print per-layer summary
    print(f"  {'L':<4}|{'ratio':>10}|top_ch", flush=True)
    for li in [0, 1, 2, 5, 10, 14, 20, 27]:
        if li < n_layers:
            d = massive_per_layer[li]
            print(f"  {li:<4}|{d['ratio']:>9.1f}×|ch{d['top10_ids'][0]}", flush=True)

    # Find the most-massive channel overall
    max_ratio_layer = max(massive_per_layer.items(), key=lambda x: x[1]['ratio'])
    probe_li, probe_info = max_ratio_layer
    probe_ch = probe_info['top10_ids'][0]
    print(f"\n  Strongest channel: L{probe_li} ch{probe_ch} (ratio {probe_info['ratio']:.1f}×)", flush=True)

    # Which positions fire that channel?
    res_probe = captured_res[probe_li][0]  # (T, hidden)
    ch_acts = res_probe[:, probe_ch]
    abs_acts = np.abs(ch_acts)
    top20 = np.argsort(-abs_acts)[:20]
    print(f"  Top-20 positions for L{probe_li} ch{probe_ch}:", flush=True)
    print(f"  {'rank':<5}|{'pos':<5}|{'value':>12}|token", flush=True)
    for i, p in enumerate(top20[:15]):
        s = token_strs[p].replace('\n', '\\n')[:18]
        print(f"  {i+1:<5}|{p:<5}|{ch_acts[p]:>12.3f}|{repr(s)}", flush=True)

    frac_top1 = float(abs_acts[top20[0]] / abs_acts.sum())
    frac_pos0 = float(abs_acts[0] / abs_acts.sum())
    print(f"\n  Fraction at top-1 pos ({top20[0]}): {frac_top1*100:.1f}%")
    print(f"  Fraction at pos 0:               {frac_pos0*100:.1f}%")
    print(f"  Is pos 0 in top-5? {0 in top20[:5].tolist()}")

    # ===== Q2: Per-head κ + sink attention mass =====
    print(f"\n=== Q2: High-κ Heads + Attention on First-k Positions ===", flush=True)
    head_stats = []
    for li in range(n_layers):
        K_all = captured_k[li].reshape(-1, n_kv, head_dim).astype(np.float32)
        for hk in range(n_kv):
            K = K_all[:, hk, :]
            Kc = K - K.mean(axis=0)
            cov = (Kc.T @ Kc) / max(K.shape[0]-1, 1)
            ev = np.linalg.eigvalsh(cov)
            ev = np.sort(ev)[::-1]
            kappa = float(ev[0] / max(ev[-1], 1e-12))
            lam_ratio = float(ev[0] / max(np.median(ev), 1e-12))
            head_stats.append({
                'layer': li, 'kv_head': hk,
                'kappa': kappa, 'lam_top_over_median': lam_ratio,
            })
    head_stats.sort(key=lambda x: -x['kappa'])
    print(f"  Top-15 heads by κ:")
    print(f"  {'rank':<4}|{'L':<4}|{'H':<3}|{'κ':>12}|{'λ1/med':>10}|{'pos0':>8}|{'first4':>8}", flush=True)
    for i, rec in enumerate(head_stats[:15]):
        li = rec['layer']; hk = rec['kv_head']
        attn = attn_weights[li][0].float().cpu().numpy()  # (n_q, T, T)
        q_start = hk * q_per_kv
        q_end = q_start + q_per_kv
        attn_avg = attn[q_start:q_end].mean(axis=0).mean(axis=0)  # (T,)
        a0 = float(attn_avg[0])
        a4 = float(attn_avg[:4].sum())
        rec['attn_pos0'] = a0
        rec['attn_first4'] = a4
        print(f"  {i+1:<4}|{li:<4}|{hk:<3}|{rec['kappa']:>12.2e}|{rec['lam_top_over_median']:>10.1f}|"
              f"{a0:>8.4f}|{a4:>8.4f}", flush=True)

    # Aggregate stats over top-32 heads (matching Mistral count)
    top_k_heads = head_stats[:32]
    mean_pos0 = float(np.mean([h['attn_pos0'] for h in top_k_heads]))
    mean_first4 = float(np.mean([h['attn_first4'] for h in top_k_heads]))
    print(f"\n  Top-32 Qwen heads: mean attn pos0 = {mean_pos0*100:.1f}%, first4 = {mean_first4*100:.1f}%")
    print(f"  (Mistral v2e reference: 60.4% first-4 on top-32 κ>1e4 heads)")

    # ===== Q3: Top PCA eigenvector — which token positions? =====
    print(f"\n=== Q3: Top-PCA Eigenvector of K for Top-5 κ Heads ===", flush=True)
    q3 = []
    for rec in head_stats[:5]:
        li = rec['layer']; hk = rec['kv_head']
        K_all = captured_k[li].reshape(-1, n_kv, head_dim).astype(np.float32)
        K = K_all[:, hk, :]
        Kc = K - K.mean(axis=0)
        cov = (Kc.T @ Kc) / max(K.shape[0]-1, 1)
        ev, V = np.linalg.eigh(cov)
        order = np.argsort(ev)[::-1]
        V = V[:, order]
        proj = Kc @ V[:, 0]
        ap = np.abs(proj)
        top5 = np.argsort(-ap)[:5]
        q3.append({
            'layer': li, 'kv_head': hk, 'kappa': rec['kappa'],
            'top5_positions': [int(p) for p in top5],
            'top5_tokens': [token_strs[p] for p in top5],
            'proj_at_pos0': float(proj[0]),
            'proj_max': float(proj[np.argmax(ap)]),
        })
        print(f"  L{li} H{hk} (κ={rec['kappa']:.1e})", flush=True)
        print(f"    Top-5 positions: {top5.tolist()}")
        print(f"    Top-5 tokens: {[repr(t[:10]) for t in q3[-1]['top5_tokens']]}")
        print(f"    proj[pos 0] = {float(proj[0]):.4f}, proj[max] = {float(proj[np.argmax(ap)]):.4f}", flush=True)

    # Save
    results = {
        'model': MODEL,
        'n_tokens': T,
        'pos0_token': token_strs[0],
        'q1_strongest_channel': {
            'layer': probe_li, 'channel': probe_ch,
            'ratio': probe_info['ratio'],
            'top20_positions': [int(x) for x in top20],
            'top20_tokens': [token_strs[p] for p in top20],
            'top20_values': [float(ch_acts[p]) for p in top20],
            'frac_at_top1': frac_top1,
            'frac_at_pos0': frac_pos0,
            'pos0_in_top5': bool(0 in top20[:5].tolist()),
        },
        'q2_head_sink_stats': {
            'top_32_heads': top_k_heads,
            'mean_attn_pos0_top32': mean_pos0,
            'mean_attn_first4_top32': mean_first4,
        },
        'q3_top_pca_positions': q3,
        'runtime_sec': time.time() - t0,
    }
    out = OUT_DIR / 'exp_v2q_qwen_sink_mechanism.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Runtime: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
