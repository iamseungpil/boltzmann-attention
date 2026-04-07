#!/usr/bin/env python3
"""
V2e: Attention Sinks & Token-Level Massive Activation Diagnosis
================================================================

Questions:
  Q1: Which TOKEN POSITIONS fire channel 2070 (the Mistral massive channel)?
      → If they match BOS/delimiters, this is an attention sink phenomenon.
  Q2: For the 32 high-κ heads (v2d), what fraction of attention mass goes to
      the first 4 tokens vs the rest?
  Q3: For the top-1 PCA projection of each high-κ head's K, which token
      positions are the extreme values?

If Q1 says "only BOS" and Q2 says "first 4 tokens get >80% of attention on
high-κ heads", then the entire 2-bit catastrophe can be fixed by keeping
the first 4 tokens of KV cache in FP16 (negligible cost).

GPU: 1 (exp_v2f uses GPU 0)
"""
import json, os, time
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

MODEL = 'mistralai/Mistral-7B-v0.3'
DTYPE = torch.bfloat16
N_TOKENS = 1024  # shorter for attention-weight capture (T² memory)
OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')

MASSIVE_CHANNEL = 2070  # from exp_v2 Mistral Test 1


def main():
    print("="*70)
    print("V2e: Attention Sinks & Token-Level Diagnosis")
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
    head_dim = model.config.hidden_size // n_q
    q_per_kv = n_q // n_kv
    print(f"  n_layers={n_layers}, n_kv={n_kv}, n_q={n_q}, head_dim={head_dim}, q_per_kv={q_per_kv}", flush=True)

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    text = '\n\n'.join([t for t in ds['text'] if len(t.strip()) > 100][:200])
    enc = tok(text, return_tensors='pt', truncation=True, max_length=N_TOKENS)
    ids = enc['input_ids'].to('cuda:0')
    token_strs = [tok.decode([i.item()]) for i in ids[0]]
    T = ids.shape[1]
    print(f"  Tokens: {T}", flush=True)

    # ---- Capture per-layer residual AND k_proj outputs ----
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
    attn_weights = out.attentions  # tuple of (B, n_q, T, T) per layer

    for h in handles:
        h.remove()

    # ============================================================
    # Q1: Which tokens fire channel 2070?
    # ============================================================
    print(f"\n=== Q1: Token positions firing channel {MASSIVE_CHANNEL} ===", flush=True)
    # Use L5 residual (strong massive signal, past initial ramp)
    L_PROBE = 5
    res = captured_res[L_PROBE][0]  # (T, hidden)
    ch_acts = res[:, MASSIVE_CHANNEL]  # (T,)
    abs_acts = np.abs(ch_acts)
    top_tokens = np.argsort(-abs_acts)[:20]
    print(f"  Layer {L_PROBE}, channel {MASSIVE_CHANNEL}:", flush=True)
    print(f"  Top-20 token positions by |activation|:", flush=True)
    print(f"  {'rank':<5}|{'pos':<5}|{'value':>12}|{'token':<20}", flush=True)
    for i, p in enumerate(top_tokens):
        s = token_strs[p].replace('\n', '\\n')[:18]
        print(f"  {i+1:<5}|{p:<5}|{ch_acts[p]:>12.3f}|{repr(s):<20}", flush=True)

    # Stats: what fraction of total |activation| comes from top-1, top-5, top-20 positions?
    total = float(abs_acts.sum())
    frac_top1 = float(abs_acts[top_tokens[0]] / total)
    frac_top5 = float(abs_acts[top_tokens[:5]].sum() / total)
    frac_top20 = float(abs_acts[top_tokens[:20]].sum() / total)
    pos_top1 = int(top_tokens[0])
    print(f"\n  Concentration (fraction of total |activation|):", flush=True)
    print(f"    Top-1 position ({pos_top1}): {frac_top1*100:.1f}%")
    print(f"    Top-5 positions:            {frac_top5*100:.1f}%")
    print(f"    Top-20 positions:           {frac_top20*100:.1f}%")

    q1 = {
        'probe_layer': L_PROBE,
        'channel': MASSIVE_CHANNEL,
        'top20_positions': [int(x) for x in top_tokens],
        'top20_values': [float(ch_acts[p]) for p in top_tokens],
        'top20_token_strs': [token_strs[p] for p in top_tokens],
        'frac_total_top1': frac_top1,
        'frac_total_top5': frac_top5,
        'frac_total_top20': frac_top20,
    }

    # ============================================================
    # Q2: Attention mass on first 4 tokens for high-κ heads
    # ============================================================
    print(f"\n=== Q2: Attention Sink — first-4-tokens mass on high-κ heads ===", flush=True)

    # Load high-κ heads from v2d
    with open(OUT_DIR / 'exp_v2d_head_bit_analysis.json') as f:
        v2d = json.load(f)
    high_kappa_heads = [(h['layer'], h['kv_head']) for h in v2d['head_stats'] if h['kappa'] > 1e4]
    print(f"  {len(high_kappa_heads)} heads with κ > 1e4", flush=True)

    sink_stats = []
    for li, hk in high_kappa_heads:
        attn = attn_weights[li][0].float().cpu().numpy()  # (n_q, T, T)
        # Q heads associated with this KV head
        q_start = hk * q_per_kv
        q_end = q_start + q_per_kv
        attn_kv = attn[q_start:q_end]  # (q_per_kv, T, T)
        # For each query row, attention distribution over keys
        # Averaged over queries (excluding early positions which are forced-sink)
        avg_attn = attn_kv.mean(axis=0).mean(axis=0)  # (T,) avg over (heads, queries)
        sink_mass_1 = float(avg_attn[0])
        sink_mass_4 = float(avg_attn[:4].sum())
        sink_mass_16 = float(avg_attn[:16].sum())
        sink_stats.append({
            'layer': li, 'kv_head': hk,
            'avg_attn_pos0': sink_mass_1,
            'avg_attn_first4': sink_mass_4,
            'avg_attn_first16': sink_mass_16,
        })

    # Sort by first-4 mass
    sink_stats.sort(key=lambda x: -x['avg_attn_first4'])
    print(f"\n  Top-15 high-κ heads by attention-on-first-4-tokens:")
    print(f"  {'L':<3}|{'H':<3}|{'pos0':>10}|{'first4':>10}|{'first16':>10}", flush=True)
    for s in sink_stats[:15]:
        print(f"  {s['layer']:<3}|{s['kv_head']:<3}|{s['avg_attn_pos0']:>10.4f}|"
              f"{s['avg_attn_first4']:>10.4f}|{s['avg_attn_first16']:>10.4f}", flush=True)

    mean_first4 = float(np.mean([s['avg_attn_first4'] for s in sink_stats]))
    median_first4 = float(np.median([s['avg_attn_first4'] for s in sink_stats]))
    n_sink_dominated = sum(1 for s in sink_stats if s['avg_attn_first4'] > 0.5)
    print(f"\n  Mean first-4 attention on high-κ heads:   {mean_first4*100:.1f}%")
    print(f"  Median first-4 attention on high-κ heads: {median_first4*100:.1f}%")
    print(f"  High-κ heads with >50% attention on first-4: {n_sink_dominated}/{len(sink_stats)}")

    # ============================================================
    # Q3: Top-1 PCA projection of K — which token positions?
    # ============================================================
    print(f"\n=== Q3: Top-PCA projection — token positions ===", flush=True)
    # For top-5 highest-κ heads
    top5 = v2d['head_stats'][:5]  # already sorted by κ
    q3_stats = []
    for rec in top5:
        li = rec['layer']; hk = rec['kv_head']
        K_all = captured_k[li].reshape(-1, n_kv, head_dim).astype(np.float32)
        K = K_all[:, hk, :]
        K_c = K - K.mean(axis=0)
        cov = (K_c.T @ K_c) / max(K.shape[0] - 1, 1)
        ev, V = np.linalg.eigh(cov)
        order = np.argsort(ev)[::-1]
        V = V[:, order]
        top_proj = K_c @ V[:, 0]  # (T,)
        abs_proj = np.abs(top_proj)
        top_pos = np.argsort(-abs_proj)[:5]
        top_toks = [token_strs[p] for p in top_pos]
        frac_at_top1 = float(abs_proj[top_pos[0]] / abs_proj.sum())
        frac_at_top5 = float(abs_proj[top_pos[:5]].sum() / abs_proj.sum())
        q3_stats.append({
            'layer': li, 'kv_head': hk,
            'kappa': rec['kappa'],
            'top_positions': [int(p) for p in top_pos],
            'top_tokens': top_toks,
            'frac_top1': frac_at_top1,
            'frac_top5': frac_at_top5,
        })
        print(f"  L{li} H{hk} (κ={rec['kappa']:.1e}):")
        print(f"    Top-5 positions: {top_pos.tolist()}")
        print(f"    Top-5 tokens:    {[repr(t[:10]) for t in top_toks]}")
        print(f"    frac at top-1: {frac_at_top1*100:.1f}%, top-5: {frac_at_top5*100:.1f}%", flush=True)

    # ============================================================
    # Save
    # ============================================================
    results = {
        'model': MODEL,
        'n_tokens': T,
        'q1_channel_activation': q1,
        'q2_sink_attention': {
            'n_high_kappa_heads': len(high_kappa_heads),
            'mean_first4_attn': mean_first4,
            'median_first4_attn': median_first4,
            'n_sink_dominated_gt_50pct': n_sink_dominated,
            'per_head': sink_stats,
        },
        'q3_top_pca_positions': q3_stats,
        'runtime_sec': time.time() - t0,
    }
    out = OUT_DIR / 'exp_v2e_attention_sinks.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Runtime: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
