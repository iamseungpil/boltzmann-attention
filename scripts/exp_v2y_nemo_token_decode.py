#!/usr/bin/env python3
"""
V2y: Nemo Top-κ Head Attended Token Identification
=====================================================

v2w showed Nemo's high-κ heads are NOT attention sinks (15.3% pos 0 vs
Mistral 56.3%). They attend to mid/late positions. This script answers:
WHICH tokens are they attending to?

If the attended tokens are recognizable (e.g., delimiters, numbers, rare
unicode, specific punctuation), that tells us:
  (a) Whether the problem is a specific token class that could be protected
  (b) Whether Nemo has "delayed sinks" (tokens at fixed positions but not 0)
  (c) Whether the attended tokens repeat or are unique per sequence

Protocol: for Nemo-12B top-32 high-κ heads, dump the top-10 attended
positions and decode each to its token string.
"""
import json, os, time
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from collections import Counter

MODEL = 'mistralai/Mistral-Nemo-Base-2407'
DTYPE = torch.bfloat16
N_TOKENS = 2048
OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')


def main():
    print("="*70)
    print("V2y: Nemo Top-κ Attended Token Decoding")
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
    print(f"  n_layers={n_layers}, n_kv={n_kv}, n_q={n_q}, head_dim={head_dim}, loaded in {time.time()-t0:.1f}s")

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    text = '\n\n'.join([t for t in ds['text'] if len(t.strip()) > 100][:300])
    enc = tok(text, return_tensors='pt', truncation=True, max_length=N_TOKENS)
    ids = enc['input_ids'].to('cuda:0')
    T = ids.shape[1]
    token_strs = [tok.decode([i.item()]) for i in ids[0]]
    print(f"  Tokens: {T}", flush=True)

    captured_k = {}
    def mk_k(li):
        def h(m, i, o): captured_k[li] = o.detach().cpu().float().numpy()
        return h
    handles = [model.model.layers[li].self_attn.k_proj.register_forward_hook(mk_k(li)) for li in range(n_layers)]
    with torch.no_grad():
        out = model(ids, use_cache=False, output_attentions=True)
    attn_weights = out.attentions
    for h in handles: h.remove()

    # Compute κ per head
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
            head_stats.append({'layer': li, 'kv_head': hk, 'kappa': kappa})
    head_stats.sort(key=lambda x: -x['kappa'])
    top32 = head_stats[:32]

    # For each top head, get top-10 attended positions and their tokens
    print(f"\n{'='*70}")
    print(f"Top-15 High-κ Heads — Decoded Top Attended Tokens")
    print(f"{'='*70}\n")
    all_attended_tokens = []
    per_head_decoded = []

    for rank, rec in enumerate(top32):
        li = rec['layer']; hk = rec['kv_head']
        attn = attn_weights[li][0].float().cpu().numpy()
        q_start = hk * q_per_kv
        q_end = q_start + q_per_kv
        attn_avg = attn[q_start:q_end].mean(axis=0).mean(axis=0)  # (T,)
        top_pos = np.argsort(-attn_avg)[:10]
        top_mass = attn_avg[top_pos]
        tokens = [token_strs[p] for p in top_pos]
        per_head_decoded.append({
            'rank': rank + 1,
            'layer': li,
            'kv_head': hk,
            'kappa': rec['kappa'],
            'top10_positions': [int(p) for p in top_pos],
            'top10_masses': [float(m) for m in top_mass],
            'top10_tokens': tokens,
            'total_top10_mass': float(top_mass.sum()),
        })
        all_attended_tokens.extend(tokens)

        if rank < 15:
            print(f"Rank {rank+1:2d}: L{li:<2} H{hk} κ={rec['kappa']:.1e} (top10 mass={top_mass.sum():.3f})")
            for i, (p, m, t) in enumerate(zip(top_pos, top_mass, tokens)):
                ts = t.replace('\n', '\\n').replace('\t', '\\t')[:15]
                print(f"    #{i+1}: pos={p:>4}, mass={m:.4f}, tok={ts!r}")
            print()

    # Token frequency across all top-32 heads × top-10 attended positions
    print(f"\n{'='*70}")
    print(f"Aggregated Token Frequency (top-10 attended across 32 high-κ heads = 320 slots)")
    print(f"{'='*70}")
    token_counts = Counter(all_attended_tokens)
    print(f"\n  Top-20 most frequent attended tokens:")
    for tok_str, count in token_counts.most_common(20):
        ts = tok_str.replace('\n', '\\n').replace('\t', '\\t')[:20]
        print(f"    {count:>4} × {ts!r}")

    # Category analysis
    def categorize(t):
        ts = t.strip()
        if not ts:
            return 'whitespace'
        if ts == '<s>' or ts == '</s>' or ts == '<unk>':
            return 'special'
        if all(c in '.,;:!?-()[]{}"\'"' for c in ts):
            return 'punct'
        if ts.isdigit():
            return 'digit'
        if any(not c.isascii() for c in ts):
            return 'unicode'
        if '\n' in t:
            return 'newline'
        return 'word'

    categories = Counter(categorize(t) for t in all_attended_tokens)
    total = sum(categories.values())
    print(f"\n  Category breakdown:")
    for cat, n in categories.most_common():
        print(f"    {cat:<12}: {n:>4} ({n/total*100:.1f}%)")

    unique = len(set(all_attended_tokens))
    print(f"\n  Unique token strings: {unique}/{total}")
    print(f"  Top-token dominance: {token_counts.most_common(1)[0][1]}/{total}")

    # Save
    results = {
        'model': MODEL,
        'n_tokens': T,
        'per_head_top10': per_head_decoded,
        'aggregate_token_freq': dict(token_counts.most_common(50)),
        'category_breakdown': dict(categories),
        'unique_token_count': unique,
    }
    out = OUT_DIR / 'exp_v2y_nemo_token_decode.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Runtime: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
