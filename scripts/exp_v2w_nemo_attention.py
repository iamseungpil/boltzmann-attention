#!/usr/bin/env python3
"""
V2w: Nemo Top-κ Head Attention Pattern Analysis
=================================================

v2u showed Nemo Lloyd+sink_k=1 barely helps at L=32768 (14.84 vs 17.18 no sink).
v2v and v2v2 didn't find a clean signature to explain why Nemo differs from
Mistral.

The missing measurement: DO Nemo's top-κ heads actually attend to pos 0?
If not → sink_k=1 obviously can't help (the protected token isn't attended).
If yes → something else (e.g., calibration mismatch) is responsible.

This is the Nemo counterpart of v2e Q2 + Q3 for Mistral.

Also: run the SAME analysis on Mistral-7B at long context (L=8192, not 1024
as in v2e) to see if Mistral's sink pattern holds at longer context.
"""
import json, os, time
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

DTYPE = torch.bfloat16
N_TOKENS = 2048  # moderate length (attention weights require T² memory)
OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')

MODELS = [
    ('mistralai/Mistral-7B-v0.3', 'mistral-7b'),
    ('mistralai/Mistral-Nemo-Base-2407', 'nemo-12b'),
]


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
    print(f"  loaded n_layers={n_layers}, n_kv={n_kv}, n_q={n_q}, head_dim={head_dim}, q/kv={q_per_kv}", flush=True)

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    text = '\n\n'.join([t for t in ds['text'] if len(t.strip()) > 100][:300])
    enc = tok(text, return_tensors='pt', truncation=True, max_length=N_TOKENS)
    ids = enc['input_ids'].to('cuda:0')
    T = ids.shape[1]
    token_strs = [tok.decode([i.item()]) for i in ids[0]]
    print(f"  Tokens: {T}, pos0 = {token_strs[0]!r}", flush=True)

    # Capture k_proj
    captured_k = {}
    def mk_k(li):
        def h(m, i, o): captured_k[li] = o.detach().cpu().float().numpy()
        return h
    handles = [model.model.layers[li].self_attn.k_proj.register_forward_hook(mk_k(li)) for li in range(n_layers)]
    with torch.no_grad():
        out = model(ids, use_cache=False, output_attentions=True)
    attn_weights = out.attentions
    for h in handles: h.remove()

    # Compute per-head κ
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
            head_stats.append({'layer': li, 'kv_head': hk, 'kappa': kappa, 'lam_top_med': lam_ratio})
    head_stats.sort(key=lambda x: -x['kappa'])
    top32 = head_stats[:32]

    # Compute attention mass on first-k positions for top-32
    for rec in top32:
        li = rec['layer']; hk = rec['kv_head']
        attn = attn_weights[li][0].float().cpu().numpy()  # (n_q, T, T)
        q_start = hk * q_per_kv
        q_end = q_start + q_per_kv
        attn_avg = attn[q_start:q_end].mean(axis=0).mean(axis=0)  # (T,)
        rec['attn_pos0'] = float(attn_avg[0])
        rec['attn_first4'] = float(attn_avg[:4].sum())
        rec['attn_first16'] = float(attn_avg[:16].sum())
        rec['attn_first64'] = float(attn_avg[:64].sum())
        # Top-k attended positions (NOT just early ones)
        top_pos = np.argsort(-attn_avg)[:10]
        rec['top10_attended_positions'] = [int(p) for p in top_pos]
        rec['top10_attended_mass'] = [float(attn_avg[p]) for p in top_pos]
        rec['frac_mass_top10'] = float(attn_avg[top_pos].sum())

    # Print top-15
    print(f"\n  Top-15 high-κ heads — attention pattern:", flush=True)
    print(f"  {'r':<3}|{'L':<4}|{'H':<3}|{'κ':>10}|{'pos0':>8}|{'first4':>8}|{'first16':>9}|{'first64':>9}|{'top10':>8}", flush=True)
    for i, rec in enumerate(top32[:15]):
        print(f"  {i+1:<3}|{rec['layer']:<4}|{rec['kv_head']:<3}|{rec['kappa']:>10.1e}|"
              f"{rec['attn_pos0']:>8.4f}|{rec['attn_first4']:>8.4f}|{rec['attn_first16']:>9.4f}|"
              f"{rec['attn_first64']:>9.4f}|{rec['frac_mass_top10']:>8.4f}", flush=True)

    # Aggregates
    mean_p0 = float(np.mean([r['attn_pos0'] for r in top32]))
    mean_f4 = float(np.mean([r['attn_first4'] for r in top32]))
    mean_f16 = float(np.mean([r['attn_first16'] for r in top32]))
    mean_f64 = float(np.mean([r['attn_first64'] for r in top32]))
    print(f"\n  Top-32 averages: pos0={mean_p0*100:.1f}% first4={mean_f4*100:.1f}% "
          f"first16={mean_f16*100:.1f}% first64={mean_f64*100:.1f}%", flush=True)

    # Where are the top-10 most-attended positions for each head?
    # Are they early (pos < 4) or distributed?
    all_top_pos = []
    for rec in top32:
        for p in rec['top10_attended_positions']:
            all_top_pos.append(p)
    early = sum(1 for p in all_top_pos if p < 4)
    mid = sum(1 for p in all_top_pos if 4 <= p < 128)
    far = sum(1 for p in all_top_pos if p >= 128)
    total = len(all_top_pos)
    print(f"\n  Top-10 attended positions across top-32 heads ({total} total):")
    print(f"    Early (pos 0-3):    {early}/{total} = {early/total*100:.1f}%")
    print(f"    Mid   (pos 4-127):  {mid}/{total} = {mid/total*100:.1f}%")
    print(f"    Far   (pos 128+):   {far}/{total} = {far/total*100:.1f}%", flush=True)

    del model, tok
    torch.cuda.empty_cache()

    return {
        'model': model_id, 'short_name': sn, 'n_tokens': T,
        'top32_heads': top32,
        'mean_pos0': mean_p0, 'mean_first4': mean_f4,
        'mean_first16': mean_f16, 'mean_first64': mean_f64,
        'top_pos_breakdown': {'early': early, 'mid': mid, 'far': far, 'total': total},
    }


def main():
    print("="*70)
    print("V2w: Nemo vs Mistral Top-κ Attention Patterns")
    print("="*70, flush=True)
    t_start = time.time()
    results = {}
    for mid, sn in MODELS:
        try:
            results[sn] = analyze_model(mid, sn)
        except Exception as e:
            print(f"ERROR on {sn}: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"  {'model':<14}|{'pos0':>8}|{'first4':>8}|{'first16':>9}|{'first64':>9}|{'early10%':>10}|{'mid10%':>10}|{'far10%':>10}")
    for sn, r in results.items():
        tp = r['top_pos_breakdown']; tot = tp['total']
        print(f"  {sn:<14}|{r['mean_pos0']*100:>7.1f}%|{r['mean_first4']*100:>7.1f}%|"
              f"{r['mean_first16']*100:>8.1f}%|{r['mean_first64']*100:>8.1f}%|"
              f"{tp['early']/tot*100:>9.1f}%|{tp['mid']/tot*100:>9.1f}%|{tp['far']/tot*100:>9.1f}%")

    out = OUT_DIR / 'exp_v2w_nemo_mistral_attention.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Runtime: {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
