#!/usr/bin/env python3
"""
exp_next11: Theorem-faithful mode classification
==================================================

Apply Corollaries 6.4 / 6.5 / 6.6 of PART1_PAPER_DRAFT_v3.md directly,
without the §5.1 "mean over top-32 by κ" proxy that fails on full-MHA
models (Llama-2-7B, Phi-3-mini).

Corollary 6.4 (Mode A): ∃ (layer, head) with κ ≥ κ_high AND pos0_attn ≥ 1−ε.
Corollary 6.5 (Mode B): κ_max ≥ κ_high but no Cor 6.4 witness exists;
  attention concentrates on m∼5–15 positions instead.
Corollary 6.6 (Mode C): κ_max < κ_low; attention diffuse.

Thresholds (matching v3 §5.1 numerical conventions):
  κ_high = 1e6
  κ_low  = 1e5
  ε      = 0.5  (so pos0_attn ≥ 0.5 ⇒ Cor 6.4 single-position dominance)

For each tested model we report:
  - max pos0_attn over all heads
  - max pos0_attn over high-κ heads (κ ≥ 1e6)
  - number of (l,h) satisfying Cor 6.4 simultaneously
  - assigned mode by the theorem-faithful rule
  - mean pos0_attn over top-32 by κ (for reference vs §5.1)
"""
import json, os, sys, time
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

DTYPE = torch.bfloat16
N_CALIB = 2048
KAPPA_HIGH = 1e6
KAPPA_LOW  = 1e5
EPS = 0.5  # Cor 6.4 single-position dominance threshold
OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')

# Models: counterexamples + controls
MODELS = [
    'NousResearch/Llama-2-7b-hf',
    'microsoft/Phi-3-mini-4k-instruct',
    'NousResearch/Meta-Llama-3.1-8B',
    'mistralai/Mistral-7B-v0.3',
    'Qwen/Qwen2.5-7B',
    'Qwen/Qwen2.5-14B',
    'Qwen/Qwen2-7B',
    'Qwen/Qwen2.5-0.5B-Instruct',
]


def get_k_module(layer):
    sa = layer.self_attn
    if hasattr(sa, 'k_proj'):
        return sa.k_proj, (lambda out: out)
    if hasattr(sa, 'qkv_proj'):
        n_q = sa.config.num_attention_heads
        n_kv = sa.config.num_key_value_heads
        hd = getattr(sa.config, 'head_dim', None) or (sa.config.hidden_size // n_q)
        q_sz = n_q * hd; k_sz = n_kv * hd
        return sa.qkv_proj, (lambda out: out[..., q_sz:q_sz + k_sz])
    raise RuntimeError("no k_proj / qkv_proj")


def measure(model_id):
    short = model_id.split('/')[-1].lower()
    print(f"\n{'=' * 70}\n  {short}\n{'=' * 70}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_q = cfg.num_attention_heads
    n_kv = getattr(cfg, 'num_key_value_heads', n_q)
    head_dim = getattr(cfg, 'head_dim', None) or (cfg.hidden_size // n_q)
    q_per_kv = n_q // n_kv
    print(f"  n_layers={n_layers} n_q={n_q} n_kv={n_kv} head_dim={head_dim}  load {time.time()-t0:.1f}s", flush=True)

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    text = '\n\n'.join([t for t in ds['text'] if len(t.strip()) > 100][:300])
    ids = tok(text, return_tensors='pt', truncation=True, max_length=N_CALIB)['input_ids'].to('cuda:0')

    captured_k = {}
    def mk(li, slc):
        def h(m, i, o): captured_k[li] = slc(o).detach().cpu().float().numpy()
        return h
    handles = []
    for li in range(n_layers):
        kmod, slc = get_k_module(model.model.layers[li])
        handles.append(kmod.register_forward_hook(mk(li, slc)))
    with torch.no_grad():
        out = model(ids, use_cache=False, output_attentions=True)
    attn_all = out.attentions
    for h in handles: h.remove()

    # per-(layer, kv_head): κ, pos0_attn
    rows = []
    for li in range(n_layers):
        K_all = captured_k[li].reshape(-1, n_kv, head_dim).astype(np.float32)
        attn_l = attn_all[li][0].float().cpu().numpy()  # (n_q, T, T)
        for hk in range(n_kv):
            K = K_all[:, hk, :]; Kc = K - K.mean(axis=0)
            cov = (Kc.T @ Kc) / max(K.shape[0]-1, 1)
            ev = np.linalg.eigvalsh(cov); ev = np.sort(ev)[::-1]
            kappa = float(ev[0] / max(ev[-1], 1e-12))
            q_start = hk * q_per_kv; q_end = q_start + q_per_kv
            attn_avg = attn_l[q_start:q_end].mean(axis=0).mean(axis=0)
            p0 = float(attn_avg[0])
            rows.append({'layer': li, 'kv_head': hk, 'kappa': kappa, 'pos0': p0})

    n_total = len(rows)
    kappa_max = max(r['kappa'] for r in rows)

    # Cor 6.4 witnesses: heads with κ ≥ κ_high AND pos0 ≥ 1-ε = 0.5
    cor64_witnesses = [r for r in rows
                        if r['kappa'] >= KAPPA_HIGH and r['pos0'] >= (1 - EPS)]
    high_kappa_heads = [r for r in rows if r['kappa'] >= KAPPA_HIGH]

    max_p0_all = max(r['pos0'] for r in rows)
    max_p0_high_k = max((r['pos0'] for r in high_kappa_heads), default=0.0)
    n_high_k = len(high_kappa_heads)
    n_witness = len(cor64_witnesses)

    # §5.1 reference: mean pos0 over top-32 by κ
    by_k = sorted(rows, key=lambda r: -r['kappa'])
    top32 = by_k[:32]
    mean_p0_top32k = float(np.mean([r['pos0'] for r in top32]))

    # Theorem-faithful classification
    if kappa_max < KAPPA_LOW:
        mode = 'C (Cor 6.6: low anisotropy, bulk-tail)'
    elif n_witness >= 1:
        mode = 'A (Cor 6.4: localized positional sink)'
    else:
        # high κ_max but no single-position dominator → Cor 6.5 distributed
        mode = 'B (Cor 6.5: distributed structural tail)'

    # §5.1 reference classification (the broken one)
    if mean_p0_top32k > 0.40 and kappa_max > 1e6:
        mode_v51 = 'A'
    elif mean_p0_top32k < 0.20 and kappa_max > 1e6:
        mode_v51 = 'B'
    elif kappa_max < 1e5:
        mode_v51 = 'C'
    else:
        mode_v51 = 'BOUNDARY'

    print(f"  κ_max               = {kappa_max:.2e}")
    print(f"  max pos0 (all heads)   = {max_p0_all*100:.2f}%")
    print(f"  max pos0 (κ≥{KAPPA_HIGH:.0e})  = {max_p0_high_k*100:.2f}%")
    print(f"  #heads κ ≥ {KAPPA_HIGH:.0e}     = {n_high_k}/{n_total}")
    print(f"  #Cor 6.4 witnesses     = {n_witness}  (κ≥{KAPPA_HIGH:.0e} AND pos0≥{1-EPS:.2f})")
    print(f"  mean pos0 (top-32 κ)   = {mean_p0_top32k*100:.2f}%   [§5.1 reference]")
    print(f"  → §5.1 (broken)        : {mode_v51}")
    print(f"  → THEOREM-FAITHFUL     : {mode}")

    # show witness heads (or strongest candidates)
    if cor64_witnesses:
        print(f"  Cor 6.4 witness heads (top 5 by pos0):")
        for r in sorted(cor64_witnesses, key=lambda x: -x['pos0'])[:5]:
            print(f"    L{r['layer']:>2} H{r['kv_head']:>2}  κ={r['kappa']:.2e}  pos0={r['pos0']*100:.2f}%")
    else:
        print(f"  No Cor 6.4 witness; top-5 high-κ heads by pos0:")
        for r in sorted(high_kappa_heads, key=lambda x: -x['pos0'])[:5]:
            print(f"    L{r['layer']:>2} H{r['kv_head']:>2}  κ={r['kappa']:.2e}  pos0={r['pos0']*100:.2f}%")

    del model, tok
    torch.cuda.empty_cache()

    return {
        'model': model_id, 'short_name': short,
        'n_layers': n_layers, 'n_q': n_q, 'n_kv': n_kv, 'head_dim': head_dim,
        'n_total_heads': n_total,
        'kappa_max': kappa_max,
        'max_pos0_all': max_p0_all,
        'max_pos0_high_kappa': max_p0_high_k,
        'n_high_kappa_heads': n_high_k,
        'n_cor64_witnesses': n_witness,
        'mean_pos0_top32_kappa': mean_p0_top32k,
        'mode_theorem': mode,
        'mode_v51': mode_v51,
        'cor64_witness_heads': sorted(cor64_witnesses, key=lambda x: -x['pos0'])[:10],
        'all_high_kappa_heads': sorted(high_kappa_heads, key=lambda x: -x['pos0'])[:20],
    }


def main():
    print("=" * 70)
    print("exp_next11: Theorem-faithful mode classification (v3 Cor 6.4/5/6)")
    print("=" * 70, flush=True)

    results = {}
    for mid in MODELS:
        try:
            results[mid.split('/')[-1].lower()] = measure(mid)
        except Exception as e:
            print(f"ERROR {mid}: {e}")
            import traceback; traceback.print_exc()

    out = OUT_DIR / 'exp_next11_theorem_classify.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'model':<24}|{'n_h':>5}|{'κ_max':>10}|{'maxP0':>8}|{'maxP0(hi-κ)':>13}|{'#wit':>6}|{'§5.1':>10}|{'theorem':>14}")
    for sn, r in results.items():
        print(f"  {sn:<24}|{int(r['n_total_heads']):>5}|{r['kappa_max']:>10.2e}|"
              f"{r['max_pos0_all']*100:>7.2f}%|{r['max_pos0_high_kappa']*100:>12.2f}%|"
              f"{int(r['n_cor64_witnesses']):>6}|{r['mode_v51']:>10}|{r['mode_theorem'][:14]:>14}")


if __name__ == '__main__':
    main()
