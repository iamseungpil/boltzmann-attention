#!/usr/bin/env python3
"""
V2b: Cross-Model Massive Activation Signature
===============================================

Question: Does Llama-2-7B and Qwen2.5-7B exhibit the same massive-activation +
k_proj alignment pattern as Mistral-7B? If Mistral is more severe, that
mechanistically explains why it catastrophically fails at 2-bit while
Llama/Qwen tolerate it.

Runs Tests 1 + 2 from exp_v2 on Llama-2-7B and Qwen2.5-7B.
Mistral reference numbers are loaded from exp_v2 results.

GPU: 0 (one model at a time, reuse)
"""
import json, time, gc, os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

DTYPE = torch.bfloat16
N_CALIB_TOKENS = 2048
OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ('meta-llama/Llama-3.1-8B', 'llama-3.1-8b'),
]


def get_calib_text():
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        texts = [t for t in ds['text'] if len(t.strip()) > 100]
        return '\n\n'.join(texts[:300])
    except Exception:
        return " ".join(["Text."] * 5000)


def test_1_massive(model, input_ids, n_layers, threshold_ratio=100):
    captured = {}
    def mk(li):
        def h(m, i, o):
            a = o[0] if isinstance(o, tuple) else o
            captured[li] = a.detach().cpu().float().numpy()
        return h
    handles = [model.model.layers[li].register_forward_hook(mk(li)) for li in range(n_layers)]
    with torch.no_grad():
        _ = model(input_ids, use_cache=False)
    for h in handles:
        h.remove()

    per_layer = {}
    for li in range(n_layers):
        acts = captured[li][0]
        mpc = np.abs(acts).max(axis=0)
        med = float(np.median(mpc))
        mx = float(mpc.max())
        top10 = np.argsort(mpc)[::-1][:10]
        massive = np.where(mpc > threshold_ratio * med)[0]
        per_layer[li] = {
            'median_max': med,
            'max': mx,
            'ratio': mx / max(med, 1e-10),
            'n_massive': int(len(massive)),
            'massive_ids': [int(x) for x in massive.tolist()],
            'top10_ids': [int(x) for x in top10.tolist()],
        }
    return per_layer


def test_2_alignment(model, per_layer, n_layers, n_kv, head_dim):
    alignment = []
    for li in range(n_layers):
        massive = per_layer[li]['massive_ids']
        top3 = per_layer[li]['top10_ids'][:3]
        check = list(set(massive + top3))
        if not check:
            continue
        W = model.model.layers[li].self_attn.k_proj.weight.detach().cpu().float().numpy()
        for hk in range(n_kv):
            hw = W[hk*head_dim:(hk+1)*head_dim, :]
            me = float(np.sum(hw[:, check]**2))
            te = float(np.sum(hw**2))
            ar = me / max(te, 1e-12)
            rb = len(check) / W.shape[1]
            alignment.append({
                'layer': int(li),
                'kv_head': int(hk),
                'alignment_ratio': ar,
                'random_baseline': rb,
                'enrichment': ar / max(rb, 1e-12),
            })
    alignment.sort(key=lambda x: -x['enrichment'])
    return alignment


def analyze_model(model_id, short_name):
    print(f"\n{'='*70}\n  {short_name}: {model_id}\n{'='*70}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_q
    print(f"  n_layers={n_layers}, n_kv={n_kv}, n_q={n_q}, head_dim={head_dim}", flush=True)

    calib = get_calib_text()
    ids = tok(calib, return_tensors='pt', truncation=True, max_length=N_CALIB_TOKENS)['input_ids'].to('cuda:0')

    # Test 1
    per_layer = test_1_massive(model, ids, n_layers)
    max_ratio = max(d['ratio'] for d in per_layer.values())
    med_ratio = float(np.median([d['ratio'] for d in per_layer.values()]))
    n_with = sum(1 for d in per_layer.values() if d['n_massive'] > 0)
    print(f"\n  Test 1: max ratio {max_ratio:.1f}×, median layer ratio {med_ratio:.1f}×, "
          f"{n_with}/{n_layers} layers with massive channels", flush=True)

    # Print per-layer summary (top channel + ratio)
    print(f"  {'layer':<6}|{'ratio':>10}|top_ch", flush=True)
    for li in [0, 1, 2, 3, 4, 5, 6, 7, 8, n_layers//2, n_layers-3, n_layers-2, n_layers-1]:
        if li < n_layers:
            d = per_layer[li]
            print(f"  {li:<6}|{d['ratio']:>9.1f}×|ch{d['top10_ids'][0]}", flush=True)

    # Test 2
    alignment = test_2_alignment(model, per_layer, n_layers, n_kv, head_dim)
    max_enrich = max(h['enrichment'] for h in alignment) if alignment else 0
    n_strong = sum(1 for h in alignment if h['enrichment'] > 5)
    print(f"\n  Test 2: max enrichment {max_enrich:.2f}×, "
          f"{n_strong} heads with enrichment > 5×", flush=True)
    print(f"  Top-10 aligned heads:", flush=True)
    print(f"  {'rank':<5}|{'layer':<6}|{'kv_h':<5}|{'enrich':>10}", flush=True)
    for i, h in enumerate(alignment[:10]):
        print(f"  {i+1:<5}|{h['layer']:<6}|{h['kv_head']:<5}|{h['enrichment']:>9.2f}×", flush=True)

    # Free
    del model, tok, ids
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'model': model_id,
        'short_name': short_name,
        'n_layers': n_layers,
        'n_kv': n_kv,
        'head_dim': head_dim,
        'test1': {
            'max_ratio': max_ratio,
            'median_layer_ratio': med_ratio,
            'n_layers_with_massive': n_with,
            'per_layer': per_layer,
        },
        'test2': {
            'max_enrichment': max_enrich,
            'n_strong_alignment': n_strong,
            'top20': alignment[:20],
        },
    }


def main():
    print("="*70)
    print("V2b: Cross-Model Massive Activation Signature")
    print("="*70, flush=True)
    t_start = time.time()

    results = {}
    for mid, sn in MODELS:
        try:
            results[sn] = analyze_model(mid, sn)
        except Exception as e:
            print(f"  ERROR on {sn}: {e}", flush=True)
            import traceback; traceback.print_exc()
            results[sn] = {'error': str(e)}

    # Load Mistral reference from exp_v2 results
    try:
        with open(OUT_DIR / 'exp_v2_massive_activation_test.json') as f:
            v2 = json.load(f)
        mistral_t1 = v2['test1_massive_activations']
        mistral_max = max(float(d['ratio_max_to_median']) for d in mistral_t1.values())
        mistral_med = float(np.median([float(d['ratio_max_to_median']) for d in mistral_t1.values()]))
        mistral_t2 = v2['test2_kproj_alignment']
        mistral_max_enrich = max(h['enrichment'] for h in mistral_t2) if mistral_t2 else 0
        mistral_strong = sum(1 for h in mistral_t2 if h['enrichment'] > 5)
        print(f"\n{'='*70}\n  COMPARISON TABLE\n{'='*70}", flush=True)
        print(f"  {'model':<20}|{'max_ratio':>12}|{'med_ratio':>12}|{'max_enrich':>12}|{'n_enrich>5':>12}", flush=True)
        print(f"  {'-'*72}", flush=True)
        print(f"  {'mistral-7b-v0.3':<20}|{mistral_max:>11.1f}×|{mistral_med:>11.1f}×|"
              f"{mistral_max_enrich:>11.2f}×|{mistral_strong:>12}", flush=True)
        for sn, r in results.items():
            if 'error' in r:
                print(f"  {sn:<20}|   ERROR", flush=True)
                continue
            print(f"  {sn:<20}|{r['test1']['max_ratio']:>11.1f}×|"
                  f"{r['test1']['median_layer_ratio']:>11.1f}×|"
                  f"{r['test2']['max_enrichment']:>11.2f}×|"
                  f"{r['test2']['n_strong_alignment']:>12}", flush=True)
    except Exception as e:
        print(f"  Could not load Mistral reference: {e}", flush=True)

    out = OUT_DIR / 'exp_v2b_cross_model.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Total runtime: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f}m)")


if __name__ == '__main__':
    main()
