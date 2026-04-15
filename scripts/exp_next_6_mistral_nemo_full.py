#!/usr/bin/env python3
"""
Next-6: Mistral-Nemo-Base-2407 full suite (third model point)
==============================================================

Llama-3.1-8B is gated and not locally available. We use Mistral-Nemo-Base-2407
(12B, 40 layers, different architecture scale) as the third model point.

Two-phase experiment:
  Phase 1: Per-layer Lloyd failure breakdown (like Exp4/Next-3)
  Phase 2: Sensitivity-based bit allocation (like Next-5)

Goal: Test if the "per-layer localization + sensitivity-based allocation"
      finding generalizes beyond Mistral-7B and Qwen.
"""
import json
import time
import gc
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

MODEL_NAME = 'mistralai/Mistral-Nemo-Base-2407'
SHORT = 'mistral-nemo-12b'
DEVICE = 'cuda:0'
DTYPE = torch.bfloat16
N_CALIB_TOKENS = 1024
N_EVAL_TOKENS = 2048
BITS = 2

OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def lloyd_max_1d_fit(col, bits, n_iter=30):
    n_levels = 2 ** bits
    pcts = np.linspace(0, 100, n_levels + 2)[1:-1]
    centroids = np.percentile(col, pcts)
    centroids = np.sort(centroids)
    for _ in range(n_iter):
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        idx = np.searchsorted(boundaries, col)
        new_c = centroids.copy()
        for k in range(n_levels):
            m = idx == k
            if m.sum() > 0:
                new_c[k] = col[m].mean()
        if np.max(np.abs(new_c - centroids)) < 1e-6:
            break
        centroids = new_c
    return centroids


class MixedBitLloydHook:
    def __init__(self, centroids_per_head, n_kv, head_dim):
        self.centroids = centroids_per_head
        self.n_kv = n_kv
        self.head_dim = head_dim

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x = output.view(B, T, self.n_kv, self.head_dim)
        x_np = x.float().cpu().numpy()
        x_q = np.zeros_like(x_np)
        for hk in range(self.n_kv):
            data = x_np[:, :, hk, :]
            shape = data.shape
            data_flat = data.reshape(-1, self.head_dim)
            c = self.centroids[hk]
            for j in range(self.head_dim):
                boundaries = (c[j, :-1] + c[j, 1:]) / 2
                idx = np.searchsorted(boundaries, data_flat[:, j])
                data_flat[:, j] = c[j, idx]
            x_q[:, :, hk, :] = data_flat.reshape(shape)
        return torch.from_numpy(x_q).view(B, T, self.n_kv * self.head_dim).to(output.device).to(output.dtype)


def fit_lloyd_for_layer_bits(model, tok, calib_text, layer_idx, bits):
    enc = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB_TOKENS)
    input_ids = enc['input_ids'].to(DEVICE)
    captured = {}
    def hook(m, i, o):
        captured['k'] = o.detach().cpu().float().numpy()
    attn = model.model.layers[layer_idx].self_attn
    h = attn.k_proj.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(input_ids, use_cache=False)
    h.remove()

    K = captured['k']
    n_kv = model.config.num_key_value_heads
    head_dim = model.config.head_dim if hasattr(model.config, 'head_dim') and model.config.head_dim else model.config.hidden_size // model.config.num_attention_heads
    K = K.reshape(1, -1, n_kv, head_dim)[0]
    centroids_per_head = []
    for hk in range(n_kv):
        K_h = K[:, hk, :].astype(np.float32)
        centroids = np.zeros((head_dim, 2 ** bits))
        for j in range(head_dim):
            centroids[j] = lloyd_max_1d_fit(K_h[:, j], bits, n_iter=20)
        centroids_per_head.append(centroids)
    return centroids_per_head


def compute_ppl(model, input_ids):
    with torch.no_grad():
        out = model(input_ids, use_cache=False)
        logits = out.logits[:, :-1, :].contiguous()
        targets = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            reduction='mean'
        )
        return float(torch.exp(loss).item()), float(loss.item())


def get_texts(tok):
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        texts = [t for t in ds['text'] if len(t.strip()) > 100]
        return '\n\n'.join(texts[:300]), '\n\n'.join(texts[300:600])
    except Exception:
        return " ".join(["Calib."] * 5000), " ".join(["Eval."] * 5000)


def main():
    print("=" * 60)
    print(f"Next-6: {MODEL_NAME} full suite")
    print("=" * 60, flush=True)
    t_start = time.time()

    print("Loading model (12B, may take time)...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device_map=DEVICE,
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t_start:.1f}s", flush=True)

    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    head_dim = getattr(model.config, 'head_dim', None) or (model.config.hidden_size // n_q)
    print(f"  n_layers={n_layers}, n_kv={n_kv}, n_q={n_q}, head_dim={head_dim}", flush=True)

    calib_text, eval_text = get_texts(tok)
    eval_enc = tok(eval_text, return_tensors='pt', truncation=True, max_length=N_EVAL_TOKENS)
    eval_ids = eval_enc['input_ids'].to(DEVICE)
    T_eval = eval_ids.shape[1]
    print(f"  Eval tokens: {T_eval}", flush=True)

    # FP16 baseline
    print("\nFP16 baseline...", flush=True)
    ppl_fp16, _ = compute_ppl(model, eval_ids)
    print(f"  FP16 PPL: {ppl_fp16:.4f}", flush=True)

    # ======================================
    # PHASE 1: Per-layer Lloyd breakdown
    # ======================================
    print(f"\n{'='*40}")
    print("PHASE 1: Per-layer Lloyd breakdown")
    print('='*40, flush=True)

    per_layer = []
    for li in range(n_layers):
        t_li = time.time()
        centroids = fit_lloyd_for_layer_bits(model, tok, calib_text, li, BITS)
        hook = MixedBitLloydHook(centroids, n_kv, head_dim)
        handle = model.model.layers[li].self_attn.k_proj.register_forward_hook(hook)
        ppl_li, loss_li = compute_ppl(model, eval_ids)
        handle.remove()

        delta = ppl_li - ppl_fp16
        ratio = ppl_li / ppl_fp16
        marker = '!!!' if ratio > 1.1 else '   '
        print(f"  {marker} L{li:3d}: PPL={ppl_li:8.3f}  Δ={delta:+7.3f}  r={ratio:.3f} ({time.time()-t_li:.1f}s)", flush=True)
        per_layer.append({
            'layer': li, 'ppl': ppl_li, 'delta_ppl': delta, 'ratio': ratio,
        })

    sorted_delta = sorted(per_layer, key=lambda x: -x['delta_ppl'])
    print(f"\n  Top-10 catastrophic layers:")
    for r in sorted_delta[:10]:
        print(f"    L{r['layer']:3d}: Δ={r['delta_ppl']:+.3f}, r={r['ratio']:.3f}")

    # ======================================
    # PHASE 2: Sensitivity-based allocation
    # ======================================
    print(f"\n{'='*40}")
    print("PHASE 2: Sensitivity-based allocation")
    print('='*40, flush=True)

    sorted_layers_by_sens = [r['layer'] for r in sorted_delta]

    def make_config(top_k, extra_bits):
        bits = [2] * n_layers
        for li in sorted_layers_by_sens[:top_k]:
            bits[li] = 2 + extra_bits
        return bits

    def make_mixed(top_a, top_b, extra_a, extra_b):
        bits = [2] * n_layers
        for li in sorted_layers_by_sens[:top_a]:
            bits[li] = 2 + extra_a
        for li in sorted_layers_by_sens[top_a:top_b]:
            bits[li] = 2 + extra_b
        return bits

    configs = {
        'baseline_all_2b': [2] * n_layers,
        'top1_4b': make_config(1, 2),
        'top3_3b': make_config(3, 1),
        'top5_3b': make_config(5, 1),
        'top10_3b': make_config(10, 1),
        'top15_3b': make_config(15, 1),
        'top20_3b': make_config(20, 1),
        'top10_4b': make_config(10, 2),
        'mixed_top5_4b_top15_3b': make_mixed(5, 15, 2, 1),
        'all_4b': [4] * n_layers,
    }

    # Fit unique pairs
    unique_pairs = set()
    for bl in configs.values():
        for li, b in enumerate(bl):
            unique_pairs.add((li, b))
    print(f"\n  Fitting {len(unique_pairs)} unique (layer, bits) pairs...", flush=True)

    centroid_cache = {}
    t_fit = time.time()
    # Cache the phase-1 layer centroids
    # (phase-1 only had 2-bit, so we need the others)
    for idx, (li, b) in enumerate(sorted(unique_pairs)):
        if (li, b) in centroid_cache:
            continue
        centroid_cache[(li, b)] = fit_lloyd_for_layer_bits(model, tok, calib_text, li, b)
        if (idx + 1) % 20 == 0:
            print(f"  Fit progress: {idx+1}/{len(unique_pairs)}", flush=True)
    print(f"  Fit done in {time.time()-t_fit:.1f}s", flush=True)

    results_config = {}
    for name, bits_list in configs.items():
        avg_b = sum(bits_list) / len(bits_list)
        per_layer_cent = [centroid_cache[(li, bits_list[li])] for li in range(n_layers)]
        handles = []
        for li, cent in enumerate(per_layer_cent):
            hook = MixedBitLloydHook(cent, n_kv, head_dim)
            h = model.model.layers[li].self_attn.k_proj.register_forward_hook(hook)
            handles.append(h)
        t0 = time.time()
        ppl, loss = compute_ppl(model, eval_ids)
        elapsed = time.time() - t0
        for h in handles:
            h.remove()

        results_config[name] = {
            'avg_bits': avg_b,
            'ppl': ppl,
            'loss': loss,
            'delta_vs_fp16_pct': (ppl - ppl_fp16) / ppl_fp16 * 100,
            'bits_per_layer': bits_list,
        }
        print(f"  {name:<28} avg={avg_b:.3f} PPL={ppl:.4f} Δ={((ppl-ppl_fp16)/ppl_fp16*100):+.2f}% ({elapsed:.1f}s)", flush=True)

    # Pareto frontier
    base_ppl = results_config['baseline_all_2b']['ppl']
    by_budget = sorted(results_config.items(), key=lambda x: (x[1]['avg_bits'], x[1]['ppl']))
    pareto = []
    best_so_far = float('inf')
    for name, r in by_budget:
        if r['ppl'] < best_so_far:
            pareto.append((name, r['avg_bits'], r['ppl']))
            best_so_far = r['ppl']

    print(f"\n  Pareto frontier:")
    for name, ab, ppl in pareto:
        print(f"    {name:<28} avg={ab:.3f} PPL={ppl:.4f}")

    # Save
    result = {
        'model': MODEL_NAME,
        'short_name': SHORT,
        'n_layers': n_layers,
        'n_kv': n_kv,
        'n_q': n_q,
        'head_dim': head_dim,
        'bits': BITS,
        'ppl_fp16': ppl_fp16,
        'all_2bit_baseline_ppl': base_ppl,
        'baseline_failure_ratio': base_ppl / ppl_fp16,
        'per_layer_breakdown': per_layer,
        'sorted_by_delta': sorted_delta,
        'sensitivity_ranking': [(r['layer'], r['delta_ppl']) for r in sorted_delta],
        'allocation_configs': results_config,
        'pareto_frontier': pareto,
        'runtime_sec': time.time() - t_start,
    }
    out_file = OUT_DIR / 'exp_next6_mistral_nemo_full.json'
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_file}")
    print(f"Total runtime: {result['runtime_sec']:.1f}s ({result['runtime_sec']/60:.1f}m)")


if __name__ == '__main__':
    main()
