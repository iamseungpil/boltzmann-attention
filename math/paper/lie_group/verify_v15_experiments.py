#!/usr/bin/env python3
"""
V15 Experiments: Theory-Practice Gap Resolution
================================================

Exp V15-1: KVTC comparison (shared vs per-head PCA)
Exp V15-2: Calibration size effect on Mistral
Exp V15-3: Adaptive Lloyd-Max (data-driven codebook)
Exp V15-4: Water-filling with minimum bit floor

Usage:
  python verify_v15_experiments.py --exp 1 --model meta-llama/Llama-3.1-8B --device cuda:0
  python verify_v15_experiments.py --exp 2 --model mistralai/Mistral-7B-v0.3 --device cuda:0
  python verify_v15_experiments.py --exp 3 --model Qwen/Qwen2.5-7B --device cuda:0
  python verify_v15_experiments.py --exp 4 --model Qwen/Qwen2.5-7B --device cuda:0
  python verify_v15_experiments.py --exp all --model Qwen/Qwen2.5-7B --device cuda:0
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

OUTDIR = Path(__file__).parent / "verification_results"

# ============================================================================
# Quantization functions
# ============================================================================
LLOYD_CB = {
    2: (np.array([-0.9816, 0.0, 0.9816]),
        np.array([-1.5104, -0.4528, 0.4528, 1.5104])),
    3: (np.array([-1.7480, -1.0500, -0.5006, 0.0, 0.5006, 1.0500, 1.7480]),
        np.array([-2.1519, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1519])),
    4: (np.array([-2.4008, -1.8435, -1.4370, -1.0993, -0.7996, -0.5224, -0.2582, 0.0,
                   0.2582, 0.5224, 0.7996, 1.0993, 1.4370, 1.8435, 2.4008]),
        np.array([-2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3881, -0.1284,
                   0.1284, 0.3881, 0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326])),
}


def uniform_quant_col(col, bits):
    nl = 2 ** bits
    vmin, vmax = col.min(), col.max()
    if vmax - vmin < 1e-10:
        return col.copy()
    s = (vmax - vmin) / (nl - 1)
    q = np.clip(np.round((col - vmin) / s).astype(int), 0, nl - 1)
    return q * s + vmin


def gaussian_lloyd_col(col, bits):
    """Gaussian Lloyd-Max with centering."""
    th, ct = LLOYD_CB[bits]
    mean, std = col.mean(), max(col.std(), 1e-10)
    idx = np.searchsorted(th, (col - mean) / std)
    return ct[idx] * std + mean


def adaptive_lloyd_col(col, bits, n_iter=10):
    """Data-adaptive Lloyd-Max: initialize centroids from data, then iterate.

    Unlike Gaussian Lloyd-Max which uses fixed N(0,1) codebook,
    this initializes from the actual data percentiles and runs Lloyd iteration.
    """
    nl = 2 ** bits
    vmin, vmax = col.min(), col.max()
    if vmax - vmin < 1e-10:
        return col.copy()

    # Initialize centroids from data percentiles
    percentiles = np.linspace(0, 100, nl + 2)[1:-1]  # exclude 0th and 100th
    centroids = np.percentile(col, percentiles)

    for _ in range(n_iter):
        # Assignment step: each value → nearest centroid
        dists = np.abs(col[:, None] - centroids[None, :])  # (n, nl)
        assignments = np.argmin(dists, axis=1)

        # Update step: centroid = mean of assigned values
        new_centroids = np.zeros(nl)
        for k in range(nl):
            mask = assignments == k
            if mask.sum() > 0:
                new_centroids[k] = col[mask].mean()
            else:
                new_centroids[k] = centroids[k]
        centroids = new_centroids

    # Final quantization
    dists = np.abs(col[:, None] - centroids[None, :])
    assignments = np.argmin(dists, axis=1)
    return centroids[assignments]


def compute_wf_alloc(variances, avg_bits, min_bits=1):
    """Water-filling bit allocation with minimum bit floor.

    Fixed: iterative correction to maintain total budget after floor application.
    Upper clip at 4 bits (matching Lloyd-Max codebook and hook clip).
    """
    d = len(variances)
    max_bits = 4  # match LLOYD_CB and hook clip
    log_var = np.log2(np.maximum(variances, 1e-10))
    mean_log_var = np.mean(log_var)
    b_alloc = avg_bits + 0.5 * (log_var - mean_log_var)
    b_alloc = np.clip(b_alloc, min_bits, max_bits)
    total_target = d * avg_bits

    # Iterative correction: rescale, round, fix floor, redistribute deficit
    b_alloc = b_alloc * (total_target / max(b_alloc.sum(), 1e-10))
    b_alloc = np.maximum(np.round(b_alloc), min_bits)
    b_alloc = np.minimum(b_alloc, max_bits)

    # Redistribute any budget surplus/deficit
    deficit = int(total_target - b_alloc.sum())
    if deficit > 0:
        # Add bits to channels closest to their ceiling
        headroom = max_bits - b_alloc
        idx = np.argsort(-headroom)
        for i in idx[:deficit]:
            if b_alloc[i] < max_bits:
                b_alloc[i] += 1
    elif deficit < 0:
        # Remove bits from channels closest to their floor
        slack = b_alloc - min_bits
        idx = np.argsort(slack)
        for i in idx[:abs(deficit)]:
            if b_alloc[i] > min_bits:
                b_alloc[i] -= 1

    return b_alloc.astype(int)


# ============================================================================
# Hook factory
# ============================================================================
def make_k_quant_hook(layer_idx, bits, n_kv, d_head,
                      pca_bases=None, random_rot=None,
                      quant_fn=uniform_quant_col, wf_alloc=None):
    def hook_fn(module, input, output):
        k = output[0] if isinstance(output, tuple) else output
        k_np = k.detach().cpu().float().numpy()
        orig_shape = k_np.shape
        k_flat = k_np.reshape(-1, n_kv, d_head)

        for h in range(n_kv):
            Kh = k_flat[:, h, :]
            if pca_bases is not None and (layer_idx, h) in pca_bases:
                R = pca_bases[(layer_idx, h)]
            elif random_rot is not None:
                R = random_rot
            else:
                R = None

            Kr = Kh @ R if R is not None else Kh.copy()
            for j in range(d_head):
                b = int(wf_alloc[(layer_idx, h)][j]) if wf_alloc and (layer_idx, h) in wf_alloc else bits
                # WF allocator already clips to [min_bits, 4], so no hard clip needed here
                b = max(min(b, 4), 2 if quant_fn == gaussian_lloyd_col else 1)
                Kr[:, j] = quant_fn(Kr[:, j], b)
            k_flat[:, h, :] = Kr @ R.T if R is not None else Kr

        return torch.tensor(k_flat.reshape(orig_shape), dtype=k.dtype, device=k.device)
    return hook_fn


# ============================================================================
# Shared: model loading + calibration
# ============================================================================
def load_and_calibrate(model_name, device, cal_tokens=2048):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    all_ids = tokenizer.encode(text, return_tensors="pt", truncation=False)

    cfg = model.config
    n_kv = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    n_layers = cfg.num_hidden_layers
    d_head = cfg.hidden_size // cfg.num_attention_heads
    layers = model.model.layers if hasattr(model, 'model') else model.transformer.h

    # Calibration: extract pre-RoPE keys
    cal_ids = all_ids[:, :cal_tokens].to(device)
    pre_capture = {}
    hooks = []

    def make_cal_hook(li):
        def fn(mod, args_t, kwargs_t):
            hs = args_t[0] if args_t else kwargs_t.get('hidden_states')
            if hs is not None:
                k = mod.k_proj(hs)[0].detach().cpu().float().numpy().reshape(-1, n_kv, d_head)
                for h in range(n_kv):
                    pre_capture[(li, h)] = k[:, h, :]
        return fn

    for l in range(n_layers):
        hooks.append(layers[l].self_attn.register_forward_pre_hook(make_cal_hook(l), with_kwargs=True))

    with torch.no_grad():
        model(cal_ids, use_cache=False)
    for hook in hooks:
        hook.remove()

    # Compute per-head PCA
    pca_per_head = {}
    eigenvalues = {}
    for (l, h), K in pre_capture.items():
        Kc = K - K.mean(0)
        cov = Kc.T @ Kc / K.shape[0]
        eigvals, eigvecs = np.linalg.eigh(cov)
        pca_per_head[(l, h)] = eigvecs
        eigenvalues[(l, h)] = np.maximum(eigvals, 1e-10)

    # Compute shared PCA (KVTC style)
    all_keys = np.concatenate([K - K.mean(0) for K in pre_capture.values()], axis=0)
    cov_shared = all_keys.T @ all_keys / all_keys.shape[0]
    _, V_shared = np.linalg.eigh(cov_shared)
    pca_shared = {(l, h): V_shared for (l, h) in pre_capture}

    np.random.seed(42)
    R_random = np.linalg.qr(np.random.randn(d_head, d_head))[0]

    return {
        'model': model, 'tokenizer': tokenizer, 'all_ids': all_ids,
        'layers': layers, 'n_kv': n_kv, 'n_layers': n_layers, 'd_head': d_head,
        'pca_per_head': pca_per_head, 'pca_shared': pca_shared,
        'eigenvalues': eigenvalues, 'R_random': R_random,
        'pre_keys': pre_capture,
    }


def eval_ppl(model, all_ids, layers, n_layers, n_kv, d_head, device,
             bits, pca=None, rand_rot=None, quant_fn=uniform_quant_col,
             wf_alloc=None, max_tokens=50000, label=""):
    """Evaluate PPL with given quantization config."""
    chunk_len = 2048
    n_chunks = (min(all_ids.shape[1], max_tokens) - 1) // chunk_len
    hook_handles = []

    for l in range(n_layers):
        k_proj = layers[l].self_attn.k_proj
        hk = k_proj.register_forward_hook(
            make_k_quant_hook(l, bits, n_kv, d_head,
                              pca_bases=pca, random_rot=rand_rot,
                              quant_fn=quant_fn, wf_alloc=wf_alloc))
        hook_handles.append(hk)

    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for ci in range(n_chunks):
            start = ci * chunk_len
            end = start + chunk_len + 1
            if end > min(all_ids.shape[1], max_tokens):
                break
            inp = all_ids[:, start:end].to(device)
            target = inp[:, 1:]
            inp = inp[:, :-1]
            out = model(inp, use_cache=False)
            nll = nn.CrossEntropyLoss(reduction='sum')(out.logits[0], target[0]).item()
            total_nll += nll
            total_tokens += target.shape[1]

    for hk in hook_handles:
        hk.remove()

    ppl = np.exp(total_nll / total_tokens) if total_tokens > 0 else float('inf')
    print(f"    {label}: PPL={ppl:.4f} ({total_tokens} tokens)")
    return float(ppl)


# ============================================================================
# Exp V15-1: KVTC comparison
# ============================================================================
def run_v15_1(args):
    print(f"\n{'='*70}")
    print("V15-1: KVTC Comparison (Shared vs Per-Head PCA)")
    print(f"{'='*70}")

    data = load_and_calibrate(args.model, args.device)
    device = args.device
    results = {'experiment': 'V15-1', 'model': args.model}

    for bits in args.bits:
        print(f"\n  --- {bits}-bit ---")
        ppl_shared = eval_ppl(data['model'], data['all_ids'], data['layers'],
                              data['n_layers'], data['n_kv'], data['d_head'], device,
                              bits, pca=data['pca_shared'], label=f"shared_pca_{bits}b")
        ppl_perhead = eval_ppl(data['model'], data['all_ids'], data['layers'],
                               data['n_layers'], data['n_kv'], data['d_head'], device,
                               bits, pca=data['pca_per_head'], label=f"perhead_pca_{bits}b")
        results[f'{bits}bit'] = {
            'shared_pca': ppl_shared, 'perhead_pca': ppl_perhead,
            'gain_pct': (ppl_shared - ppl_perhead) / ppl_shared * 100
        }

    del data['model']; torch.cuda.empty_cache(); gc.collect()
    return results


# ============================================================================
# Exp V15-2: Calibration size sweep
# ============================================================================
def run_v15_2(args):
    print(f"\n{'='*70}")
    print("V15-2: Calibration Size Effect")
    print(f"{'='*70}")

    cal_sizes = [1024, 2048, 4096, 8192]
    results = {'experiment': 'V15-2', 'model': args.model}

    for cal_tokens in cal_sizes:
        print(f"\n  --- cal_tokens={cal_tokens} ---")
        data = load_and_calibrate(args.model, args.device, cal_tokens=cal_tokens)
        device = args.device

        for bits in [2]:  # Focus on 2-bit where Mistral exception occurs
            ppl_pca = eval_ppl(data['model'], data['all_ids'], data['layers'],
                               data['n_layers'], data['n_kv'], data['d_head'], device,
                               bits, pca=data['pca_per_head'],
                               label=f"pca_{bits}b_cal{cal_tokens}")
            ppl_turbo = eval_ppl(data['model'], data['all_ids'], data['layers'],
                                 data['n_layers'], data['n_kv'], data['d_head'], device,
                                 bits, rand_rot=data['R_random'],
                                 label=f"turbo_{bits}b_cal{cal_tokens}")
            results[f'cal{cal_tokens}_{bits}bit'] = {
                'pca': ppl_pca, 'turbo': ppl_turbo,
                'pca_wins': ppl_pca < ppl_turbo
            }

        del data['model']; torch.cuda.empty_cache(); gc.collect()

    return results


# ============================================================================
# Exp V15-3: Adaptive Lloyd-Max
# ============================================================================
def run_v15_3(args):
    print(f"\n{'='*70}")
    print("V15-3: Adaptive Lloyd-Max")
    print(f"{'='*70}")

    data = load_and_calibrate(args.model, args.device)
    device = args.device
    results = {'experiment': 'V15-3', 'model': args.model}

    for bits in args.bits:
        print(f"\n  --- {bits}-bit ---")
        ppl_uni = eval_ppl(data['model'], data['all_ids'], data['layers'],
                           data['n_layers'], data['n_kv'], data['d_head'], device,
                           bits, pca=data['pca_per_head'],
                           quant_fn=uniform_quant_col, label=f"pca_uni_{bits}b")
        ppl_gauss = eval_ppl(data['model'], data['all_ids'], data['layers'],
                             data['n_layers'], data['n_kv'], data['d_head'], device,
                             bits, pca=data['pca_per_head'],
                             quant_fn=gaussian_lloyd_col, label=f"pca_gauss_lloyd_{bits}b")
        ppl_adapt = eval_ppl(data['model'], data['all_ids'], data['layers'],
                             data['n_layers'], data['n_kv'], data['d_head'], device,
                             bits, pca=data['pca_per_head'],
                             quant_fn=adaptive_lloyd_col, label=f"pca_adapt_lloyd_{bits}b")
        results[f'{bits}bit'] = {
            'uniform': ppl_uni, 'gaussian_lloyd': ppl_gauss, 'adaptive_lloyd': ppl_adapt,
            'adapt_beats_gauss': ppl_adapt < ppl_gauss,
            'adapt_beats_uniform': ppl_adapt < ppl_uni,
        }

    del data['model']; torch.cuda.empty_cache(); gc.collect()
    return results


# ============================================================================
# Exp V15-4: WF with min-bit floor
# ============================================================================
def run_v15_4(args):
    print(f"\n{'='*70}")
    print("V15-4: Water-Filling with Min-Bit Floor")
    print(f"{'='*70}")

    data = load_and_calibrate(args.model, args.device)
    device = args.device
    results = {'experiment': 'V15-4', 'model': args.model}

    for bits in [2, 3]:
        print(f"\n  --- {bits}-bit ---")

        # Uniform baseline
        ppl_uni = eval_ppl(data['model'], data['all_ids'], data['layers'],
                           data['n_layers'], data['n_kv'], data['d_head'], device,
                           bits, pca=data['pca_per_head'],
                           label=f"pca_uni_{bits}b")

        # WF floor=1 (current, no floor)
        wf1 = {k: compute_wf_alloc(v, bits, min_bits=1) for k, v in data['eigenvalues'].items()}
        ppl_wf1 = eval_ppl(data['model'], data['all_ids'], data['layers'],
                           data['n_layers'], data['n_kv'], data['d_head'], device,
                           bits, pca=data['pca_per_head'], wf_alloc=wf1,
                           label=f"pca_wf_floor1_{bits}b")

        # WF floor=2
        wf2 = {k: compute_wf_alloc(v, bits, min_bits=2) for k, v in data['eigenvalues'].items()}
        ppl_wf2 = eval_ppl(data['model'], data['all_ids'], data['layers'],
                           data['n_layers'], data['n_kv'], data['d_head'], device,
                           bits, pca=data['pca_per_head'], wf_alloc=wf2,
                           label=f"pca_wf_floor2_{bits}b")

        results[f'{bits}bit'] = {
            'uniform': ppl_uni, 'wf_floor1': ppl_wf1, 'wf_floor2': ppl_wf2,
            'floor2_beats_floor1': ppl_wf2 < ppl_wf1,
            'floor2_beats_uniform': ppl_wf2 < ppl_uni,
        }

    del data['model']; torch.cuda.empty_cache(); gc.collect()
    return results


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="all", help="1,2,3,4 or all")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    args = parser.parse_args()

    exps = [1, 2, 3, 4] if args.exp == "all" else [int(x) for x in args.exp.split(",")]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.replace("/", "_")

    all_results = {}
    for exp_num in exps:
        t0 = time.time()
        if exp_num == 1:
            all_results['V15-1'] = run_v15_1(args)
        elif exp_num == 2:
            all_results['V15-2'] = run_v15_2(args)
        elif exp_num == 3:
            all_results['V15-3'] = run_v15_3(args)
        elif exp_num == 4:
            all_results['V15-4'] = run_v15_4(args)
        elapsed = time.time() - t0
        print(f"\n  V15-{exp_num} completed in {elapsed:.1f}s")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    outfile = OUTDIR / f"v15_{model_tag}_{timestamp}.json"
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved: {outfile}")


if __name__ == '__main__':
    main()
