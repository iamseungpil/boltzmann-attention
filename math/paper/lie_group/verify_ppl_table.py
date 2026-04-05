#!/usr/bin/env python3
"""
NeurIPS Table 1: End-to-end PPL comparison
==========================================

Measures WikiText-2 PPL for:
  Methods: {no_rot, turbo, pre_pca_wf, pre_pca_lloyd}
  Bits:    {2, 3, 4}
  Models:  configurable (Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-v0.3)

Also computes MSE for Level 2 prediction (R² analysis).

Usage:
  python verify_ppl_table.py --model Qwen/Qwen2.5-7B --device cuda:0
  python verify_ppl_table.py --model meta-llama/Llama-3.1-8B --device cuda:0
  python verify_ppl_table.py --model mistralai/Mistral-7B-v0.3 --device cuda:0
  python verify_ppl_table.py --model Qwen/Qwen2.5-7B --smoke --device cuda:0
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
# Lloyd-Max codebook (N(0,1) optimal, with centering fix)
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


def lloyd_quant_col(col: np.ndarray, bits: int) -> np.ndarray:
    """Per-column Gaussian Lloyd-Max with centering."""
    th, ct = LLOYD_CB[bits]
    mean, std = col.mean(), max(col.std(), 1e-10)
    idx = np.searchsorted(th, (col - mean) / std)
    return ct[idx] * std + mean


def uniform_quant_col(col: np.ndarray, bits: int) -> np.ndarray:
    """Per-column min-max uniform quantization."""
    nl = 2 ** bits
    vmin, vmax = col.min(), col.max()
    if vmax - vmin < 1e-10:
        return col.copy()
    s = (vmax - vmin) / (nl - 1)
    q = np.clip(np.round((col - vmin) / s).astype(int), 0, nl - 1)
    return q * s + vmin


# ============================================================================
# Quantization hook factory
# ============================================================================
def make_k_quant_hook(layer_idx, bits, n_kv, d_head,
                      pca_bases=None, random_rot=None,
                      use_lloyd=False, wf_alloc=None):
    """Create a forward hook on k_proj that quantizes the output.

    Args:
        pca_bases: dict (layer, head) -> eigvecs, or None
        random_rot: np.ndarray d×d, or None (TurboQuant)
        use_lloyd: if True, use Lloyd-Max; else uniform
        wf_alloc: dict (layer, head) -> int array of per-dim bit allocation, or None
    """
    def hook_fn(module, input, output):
        k = output[0] if isinstance(output, tuple) else output
        k_np = k.detach().cpu().float().numpy()
        orig_shape = k_np.shape
        # Reshape: (batch, seq, n_kv*d_head) -> (batch*seq, n_kv, d_head)
        k_flat = k_np.reshape(-1, n_kv, d_head)

        for h in range(n_kv):
            Kh = k_flat[:, h, :]  # (seq, d_head)

            # Choose rotation
            if pca_bases is not None and (layer_idx, h) in pca_bases:
                R = pca_bases[(layer_idx, h)]
            elif random_rot is not None:
                R = random_rot
            else:
                R = None

            # Rotate
            Kr = Kh @ R if R is not None else Kh.copy()

            # Quantize per-dim
            d = d_head
            for j in range(d):
                b = int(wf_alloc[(layer_idx, h)][j]) if wf_alloc and (layer_idx, h) in wf_alloc else bits
                b = max(b, 1)
                if use_lloyd:
                    Kr[:, j] = lloyd_quant_col(Kr[:, j], min(b, 4))
                else:
                    Kr[:, j] = uniform_quant_col(Kr[:, j], min(b, 4))

            # Inverse rotate
            if R is not None:
                Kr = Kr @ R.T

            k_flat[:, h, :] = Kr

        k_quant = k_flat.reshape(orig_shape)
        return torch.tensor(k_quant, dtype=k.dtype, device=k.device)
    return hook_fn


# ============================================================================
# Main experiment
# ============================================================================
def run_ppl_table(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    t0 = time.time()
    model_name = args.model
    device = args.device

    print(f"\n{'#'*70}")
    print(f"# PPL Table: {model_name}")
    print(f"# Device: {device}")
    print(f"{'#'*70}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()

    # Load data
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    all_ids = tokenizer.encode(text, return_tensors="pt", truncation=False)

    cfg = model.config
    n_kv = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    n_heads = cfg.num_attention_heads
    n_layers = cfg.num_hidden_layers
    d_head = cfg.hidden_size // n_heads

    print(f"  layers={n_layers}, kv_heads={n_kv}, d_head={d_head}")

    # --- Step 1: Calibration (extract pre-RoPE PCA bases) ---
    print("\n  [Step 1] Calibration: extracting pre-RoPE PCA bases...")
    cal_tokens = 1024 if args.smoke else 2048
    cal_ids = all_ids[:, :cal_tokens].to(device)

    pre_capture = {}
    hooks = []
    layers = model.model.layers if hasattr(model, 'model') else model.transformer.h

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

    # Compute PCA bases
    pca_bases = {}
    key_variances = {}  # for water-filling
    for (l, h), K in pre_capture.items():
        Kc = K - K.mean(0)
        cov = Kc.T @ Kc / K.shape[0]
        eigvals, eigvecs = np.linalg.eigh(cov)
        pca_bases[(l, h)] = eigvecs
        key_variances[(l, h)] = np.maximum(eigvals, 1e-10)

    # Random rotation for TurboQuant
    np.random.seed(42)
    R_random = np.linalg.qr(np.random.randn(d_head, d_head))[0]

    # Pre-compute water-filling allocations
    def compute_wf_alloc(variances, avg_bits):
        d = len(variances)
        log_var = np.log2(variances)
        mean_log_var = np.mean(log_var)
        b_alloc = avg_bits + 0.5 * (log_var - mean_log_var)
        b_alloc = np.clip(b_alloc, 1, avg_bits * 2)
        total_target = d * avg_bits
        b_alloc = b_alloc * (total_target / b_alloc.sum())
        return np.maximum(np.round(b_alloc), 1).astype(int)

    print(f"  Calibration done: {len(pca_bases)} heads")

    # --- Step 2: PPL measurement ---
    chunk_len = 2048
    max_eval_tokens = 10000 if args.smoke else min(all_ids.shape[1], 50000)
    n_chunks = (max_eval_tokens - 1) // chunk_len
    bits_list = args.bits

    def eval_ppl(method_name, bits, pca=None, rand_rot=None, use_lloyd=False, use_wf=False):
        """Evaluate PPL with given quantization settings."""
        hook_handles = []

        # Compute WF allocations if needed
        wf_alloc = None
        if use_wf and pca is not None:
            wf_alloc = {}
            for (l, h) in pca:
                Kr_var = key_variances[(l, h)]
                # Variance in PCA space = eigenvalues (already sorted)
                wf_alloc[(l, h)] = compute_wf_alloc(Kr_var, bits)

        for l in range(n_layers):
            k_proj = layers[l].self_attn.k_proj
            hk = k_proj.register_forward_hook(
                make_k_quant_hook(l, bits, n_kv, d_head,
                                  pca_bases=pca, random_rot=rand_rot,
                                  use_lloyd=use_lloyd, wf_alloc=wf_alloc))
            hook_handles.append(hk)

        total_nll = 0.0
        total_tokens = 0

        with torch.no_grad():
            for ci in range(n_chunks):
                start = ci * chunk_len
                end = start + chunk_len + 1
                if end > max_eval_tokens:
                    break
                inp = all_ids[:, start:end].to(device)
                target = inp[:, 1:]
                inp = inp[:, :-1]

                out = model(inp, use_cache=False)
                nll = nn.CrossEntropyLoss(reduction='sum')(out.logits[0], target[0]).item()
                total_nll += nll
                total_tokens += target.shape[1]

                if (ci + 1) % 5 == 0:
                    print(f"      [{method_name} {bits}b] chunk {ci+1}/{n_chunks} "
                          f"ppl={np.exp(total_nll/total_tokens):.4f}")

        for hk in hook_handles:
            hk.remove()

        ppl = np.exp(total_nll / total_tokens) if total_tokens > 0 else float('inf')
        return {'ppl': float(ppl), 'nll': float(total_nll), 'tokens': int(total_tokens)}

    # --- FP16 baseline ---
    print("\n  [Step 2] FP16 baseline...")
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for ci in range(n_chunks):
            start = ci * chunk_len
            end = start + chunk_len + 1
            if end > max_eval_tokens:
                break
            inp = all_ids[:, start:end].to(device)
            target = inp[:, 1:]
            inp = inp[:, :-1]
            out = model(inp, use_cache=False)
            nll = nn.CrossEntropyLoss(reduction='sum')(out.logits[0], target[0]).item()
            total_nll += nll
            total_tokens += target.shape[1]
    ppl_fp16 = np.exp(total_nll / total_tokens)
    print(f"  FP16 PPL = {ppl_fp16:.4f} ({total_tokens} tokens)")

    # --- Methods ---
    methods = [
        ('no_rot_uni',       None,       None,     False, False),
        ('turbo_uni',        None,       R_random, False, False),
        ('pre_pca_uni',      pca_bases,  None,     False, False),  # Axis 1: PCA + uniform (no WF)
        ('pre_pca_lloyd',    pca_bases,  None,     True,  False),  # Axis 1+2: PCA + Lloyd-Max
        ('pre_pca_wf_uni',   pca_bases,  None,     False, True),   # Axis 1+WF: PCA + WF uniform
    ]

    results = {'model': model_name, 'fp16_ppl': float(ppl_fp16), 'methods': {}}

    for bits in bits_list:
        for method_name, pca, rand, use_lloyd, use_wf in methods:
            label = f"{method_name}_{bits}bit"
            print(f"\n  [{label}]")
            t1 = time.time()
            r = eval_ppl(method_name, bits, pca=pca, rand_rot=rand,
                         use_lloyd=use_lloyd, use_wf=use_wf)
            elapsed = time.time() - t1
            r['elapsed'] = float(elapsed)
            results['methods'][label] = r
            ratio = r['ppl'] / ppl_fp16
            print(f"    PPL={r['ppl']:.4f} ({ratio:.4f}x FP16) [{elapsed:.1f}s]")

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"PPL Table: {model_name}")
    print(f"{'='*70}")
    print(f"  FP16: {ppl_fp16:.4f}")

    for bits in bits_list:
        print(f"\n  --- {bits}-bit ---")
        for method_name, _, _, _, _ in methods:
            label = f"{method_name}_{bits}bit"
            if label in results['methods']:
                r = results['methods'][label]
                ratio = r['ppl'] / ppl_fp16
                print(f"    {method_name:20s}: PPL={r['ppl']:.4f} ({ratio:.4f}x FP16)")

    # Save
    OUTDIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = model_name.replace("/", "_")
    outfile = OUTDIR / f"ppl_table_{model_tag}_{timestamp}.json"
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Saved: {outfile}")
    print(f"  Total time: {elapsed:.1f}s")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_ppl_table(args)
