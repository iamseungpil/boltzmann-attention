#!/usr/bin/env python3
"""
Baseline comparison: Pre-RoPE PCA vs KIVI-style vs GEAR-style quantization.
Implements key ideas from each baseline in our hook framework for fair comparison.

Usage:
  CUDA_VISIBLE_DEVICES=1 python run_baseline_comparison.py --model meta-llama/Llama-3.1-8B --bits 2 3 4
"""
from __future__ import annotations
import argparse, json, time, gc, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


# ============================================================================
# Quantization methods
# ============================================================================

def uniform_quant_col(col, bits):
    """Per-column min-max uniform (symmetric around range)."""
    nl = 2 ** bits
    vmin, vmax = col.min(), col.max()
    if vmax - vmin < 1e-10:
        return col.copy()
    s = (vmax - vmin) / (nl - 1)
    q = np.clip(np.round((col - vmin) / s).astype(int), 0, nl - 1)
    return q * s + vmin


def asymmetric_quant_col(col, bits):
    """Per-column asymmetric min-max (same as uniform but explicit name)."""
    return uniform_quant_col(col, bits)


def kivi_quant_key(K, bits):
    """KIVI-style key quantization: per-CHANNEL asymmetric.
    Each channel (dim j) is independently quantized using its own min/max.
    This is what KIVI does for keys."""
    d = K.shape[1]
    Kq = np.zeros_like(K)
    for j in range(d):
        Kq[:, j] = asymmetric_quant_col(K[:, j], bits)
    return Kq


def gear_quant_key(K, bits, rank=2):
    """GEAR-style key quantization: uniform quant + low-rank SVD residual.
    1. Uniform quantize K -> Kq
    2. Compute residual R = K - Kq
    3. SVD on R, keep top-rank components
    4. Final = Kq + R_lowrank
    """
    d = K.shape[1]
    # Step 1: uniform quantize per-channel
    Kq = np.zeros_like(K)
    for j in range(d):
        Kq[:, j] = uniform_quant_col(K[:, j], bits)

    # Step 2: residual
    R = K - Kq

    # Step 3: low-rank SVD of residual
    if R.shape[0] > rank and R.shape[1] > rank:
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        R_lr = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
    else:
        R_lr = R

    return Kq + R_lr


# ============================================================================
# Hook factories
# ============================================================================

def make_baseline_hook(layer_idx, bits, n_kv, d_head, method,
                       pca_bases=None, random_rot=None):
    """Create a k_proj hook for different baseline methods."""

    def hook_fn(module, input, output):
        k = output[0] if isinstance(output, tuple) else output
        k_np = k.detach().cpu().float().numpy()
        orig_shape = k_np.shape
        k_flat = k_np.reshape(-1, n_kv, d_head)

        for h in range(n_kv):
            Kh = k_flat[:, h, :]

            if method == 'no_rot':
                # No rotation, per-channel uniform (our "no rotation" baseline)
                Kh_q = kivi_quant_key(Kh, bits)

            elif method == 'kivi_style':
                # KIVI: per-channel asymmetric, no rotation
                # Same as no_rot but explicit
                Kh_q = kivi_quant_key(Kh, bits)

            elif method == 'gear_style':
                # GEAR: uniform + low-rank SVD residual
                Kh_q = gear_quant_key(Kh, bits, rank=2)

            elif method == 'random_rot':
                # Haar random rotation + per-channel uniform
                Kr = Kh @ random_rot
                Kq = kivi_quant_key(Kr, bits)
                Kh_q = Kq @ random_rot.T

            elif method == 'pre_rope_pca':
                # Our method: Pre-RoPE PCA + per-channel uniform
                R = pca_bases.get((layer_idx, h))
                if R is not None:
                    Kr = Kh @ R
                    Kq = kivi_quant_key(Kr, bits)
                    Kh_q = Kq @ R.T
                else:
                    Kh_q = kivi_quant_key(Kh, bits)

            else:
                raise ValueError(f"Unknown method: {method}")

            k_flat[:, h, :] = Kh_q

        return torch.tensor(k_flat.reshape(orig_shape), dtype=k.dtype, device=k.device)

    return hook_fn


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    device = "cuda:0"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"\n{'#'*70}")
    print(f"# Baseline Comparison: {args.model}")
    print(f"{'#'*70}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    all_ids = tokenizer.encode(text, return_tensors="pt", truncation=False)

    cfg = model.config
    n_kv = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    n_layers = cfg.num_hidden_layers
    d_head = cfg.hidden_size // cfg.num_attention_heads
    layers = model.model.layers if hasattr(model, 'model') else model.transformer.h

    # Calibration for PCA
    cal_ids = all_ids[:, :2048].to(device)
    pre_capture = {}
    cal_hooks = []
    def make_cal_hook(li):
        def fn(mod, a, kw):
            hs = a[0] if a else kw.get('hidden_states')
            if hs is not None:
                k = mod.k_proj(hs)[0].detach().cpu().float().numpy().reshape(-1, n_kv, d_head)
                for h in range(n_kv):
                    pre_capture[(li, h)] = k[:, h, :]
        return fn
    for l in range(n_layers):
        cal_hooks.append(layers[l].self_attn.register_forward_pre_hook(make_cal_hook(l), with_kwargs=True))
    with torch.no_grad():
        model(cal_ids, use_cache=False)
    for hook in cal_hooks:
        hook.remove()

    pca_bases = {}
    for (l, h), K in pre_capture.items():
        Kc = K - K.mean(0)
        _, V = np.linalg.eigh(Kc.T @ Kc / K.shape[0])
        pca_bases[(l, h)] = V

    np.random.seed(42)
    R_random = np.linalg.qr(np.random.randn(d_head, d_head))[0]

    # PPL evaluation
    chunk_len, max_tokens = 2048, min(all_ids.shape[1], 50000)
    n_chunks = (max_tokens - 1) // chunk_len

    def eval_ppl(method_name, bits):
        hks = []
        for l in range(n_layers):
            hks.append(layers[l].self_attn.k_proj.register_forward_hook(
                make_baseline_hook(l, bits, n_kv, d_head, method_name,
                                   pca_bases=pca_bases, random_rot=R_random)))
        nll, tok = 0.0, 0
        with torch.no_grad():
            for ci in range(n_chunks):
                s = ci * chunk_len
                e = s + chunk_len + 1
                if e > max_tokens: break
                inp = all_ids[:, s:e].to(device)
                out = model(inp[:, :-1], use_cache=False)
                nll += nn.CrossEntropyLoss(reduction='sum')(out.logits[0], inp[0, 1:]).item()
                tok += inp.shape[1] - 1
        for hk in hks:
            hk.remove()
        ppl = np.exp(nll / tok)
        print(f"    {method_name}_{bits}b: PPL={ppl:.4f} ({tok} tokens)")
        return float(ppl)

    # FP16 baseline (no hooks)
    print("\n  FP16 baseline...")
    nll, tok = 0.0, 0
    with torch.no_grad():
        for ci in range(n_chunks):
            s = ci * chunk_len
            e = s + chunk_len + 1
            if e > max_tokens: break
            inp = all_ids[:, s:e].to(device)
            out = model(inp[:, :-1], use_cache=False)
            nll += nn.CrossEntropyLoss(reduction='sum')(out.logits[0], inp[0, 1:]).item()
            tok += inp.shape[1] - 1
    fp16_ppl = np.exp(nll / tok)
    print(f"    FP16: PPL={fp16_ppl:.4f}")

    # Methods to compare
    methods = ['kivi_style', 'gear_style', 'random_rot', 'pre_rope_pca']

    results = {'model': args.model, 'fp16': fp16_ppl, 'methods': {}}
    for bits in args.bits:
        print(f"\n  --- {bits}-bit ---")
        for method in methods:
            ppl = eval_ppl(method, bits)
            results['methods'][f'{method}_{bits}bit'] = ppl

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.model.replace("/", "_")
    outf = Path(args.output_dir) / f"baseline_comparison_{tag}_{ts}.json"
    with open(outf, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outf}")

    del model; torch.cuda.empty_cache(); gc.collect()


if __name__ == '__main__':
    main()
