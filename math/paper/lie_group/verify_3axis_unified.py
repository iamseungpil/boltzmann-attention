#!/usr/bin/env python3
"""
Unified 3-Axis Verification for NeurIPS 2026 Lie Group Framework
================================================================

Verifies all three axes of the optimality claim:
  Axis 1 (Rotation):  Pre-RoPE PCA is MSE-optimal within Class C
  Axis 2 (Quantizer):  Per-channel Gaussian Lloyd-Max beats uniform
  Axis 3 (Long-context): NIAH retrieval at 4K/8K/16K contexts

Supports: Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-v0.3, Qwen2.5-1.5B

Usage:
  # Full verification on one model
  python verify_3axis_unified.py --model Qwen/Qwen2.5-7B --device cuda:0

  # Axis 1 only (fast, ~2 min)
  python verify_3axis_unified.py --model Qwen/Qwen2.5-7B --axis 1 --device cuda:0

  # Axis 2 only (Lloyd-Max fix verification)
  python verify_3axis_unified.py --model Qwen/Qwen2.5-7B --axis 2 --device cuda:0

  # Axis 3 only (NIAH, slow)
  python verify_3axis_unified.py --model Qwen/Qwen2.5-7B --axis 3 --device cuda:0

  # Smoke test (reduced tokens, fewer heads)
  python verify_3axis_unified.py --model Qwen/Qwen2.5-7B --smoke --device cuda:0
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

OUTDIR = Path(__file__).parent / "verification_results"

# ============================================================================
# Gaussian Lloyd-Max Codebook (N(0,1) optimal)
# ============================================================================
LLOYD_MAX_CB = {
    2: {
        'thresholds': np.array([-0.9816, 0.0, 0.9816]),
        'centroids':  np.array([-1.5104, -0.4528, 0.4528, 1.5104]),
    },
    3: {
        'thresholds': np.array([-1.7480, -1.0500, -0.5006, 0.0, 0.5006, 1.0500, 1.7480]),
        'centroids':  np.array([-2.1519, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1519]),
    },
    4: {
        'thresholds': np.array([-2.4008, -1.8435, -1.4370, -1.0993, -0.7996, -0.5224, -0.2582, 0.0,
                                  0.2582, 0.5224, 0.7996, 1.0993, 1.4370, 1.8435, 2.4008]),
        'centroids':  np.array([-2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3881, -0.1284,
                                  0.1284, 0.3881, 0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326]),
    },
}


def gaussian_lloyd_max(x: np.ndarray, bits: int) -> np.ndarray:
    """Per-channel Gaussian Lloyd-Max quantization.

    CRITICAL FIX: Centers data before applying N(0,1) codebook, then restores mean.
    Previous version did not center, causing all values to map to one bin when mean != 0.
    """
    cb = LLOYD_MAX_CB[bits]
    th, ct = cb['thresholds'], cb['centroids']

    # Step 1: Center and scale
    mean = np.mean(x)
    std = max(np.std(x), 1e-10)
    x_norm = (x - mean) / std

    # Step 2: Apply N(0,1) codebook
    idx = np.searchsorted(th, x_norm)
    x_quant_norm = ct[idx]

    # Step 3: Restore scale and mean
    return x_quant_norm * std + mean


def uniform_quant(x: np.ndarray, bits: int) -> np.ndarray:
    """Standard min-max uniform quantization."""
    n_levels = 2 ** bits
    vmin, vmax = x.min(), x.max()
    if vmax - vmin < 1e-10:
        return x.copy()
    scale = (vmax - vmin) / (n_levels - 1)
    q = np.clip(np.round((x - vmin) / scale).astype(int), 0, n_levels - 1)
    return q * scale + vmin


# ============================================================================
# Model Loading & Key Extraction
# ============================================================================
def load_model_and_extract_keys(
    model_name: str,
    device: str,
    max_tokens: int = 2048,
    layer_stride: int = 1,
) -> dict:
    """Load model, extract pre-RoPE and post-RoPE keys per head."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"\n  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    all_ids = tokenizer.encode(text, return_tensors="pt", truncation=False)

    cfg = model.config
    n_kv = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    n_heads = cfg.num_attention_heads
    n_layers = cfg.num_hidden_layers
    d_head = cfg.hidden_size // n_heads
    G = n_heads // n_kv  # GQA group size

    cal_ids = all_ids[:, :max_tokens].to(device)
    print(f"  Config: layers={n_layers}, kv_heads={n_kv}, d_head={d_head}, "
          f"GQA_G={G}, tokens={cal_ids.shape[1]}")

    # Hook to capture pre-RoPE K
    pre_capture = {}
    hooks = []

    # Determine attention module structure
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    def make_hook(li):
        def fn(mod, args, kwargs):
            hs = args[0] if args else kwargs.get('hidden_states')
            if hs is not None:
                k = mod.k_proj(hs)[0].detach().cpu().float().numpy().reshape(-1, n_kv, d_head)
                for h in range(n_kv):
                    pre_capture[(li, h)] = k[:, h, :]
        return fn

    sampled_layers = list(range(0, n_layers, layer_stride))
    for l in sampled_layers:
        attn = layers[l].self_attn
        hooks.append(attn.register_forward_pre_hook(make_hook(l), with_kwargs=True))

    post_keys = {}
    sampled_set = set(sampled_layers)
    with torch.no_grad():
        out = model(cal_ids, use_cache=True)

    # Iterate past_key_values (DynamicCache in transformers v5.x is not subscriptable)
    for l, item in enumerate(out.past_key_values):
        if l not in sampled_set:
            continue
        K = (item[0] if isinstance(item, tuple) else item.keys)[0].cpu().float().numpy()
        for h in range(n_kv):
            post_keys[(l, h)] = K[h]

    for hook in hooks:
        hook.remove()

    pre_keys = dict(pre_capture)

    # Compute PCA bases
    pca_pre = {}
    for (l, h), K in pre_keys.items():
        Kc = K - K.mean(0)
        cov = Kc.T @ Kc / K.shape[0]
        eigvals, eigvecs = np.linalg.eigh(cov)
        pca_pre[(l, h)] = eigvecs  # columns are eigenvectors, ascending order

    # Random rotation (fixed seed)
    np.random.seed(42)
    R_random = np.linalg.qr(np.random.randn(d_head, d_head))[0]

    info = {
        'model_name': model_name,
        'n_layers': n_layers,
        'n_kv': n_kv,
        'n_heads': n_heads,
        'd_head': d_head,
        'G': G,
        'sampled_layers': sampled_layers,
        'n_tokens': cal_ids.shape[1],
    }

    result = {
        'model': model,
        'tokenizer': tokenizer,
        'all_ids': all_ids,
        'pre_keys': pre_keys,
        'post_keys': post_keys,
        'pca_pre': pca_pre,
        'R_random': R_random,
        'info': info,
    }

    return result


# ============================================================================
# Axis 1: Rotation MSE Verification
# ============================================================================
def verify_axis1_rotation(data: dict, bits_list: List[int] = [2, 3, 4]) -> dict:
    """Verify Pre-RoPE PCA+WF is MSE-optimal within Class C."""
    print(f"\n{'='*70}")
    print("AXIS 1: Pre-RoPE PCA Rotation Verification (MSE)")
    print(f"{'='*70}")

    pre_keys = data['pre_keys']
    post_keys = data['post_keys']
    pca_pre = data['pca_pre']
    R_random = data['R_random']
    info = data['info']
    d = info['d_head']

    results = []

    for (l, h) in sorted(pre_keys.keys()):
        K_pre = pre_keys[(l, h)]
        K_post = post_keys[(l, h)]
        V = pca_pre[(l, h)]

        # Compute R_aniso
        Kc = K_pre - K_pre.mean(0)
        cov = Kc.T @ Kc / K_pre.shape[0]
        eigvals = np.maximum(np.linalg.eigvalsh(cov), 1e-10)
        G_pca = np.exp(np.mean(np.log(eigvals)))
        G_uniform = np.mean(eigvals)
        R_aniso = G_uniform / G_pca

        for bits in bits_list:
            n_levels = 2 ** bits

            def quant_mse(K, rot=None, water_fill=False):
                Kr = K @ rot if rot is not None else K.copy()
                Kq = np.zeros_like(Kr)

                if water_fill and rot is not None:
                    # Water-filling bit allocation
                    variances = np.maximum(np.var(Kr, axis=0), 1e-10)
                    log_var = np.log2(variances)
                    mean_log_var = np.mean(log_var)
                    b_alloc = bits + 0.5 * (log_var - mean_log_var)
                    b_alloc = np.clip(b_alloc, 1, bits * 2)
                    total_bits = d * bits
                    b_alloc = b_alloc * (total_bits / b_alloc.sum())
                    b_alloc = np.maximum(np.round(b_alloc), 1).astype(int)
                    for j in range(d):
                        Kq[:, j] = uniform_quant(Kr[:, j], int(b_alloc[j]))
                else:
                    for j in range(d):
                        Kq[:, j] = uniform_quant(Kr[:, j], bits)

                if rot is not None:
                    Kq = Kq @ rot.T
                return float(np.mean((K - Kq) ** 2))

            # Compute post-RoPE PCA once per head (not per bit)
            cov_post = (K_post - K_post.mean(0)).T @ (K_post - K_post.mean(0)) / K_post.shape[0]
            V_post = np.linalg.eigh(cov_post)[1]

            results.append({
                'l': l, 'h': h, 'bits': bits,
                'R_aniso': float(R_aniso),
                'identity':       quant_mse(K_post),
                'turbo':          quant_mse(K_post, R_random),
                'post_pca_uni':   quant_mse(K_post, V_post),
                'post_pca_wf':    quant_mse(K_post, V_post, water_fill=True),
                'pre_pca_uni':    quant_mse(K_pre, V),
                'pre_pca_wf':     quant_mse(K_pre, V, water_fill=True),
            })

    # Summary
    methods = ['identity', 'turbo', 'post_pca_uni', 'post_pca_wf', 'pre_pca_uni', 'pre_pca_wf']
    labels = ['No rotation', 'TurboQuant', 'PostRoPE PCA+uni', 'PostRoPE PCA+WF',
              'PreRoPE PCA+uni', 'PreRoPE PCA+WF']

    summary = {}
    for bits in bits_list:
        br = [r for r in results if r['bits'] == bits]
        vals = {m: np.mean([r[m] for r in br]) for m in methods}
        best = min(vals.values())
        R_aniso_avg = np.mean([r['R_aniso'] for r in br])

        print(f"\n--- {bits}-bit (avg over {len(br)} heads) ---")
        for m, label in zip(methods, labels):
            v = vals[m]
            mark = ' *** BEST ***' if abs(v - best) / max(best, 1e-10) < 0.01 else ''
            print(f"  {label:25s}: MSE={v:.6f} (x{v/best:.2f}){mark}")

        ratio_turbo = vals['turbo'] / max(vals['pre_pca_wf'], 1e-10)
        print(f"\n  R_aniso = {R_aniso_avg:.2f}")
        print(f"  Theory predicts PreRoPE PCA+WF {R_aniso_avg:.2f}x better than TurboQuant")
        print(f"  Actual: {ratio_turbo:.2f}x")

        # Key checks
        pre_best = vals['pre_pca_wf'] <= vals['pre_pca_uni'] <= vals['turbo']
        post_2bit_loss = vals['post_pca_wf'] > vals['turbo'] if bits == 2 else None

        summary[bits] = {
            'vals': {m: float(v) for m, v in vals.items()},
            'R_aniso': float(R_aniso_avg),
            'pre_pca_wf_is_best': bool(abs(vals['pre_pca_wf'] - best) / max(best, 1e-10) < 0.01),
            'turbo_over_pre_ratio': float(ratio_turbo),
            'post_2bit_loses_to_turbo': post_2bit_loss,
        }

    # Verification checks
    checks = {
        'pre_pca_wf_best_all_bits': all(s['pre_pca_wf_is_best'] for s in summary.values()),
        'turbo_worse_than_pre_all_bits': all(
            s['vals']['turbo'] > s['vals']['pre_pca_wf'] for s in summary.values()),
        'post_2bit_loses_to_turbo': summary.get(2, {}).get('post_2bit_loses_to_turbo', None),
    }

    print(f"\n  VERIFICATION CHECKS:")
    print(f"    Pre-RoPE PCA+WF is BEST at all bits: {'PASS' if checks['pre_pca_wf_best_all_bits'] else 'FAIL'}")
    print(f"    TurboQuant worse than Pre-RoPE at all bits: {'PASS' if checks['turbo_worse_than_pre_all_bits'] else 'FAIL'}")
    if checks['post_2bit_loses_to_turbo'] is not None:
        print(f"    Post-RoPE PCA+WF loses to TurboQuant at 2-bit: "
              f"{'PASS (confirms Cor 6.16.4d)' if checks['post_2bit_loses_to_turbo'] else 'UNEXPECTED'}")

    return {'per_head': results, 'summary': summary, 'checks': checks}


# ============================================================================
# Axis 2: Lloyd-Max Quantizer Verification
# ============================================================================
def verify_axis2_lloydmax(data: dict, bits_list: List[int] = [2, 3, 4]) -> dict:
    """Verify that per-channel Gaussian Lloyd-Max beats uniform quantization.

    CRITICAL: This uses the FIXED Lloyd-Max with proper centering.
    """
    print(f"\n{'='*70}")
    print("AXIS 2: Lloyd-Max Quantizer Verification (Fixed: with centering)")
    print(f"{'='*70}")

    pre_keys = data['pre_keys']
    post_keys = data['post_keys']
    pca_pre = data['pca_pre']
    R_random = data['R_random']
    info = data['info']
    d = info['d_head']

    results = []

    for (l, h) in sorted(pre_keys.keys()):
        K_pre = pre_keys[(l, h)]
        V = pca_pre[(l, h)]

        for bits in bits_list:
            def compute_mse(K, rot=None, use_lloyd=False):
                Kr = K @ rot if rot is not None else K.copy()
                Kq = np.zeros_like(Kr)
                for j in range(d):
                    if use_lloyd:
                        Kq[:, j] = gaussian_lloyd_max(Kr[:, j], bits)
                    else:
                        Kq[:, j] = uniform_quant(Kr[:, j], bits)
                if rot is not None:
                    Kq = Kq @ rot.T
                return float(np.mean((K - Kq) ** 2))

            results.append({
                'l': l, 'h': h, 'bits': bits,
                'pre_pca_uni':    compute_mse(K_pre, V, use_lloyd=False),
                'pre_pca_lloyd':  compute_mse(K_pre, V, use_lloyd=True),
                'turbo_uni':      compute_mse(K_pre, R_random, use_lloyd=False),
                'turbo_lloyd':    compute_mse(K_pre, R_random, use_lloyd=True),
                'norot_uni':      compute_mse(K_pre, use_lloyd=False),
                'norot_lloyd':    compute_mse(K_pre, use_lloyd=True),
            })

    # Summary
    methods = ['norot_uni', 'norot_lloyd', 'turbo_uni', 'turbo_lloyd', 'pre_pca_uni', 'pre_pca_lloyd']
    labels = ['NoRot+Uniform', 'NoRot+Lloyd', 'Turbo+Uniform', 'Turbo+Lloyd',
              'PrePCA+Uniform', 'PrePCA+Lloyd']

    summary = {}
    for bits in bits_list:
        br = [r for r in results if r['bits'] == bits]
        vals = {m: np.mean([r[m] for r in br]) for m in methods}
        best = min(vals.values())

        print(f"\n--- {bits}-bit (avg over {len(br)} heads) ---")
        for m, label in zip(methods, labels):
            v = vals[m]
            mark = ' *** BEST ***' if abs(v - best) / max(best, 1e-10) < 0.01 else ''
            print(f"  {label:25s}: MSE={v:.6f} (x{v/best:.2f}){mark}")

        # Lloyd-Max gain per rotation
        gains = {}
        for prefix in ['norot', 'turbo', 'pre_pca']:
            uni_key = f'{prefix}_uni'
            lloyd_key = f'{prefix}_lloyd'
            gain = vals[uni_key] / max(vals[lloyd_key], 1e-10)
            gains[prefix] = gain
            print(f"  Lloyd gain ({prefix}): {gain:.3f}x {'(Lloyd BETTER)' if gain > 1 else '(Uniform BETTER)'}")

        summary[bits] = {
            'vals': {m: float(v) for m, v in vals.items()},
            'lloyd_gains': {k: float(v) for k, v in gains.items()},
            'lloyd_beats_uniform': all(g > 1.0 for g in gains.values()),
            'best_method': labels[methods.index(min(vals, key=vals.get))],
        }

    checks = {
        'lloyd_beats_uniform_all_bits': all(s['lloyd_beats_uniform'] for s in summary.values()),
        'pre_pca_lloyd_is_overall_best': all(
            s['vals']['pre_pca_lloyd'] == min(s['vals'].values()) or
            abs(s['vals']['pre_pca_lloyd'] - min(s['vals'].values())) / max(min(s['vals'].values()), 1e-10) < 0.05
            for s in summary.values()
        ),
    }

    print(f"\n  VERIFICATION CHECKS:")
    print(f"    Lloyd-Max beats Uniform at all bits+rotations: "
          f"{'PASS' if checks['lloyd_beats_uniform_all_bits'] else 'FAIL'}")
    print(f"    PrePCA+Lloyd is overall best (within 5%): "
          f"{'PASS' if checks['pre_pca_lloyd_is_overall_best'] else 'FAIL'}")

    return {'per_head': results, 'summary': summary, 'checks': checks}


# ============================================================================
# Axis 3: NIAH Long-Context Verification
# ============================================================================
def verify_axis3_niah(data: dict, ctx_lengths: List[int] = [4096],
                      bits_list: List[int] = [2, 3], n_trials: int = 10) -> dict:
    """Verify quantization impact on Needle-in-a-Haystack retrieval."""
    import torch

    print(f"\n{'='*70}")
    print(f"AXIS 3: NIAH Verification (contexts: {ctx_lengths})")
    print(f"{'='*70}")

    model = data['model']
    tokenizer = data['tokenizer']
    pca_pre = data['pca_pre']
    R_random = data['R_random']
    info = data['info']
    device = next(model.parameters()).device
    n_kv, n_layers, d = info['n_kv'], info['n_layers'], info['d_head']

    filler = "The grass is green and the sky is blue. Today is a wonderful day. "
    depths = [0.0, 0.25, 0.5, 0.75, 1.0]

    def make_quant_hook(layer_idx, bits, rot_dict=None, rand_rot=None):
        def fn(mod, inp, out):
            k = out[0] if isinstance(out, tuple) else out
            k_np = k.detach().cpu().float().numpy()
            orig_shape = k_np.shape
            k_reshaped = k_np.reshape(-1, n_kv, d)
            for h in range(n_kv):
                Kh = k_reshaped[:, h, :]
                R = rot_dict.get((layer_idx, h)) if rot_dict else rand_rot
                if R is None:
                    # No rotation for this layer/head (e.g., not in sampled_layers)
                    # Still quantize without rotation for fair comparison
                    for j in range(d):
                        Kh[:, j] = uniform_quant(Kh[:, j], bits)
                    k_reshaped[:, h, :] = Kh
                else:
                    Kr = Kh @ R
                    for j in range(d):
                        Kr[:, j] = uniform_quant(Kr[:, j], bits)
                    k_reshaped[:, h, :] = Kr @ R.T
            return torch.tensor(k_reshaped.reshape(orig_shape), dtype=k.dtype, device=k.device)
        return fn

    def find_k_proj(layer):
        if hasattr(layer, 'self_attn'):
            return layer.self_attn.k_proj
        elif hasattr(layer, 'attn'):
            return layer.attn.k_proj
        raise AttributeError(f"Cannot find k_proj in {type(layer)}")

    def test_method(name, bits, rot_dict=None, rand_rot=None, ctx_len=4096):
        hks = []
        if name != 'fp16':
            layers = model.model.layers if hasattr(model, 'model') else model.transformer.h
            for l in range(n_layers):
                hks.append(find_k_proj(layers[l]).register_forward_hook(
                    make_quant_hook(l, bits, rot_dict, rand_rot)))

        depth_acc = {}
        for depth in depths:
            correct = 0
            for trial in range(n_trials):
                pk = str(np.random.randint(1000, 9999))
                needle_text = f"SECRET: The passkey is {pk}. Remember it."
                question_text = f"\nWhat is the passkey? Answer: {pk[0]}"
                needle_ids = tokenizer.encode(needle_text, add_special_tokens=False)
                q_ids = tokenizer.encode(question_text, add_special_tokens=False)
                filler_ids = tokenizer.encode(filler, add_special_tokens=False)

                avail = ctx_len - len(needle_ids) - len(q_ids) - 2
                if avail < 10:
                    continue
                ff = (filler_ids * (avail // len(filler_ids) + 1))[:avail]
                pos = int(len(ff) * depth)
                toks = ff[:pos] + needle_ids + ff[pos:] + q_ids

                input_ids = torch.tensor([toks[:ctx_len]], device=device)
                with torch.no_grad():
                    out = model(input_ids, use_cache=False)
                    pred_tok = tokenizer.decode(out.logits[0, -1, :].argmax().item()).strip()
                    if pred_tok == pk[1]:
                        correct += 1

            depth_acc[str(depth)] = correct / max(n_trials, 1)
            print(f"    [{name} {bits}b ctx={ctx_len}] depth={depth:.2f}: "
                  f"{correct}/{n_trials}={depth_acc[str(depth)]:.0%}")

        for hk in hks:
            hk.remove()
        return depth_acc

    results = {}
    for ctx_len in ctx_lengths:
        print(f"\n  Context length: {ctx_len}")
        np.random.seed(0)
        results[f'fp16_ctx{ctx_len}'] = test_method('fp16', 0, ctx_len=ctx_len)
        for bits in bits_list:
            np.random.seed(0)
            results[f'identity_{bits}b_ctx{ctx_len}'] = test_method('identity', bits, ctx_len=ctx_len)
            np.random.seed(0)
            results[f'turbo_{bits}b_ctx{ctx_len}'] = test_method('random', bits, rand_rot=R_random, ctx_len=ctx_len)
            np.random.seed(0)
            results[f'pre_pca_{bits}b_ctx{ctx_len}'] = test_method('pre_pca', bits, rot_dict=pca_pre, ctx_len=ctx_len)

    # Summary
    print(f"\n{'='*70}")
    print("NIAH Summary")
    print(f"{'='*70}")
    for key in sorted(results.keys()):
        accs = list(results[key].values())
        avg = np.mean(accs)
        print(f"  {key:35s}: avg={avg:.0%}  {[f'{a:.0%}' for a in accs]}")

    return results


# ============================================================================
# Self-Test: Verify Lloyd-Max fix
# ============================================================================
def self_test_lloydmax():
    """Quick self-test that the Lloyd-Max centering fix works."""
    print("\n  Self-test: Lloyd-Max centering fix...")
    np.random.seed(42)

    for mean_val in [0.0, 5.0, -3.0, 100.0]:
        for std_val in [1.0, 0.1, 10.0]:
            x = np.random.normal(mean_val, std_val, 10000)
            for bits in [2, 3, 4]:
                x_uni = uniform_quant(x, bits)
                x_lm = gaussian_lloyd_max(x, bits)
                mse_uni = np.mean((x - x_uni) ** 2)
                mse_lm = np.mean((x - x_lm) ** 2)

                if mse_lm > mse_uni * 1.1:  # Lloyd should be <= uniform for Gaussian data
                    print(f"    FAIL: mean={mean_val}, std={std_val}, bits={bits}: "
                          f"Lloyd MSE={mse_lm:.6f} > Uniform MSE={mse_uni:.6f}")
                    return False

    # Test with actual key-like data (non-zero mean)
    x_key = np.random.normal(2.5, 0.3, 5000)  # typical key channel
    for bits in [2, 3, 4]:
        mse_uni = np.mean((x_key - uniform_quant(x_key, bits)) ** 2)
        mse_lm = np.mean((x_key - gaussian_lloyd_max(x_key, bits)) ** 2)
        ratio = mse_uni / max(mse_lm, 1e-10)
        print(f"    bits={bits}, mean=2.5, std=0.3: Lloyd/Uniform ratio = {ratio:.2f}x "
              f"{'(Lloyd BETTER)' if ratio > 1 else '(Uniform BETTER)'}")

    print("  Self-test: PASS")
    return True


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified 3-Axis Verification")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g. Qwen/Qwen2.5-7B)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--axis", type=int, nargs="+", default=[1, 2, 3],
                        help="Which axes to verify (1, 2, 3)")
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--niah-contexts", type=int, nargs="+", default=[4096],
                        help="NIAH context lengths")
    parser.add_argument("--niah-trials", type=int, default=10)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test (fewer tokens, fewer heads)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    outdir = Path(args.output_dir) if args.output_dir else OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.replace("/", "_")

    print(f"{'#'*70}")
    print(f"# Unified 3-Axis Verification")
    print(f"# Model: {args.model}")
    print(f"# Axes: {args.axis}")
    print(f"# Bits: {args.bits}")
    print(f"# Smoke: {args.smoke}")
    print(f"{'#'*70}")

    # Self-test
    if not self_test_lloydmax():
        print("ABORTING: Lloyd-Max self-test failed")
        return

    # Load model and extract keys
    max_tokens = 512 if args.smoke else 2048
    layer_stride = 4 if args.smoke else 1
    data = load_model_and_extract_keys(
        args.model, args.device,
        max_tokens=max_tokens,
        layer_stride=layer_stride,
    )

    all_results = {'model': args.model, 'timestamp': timestamp, 'info': data['info']}

    # Axis 1: Rotation
    if 1 in args.axis:
        axis1 = verify_axis1_rotation(data, bits_list=args.bits)
        all_results['axis1_rotation'] = axis1

    # Axis 2: Lloyd-Max
    if 2 in args.axis:
        axis2 = verify_axis2_lloydmax(data, bits_list=args.bits)
        all_results['axis2_lloydmax'] = axis2

    import torch

    # If axis 3 is not requested, free model early
    if 3 not in args.axis and 'model' in data:
        del data['model']
        torch.cuda.empty_cache()
        gc.collect()

    # Axis 3: NIAH (needs model in GPU memory)
    if 3 in args.axis:
        niah_bits = [b for b in args.bits if b <= 3]  # NIAH most relevant at low bits
        niah_contexts = [2048] if args.smoke else args.niah_contexts
        niah_trials = 3 if args.smoke else args.niah_trials
        axis3 = verify_axis3_niah(data, ctx_lengths=niah_contexts,
                                  bits_list=niah_bits, n_trials=niah_trials)
        all_results['axis3_niah'] = axis3

    # Cleanup model
    del data['model']
    torch.cuda.empty_cache()
    gc.collect()

    # Save
    def make_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    outfile = outdir / f"3axis_{model_tag}_{timestamp}.json"
    with open(outfile, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"ALL DONE in {elapsed:.1f}s")
    print(f"Results saved: {outfile}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
