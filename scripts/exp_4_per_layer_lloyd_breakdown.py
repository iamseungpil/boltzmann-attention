#!/usr/bin/env python3
"""
Experiment 4: Per-layer L² Lloyd failure breakdown
====================================================

For each layer of Mistral-7B (and optionally Qwen-7B):
  1. Baseline: full FP16 PPL on held-out text
  2. For each layer li in the model:
     - Apply L² Lloyd 2-bit only to that layer's K (via forward hook)
     - Measure PPL on the same held-out text
     - ΔPPL_li = PPL_quantized_at_li - baseline_PPL
  3. Identify "catastrophic layers" (large ΔPPL)

This tests the hypothesis from Experiment 1:
  Lloyd-Max failure is driven by a few outlier layers/heads, concentrated
  in specific locations (e.g., layer 2 observed in E1).

Output: per-layer ΔPPL breakdown for Mistral (main target for Lloyd catastrophe)
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

MODELS_TO_TEST = [
    ('mistralai/Mistral-7B-v0.3', 'mistral-7b'),
    # Qwen as optional comparison (smaller failure in v3)
]

DEVICE = 'cuda:0'
DTYPE = torch.bfloat16
N_CALIB_TOKENS = 1024    # calibration for Lloyd centroids
N_EVAL_TOKENS = 2048     # held-out PPL eval
BITS = 2

OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def lloyd_max_1d_fit(col, bits, n_iter=30):
    """Fit Lloyd-Max centroids to 1D data, return centroids array."""
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


def apply_lloyd_to_tensor(x, centroids_per_dim):
    """Apply per-dim Lloyd-Max quantization to tensor x of shape (..., d).
    centroids_per_dim: list of d arrays of centroids."""
    x_np = x.float().cpu().numpy()
    orig_shape = x_np.shape
    d = orig_shape[-1]
    x_flat = x_np.reshape(-1, d)
    x_q = np.zeros_like(x_flat)
    for j in range(d):
        centroids = centroids_per_dim[j]
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        idx = np.searchsorted(boundaries, x_flat[:, j])
        x_q[:, j] = centroids[idx]
    return torch.from_numpy(x_q.reshape(orig_shape)).to(x.device).to(x.dtype)


class LloydQuantizerHook:
    """Forward hook on k_proj to quantize output with pre-fitted Lloyd centroids."""
    def __init__(self, centroids_per_head, n_kv, head_dim):
        # centroids_per_head: list of n_kv arrays of shape (d, n_levels)
        self.centroids_per_head = centroids_per_head
        self.n_kv = n_kv
        self.head_dim = head_dim

    def __call__(self, module, inputs, output):
        # output: (B, T, n_kv * head_dim)
        B, T, _ = output.shape
        x = output.view(B, T, self.n_kv, self.head_dim)
        x_q = torch.zeros_like(x)
        for hk in range(self.n_kv):
            head_data = x[:, :, hk, :]  # (B, T, d)
            centroids = self.centroids_per_head[hk]  # (d, n_levels) np array
            x_q[:, :, hk, :] = apply_lloyd_to_tensor(head_data, centroids)
        return x_q.view(B, T, self.n_kv * self.head_dim)


def compute_ppl(model, input_ids):
    """Compute perplexity over input_ids (cross-entropy on all tokens)."""
    with torch.no_grad():
        out = model(input_ids, use_cache=False)
        logits = out.logits[:, :-1, :].contiguous()  # (B, T-1, V)
        targets = input_ids[:, 1:].contiguous()  # (B, T-1)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            reduction='mean'
        )
        ppl = torch.exp(loss).item()
    return ppl, loss.item()


def get_calib_and_eval_text(tok):
    """Get calibration and held-out evaluation text from WikiText-2."""
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        texts = [t for t in ds['text'] if len(t.strip()) > 100]
        calib_text = '\n\n'.join(texts[:300])
        eval_text = '\n\n'.join(texts[300:600])
    except Exception:
        calib_text = " ".join(["Calibration text for Lloyd quantizer."] * 5000)
        eval_text = " ".join(["Evaluation text for perplexity measurement."] * 5000)
    return calib_text, eval_text


def fit_lloyd_for_layer(model, tok, layer_idx, calib_text, bits=BITS):
    """
    Collect pre-RoPE keys from a single layer via hook on k_proj,
    then fit Lloyd-Max centroids per (head, dim).
    Returns: centroids list of shape (n_kv,) each (head_dim, n_levels).
    """
    enc = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB_TOKENS)
    input_ids = enc['input_ids'].to(DEVICE)

    captured = {}
    def hook(module, inputs, output):
        captured['k'] = output.detach().cpu().float().numpy()

    attn = model.model.layers[layer_idx].self_attn
    h = attn.k_proj.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(input_ids, use_cache=False)
    h.remove()

    K = captured['k']   # (1, T, n_kv * d)
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_q
    K = K.reshape(1, -1, n_kv, head_dim)[0]   # (T, n_kv, d)
    T = K.shape[0]

    centroids_per_head = []
    for hk in range(n_kv):
        head_K = K[:, hk, :]   # (T, d)
        centroids = np.zeros((head_dim, 2**bits))
        for j in range(head_dim):
            centroids[j] = lloyd_max_1d_fit(head_K[:, j], bits, n_iter=30)
        centroids_per_head.append(centroids)
    return centroids_per_head, (n_kv, head_dim)


def run_model(model_name, short_name):
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}", flush=True)
    t_start = time.time()

    print("Loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=DTYPE, device_map=DEVICE,
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t_start:.1f}s", flush=True)

    calib_text, eval_text = get_calib_and_eval_text(tok)
    eval_enc = tok(eval_text, return_tensors='pt', truncation=True, max_length=N_EVAL_TOKENS)
    eval_ids = eval_enc['input_ids'].to(DEVICE)
    print(f"  Calib tokens: {N_CALIB_TOKENS}, Eval tokens: {eval_ids.shape[1]}", flush=True)

    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    # Baseline PPL
    baseline_ppl, baseline_loss = compute_ppl(model, eval_ids)
    print(f"  Baseline PPL: {baseline_ppl:.4f}", flush=True)

    # Per-layer substitution
    print(f"\n  Testing {n_layers} layers for Lloyd-Max failure:", flush=True)
    per_layer_results = []

    for li in range(n_layers):
        t_li = time.time()
        # Fit Lloyd centroids for this layer
        centroids, _ = fit_lloyd_for_layer(model, tok, li, calib_text, bits=BITS)

        # Register hook
        hook = LloydQuantizerHook(centroids, n_kv, head_dim)
        handle = model.model.layers[li].self_attn.k_proj.register_forward_hook(hook)

        # Measure PPL
        ppl_li, loss_li = compute_ppl(model, eval_ids)
        handle.remove()

        delta_ppl = ppl_li - baseline_ppl
        ratio = ppl_li / baseline_ppl
        elapsed = time.time() - t_li
        marker = '!!!' if ratio > 1.5 else '   '
        print(f"  {marker} Layer {li:3d}: PPL={ppl_li:8.3f}  ΔPPL={delta_ppl:+7.3f}  ratio={ratio:5.3f}  "
              f"({elapsed:.1f}s)", flush=True)

        per_layer_results.append({
            'layer': li,
            'ppl': ppl_li,
            'loss': loss_li,
            'delta_ppl': delta_ppl,
            'ratio': ratio,
        })

    # Sort and highlight outlier layers
    sorted_by_delta = sorted(per_layer_results, key=lambda x: -x['delta_ppl'])

    print(f"\n  Top-5 catastrophic layers:")
    for r in sorted_by_delta[:5]:
        print(f"    Layer {r['layer']:3d}: ΔPPL = {r['delta_ppl']:+.3f}, ratio = {r['ratio']:.3f}")
    print(f"\n  Bottom-5 (Lloyd is safe):")
    for r in sorted_by_delta[-5:]:
        print(f"    Layer {r['layer']:3d}: ΔPPL = {r['delta_ppl']:+.3f}, ratio = {r['ratio']:.3f}")

    # Stats
    deltas = [r['delta_ppl'] for r in per_layer_results]
    stats = {
        'baseline_ppl': baseline_ppl,
        'mean_delta': float(np.mean(deltas)),
        'median_delta': float(np.median(deltas)),
        'max_delta': float(np.max(deltas)),
        'n_catastrophic_1.5x': sum(1 for r in per_layer_results if r['ratio'] > 1.5),
        'n_catastrophic_2x': sum(1 for r in per_layer_results if r['ratio'] > 2.0),
        'n_catastrophic_10x': sum(1 for r in per_layer_results if r['ratio'] > 10.0),
    }

    result = {
        'model': model_name,
        'short_name': short_name,
        'n_layers': n_layers,
        'n_kv': n_kv,
        'head_dim': head_dim,
        'bits': BITS,
        'baseline_ppl': baseline_ppl,
        'per_layer': per_layer_results,
        'sorted_by_delta': sorted_by_delta,
        'stats': stats,
        'runtime_sec': time.time() - t_start,
    }

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return result


def main():
    print("=== Experiment 4: Per-layer L² Lloyd failure breakdown ===")
    t_start = time.time()
    all_results = {}

    for model_name, short_name in MODELS_TO_TEST:
        try:
            r = run_model(model_name, short_name)
            all_results[short_name] = r
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"!!! FAILED {model_name}: {e}")
            all_results[short_name] = {'error': str(e)}

    all_results['_meta'] = {
        'total_runtime_sec': time.time() - t_start,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    out_file = OUT_DIR / 'exp4_per_layer_lloyd_breakdown.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n=== Saved: {out_file} ===")
    print(f"Total runtime: {(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    main()
