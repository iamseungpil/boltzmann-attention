#!/usr/bin/env python3
"""
Next-2: Outlier layer bit preservation on Mistral-7B
=====================================================

From Exp1/Exp4 we found that Lloyd-Max failure is concentrated in early
layers (2-6, especially layer 2). Hypothesis: giving those outlier layers
MORE bits while keeping others at 2-bit should recover PPL cheaply.

Configs tested:
  (A) All-FP16 baseline
  (B) All layers @ 2-bit L² Lloyd
  (C) Outlier layers {2,3,4,5,6} @ 3-bit, others @ 2-bit
  (D) Outlier layers {2,3,4,5,6} @ 4-bit, others @ 2-bit
  (E) Outlier layer {2} only @ 4-bit, others @ 2-bit
  (F) Outlier layers {2,3,4,5,6} @ 8-bit, others @ 2-bit (upper bound)

Effective bit rate for (C-F):
  n_outlier=5, n_other=27
  (C) 5*3 + 27*2 = 69 / 32 = 2.16 avg bits
  (D) 5*4 + 27*2 = 74 / 32 = 2.31 avg bits
  (E) 1*4 + 31*2 = 66 / 32 = 2.06 avg bits
  (F) 5*8 + 27*2 = 94 / 32 = 2.94 avg bits
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

MODEL_NAME = 'mistralai/Mistral-7B-v0.3'
SHORT = 'mistral-7b'
DEVICE = 'cuda:0'
DTYPE = torch.bfloat16
N_CALIB_TOKENS = 1024
N_EVAL_TOKENS = 2048

# Outlier layers from Exp1+Exp4
OUTLIER_LAYERS = [2, 3, 4, 5, 6]

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
    """Lloyd quantizer with configurable bits per layer."""
    def __init__(self, centroids_per_head, n_kv, head_dim):
        # centroids_per_head: list of n_kv arrays shape (head_dim, n_levels)
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
    """Fit Lloyd centroids for a given layer at a given bit count."""
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
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    K = K.reshape(1, -1, n_kv, head_dim)[0]  # (T, n_kv, d)

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
        calib = '\n\n'.join(texts[:300])
        eval_ = '\n\n'.join(texts[300:600])
    except Exception:
        calib = " ".join(["Calibration."] * 5000)
        eval_ = " ".join(["Evaluation."] * 5000)
    return calib, eval_


def install_layer_hooks(model, per_layer_centroids, n_kv, head_dim):
    """Install Lloyd hooks with per-layer centroid assignment (None = skip)."""
    handles = []
    for li, centroids in enumerate(per_layer_centroids):
        if centroids is None:
            continue
        hook = MixedBitLloydHook(centroids, n_kv, head_dim)
        h = model.model.layers[li].self_attn.k_proj.register_forward_hook(hook)
        handles.append(h)
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def avg_bits(per_layer_bits):
    return sum(per_layer_bits) / len(per_layer_bits)


def main():
    print("=" * 60)
    print("Exp Next-2: Outlier Layer Bit Preservation")
    print("=" * 60, flush=True)
    t_start = time.time()

    print("Loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device_map=DEVICE,
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t_start:.1f}s", flush=True)

    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    calib_text, eval_text = get_texts(tok)
    eval_enc = tok(eval_text, return_tensors='pt', truncation=True, max_length=N_EVAL_TOKENS)
    eval_ids = eval_enc['input_ids'].to(DEVICE)
    print(f"  Eval tokens: {eval_ids.shape[1]}", flush=True)

    # (A) Baseline FP16
    print("\n[A] Baseline FP16...", flush=True)
    ppl_fp16, _ = compute_ppl(model, eval_ids)
    print(f"  FP16 PPL: {ppl_fp16:.4f}", flush=True)

    # Define configs
    configs = {
        'B_all_2bit':      [2] * n_layers,
        'C_outlier_3b':    [3 if li in OUTLIER_LAYERS else 2 for li in range(n_layers)],
        'D_outlier_4b':    [4 if li in OUTLIER_LAYERS else 2 for li in range(n_layers)],
        'E_layer2_only_4b': [4 if li == 2 else 2 for li in range(n_layers)],
        'F_outlier_8b':    [8 if li in OUTLIER_LAYERS else 2 for li in range(n_layers)],
    }

    # Fit centroids for all unique (layer, bits) pairs
    print(f"\n[Fit] Fitting Lloyd centroids per unique (layer, bits) combination...", flush=True)
    # Determine unique (layer, bits) pairs across all configs
    unique_pairs = set()
    for name, bits_list in configs.items():
        for li, b in enumerate(bits_list):
            unique_pairs.add((li, b))
    print(f"  Need to fit {len(unique_pairs)} (layer, bits) combinations", flush=True)

    # Cache: (layer, bits) -> centroids_per_head
    centroid_cache = {}
    t_fit = time.time()
    for idx, (li, b) in enumerate(sorted(unique_pairs)):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  Fitting {idx+1}/{len(unique_pairs)}: layer {li} @ {b}-bit", flush=True)
        centroid_cache[(li, b)] = fit_lloyd_for_layer_bits(model, tok, calib_text, li, b)
    print(f"  Fit done in {time.time()-t_fit:.1f}s", flush=True)

    # Run each config
    results_config = {}
    for name, bits_list in configs.items():
        avg_b = avg_bits(bits_list)
        print(f"\n[{name}] avg_bits={avg_b:.3f}, outlier_layers={[li for li in range(n_layers) if bits_list[li]!=2]}", flush=True)

        # Build per-layer centroid list
        per_layer = [centroid_cache[(li, bits_list[li])] for li in range(n_layers)]
        handles = install_layer_hooks(model, per_layer, n_kv, head_dim)
        t0 = time.time()
        ppl, loss = compute_ppl(model, eval_ids)
        elapsed = time.time() - t0
        remove_hooks(handles)

        print(f"  PPL: {ppl:.4f} ({elapsed:.1f}s)", flush=True)
        results_config[name] = {
            'avg_bits': avg_b,
            'ppl': ppl,
            'loss': loss,
            'delta_vs_fp16_pct': (ppl - ppl_fp16) / ppl_fp16 * 100,
            'bits_per_layer': bits_list,
        }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  FP16 baseline: {ppl_fp16:.4f}")
    print()
    print(f"{'Config':<25} | {'avg_bits':>10} | {'PPL':>10} | {'Δ vs FP16':>12}")
    print('-' * 70)
    for name, r in results_config.items():
        print(f"{name:<25} | {r['avg_bits']:>10.3f} | {r['ppl']:>10.4f} | {r['delta_vs_fp16_pct']:>+11.2f}%")

    # Save
    out = {
        'model': MODEL_NAME,
        'outlier_layers': OUTLIER_LAYERS,
        'ppl_fp16': ppl_fp16,
        'configs': results_config,
        'runtime_sec': time.time() - t_start,
    }
    out_file = OUT_DIR / 'exp_next2_outlier_layer_preservation.json'
    with open(out_file, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_file}")
    print(f"Total runtime: {out['runtime_sec']:.1f}s ({out['runtime_sec']/60:.1f}m)")


if __name__ == '__main__':
    main()
