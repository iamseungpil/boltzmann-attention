#!/usr/bin/env python3
"""
Next-4: Qwen outlier layer preservation (cross-model validation of Next-2)
===========================================================================

Qwen-7B has DIFFERENT catastrophic layer distribution vs Mistral:
  Mistral top-5:  2, 4, 6, 3, 5 (all early)
  Qwen top-5:     0, 26, 4, 5, 22 (bimodal: early + late)

Test if Qwen-specific outlier preservation gives similar improvement as Mistral.

Configs:
  (A) FP16 baseline
  (B) All layers @ 2-bit L² Lloyd
  (C) Qwen outlier {0, 26, 4, 5, 22} @ 3-bit, others @ 2-bit
  (D) Qwen outlier {0, 26, 4, 5, 22} @ 4-bit, others @ 2-bit
  (E) Layer 0 only @ 4-bit, others @ 2-bit
  (F) Qwen outlier {0, 26, 4, 5, 22} @ 8-bit, others @ 2-bit
  (G) WRONG outliers {2, 3, 4, 5, 6} @ 4-bit (Mistral's pattern, expected to fail)
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

MODEL_NAME = 'Qwen/Qwen2.5-7B'
SHORT = 'qwen-7b'
DEVICE = 'cuda:0'
DTYPE = torch.bfloat16
N_CALIB_TOKENS = 1024
N_EVAL_TOKENS = 2048

QWEN_OUTLIERS = [0, 26, 4, 5, 22]     # from Next-3
WRONG_OUTLIERS = [2, 3, 4, 5, 6]       # Mistral's pattern (for control)

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
    head_dim = model.config.hidden_size // model.config.num_attention_heads
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
        return " ".join(["Calibration."] * 5000), " ".join(["Evaluation."] * 5000)


def install_layer_hooks(model, per_layer_centroids, n_kv, head_dim):
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


def avg_bits(bits_list):
    return sum(bits_list) / len(bits_list)


def main():
    print("=" * 60)
    print(f"Next-4: Qwen Outlier Layer Preservation")
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
    print(f"  n_layers={n_layers}, n_kv={n_kv}, head_dim={head_dim}", flush=True)

    calib_text, eval_text = get_texts(tok)
    eval_enc = tok(eval_text, return_tensors='pt', truncation=True, max_length=N_EVAL_TOKENS)
    eval_ids = eval_enc['input_ids'].to(DEVICE)
    print(f"  Eval tokens: {eval_ids.shape[1]}", flush=True)

    # (A) Baseline
    print("\n[A] FP16 baseline...", flush=True)
    ppl_fp16, _ = compute_ppl(model, eval_ids)
    print(f"  FP16 PPL: {ppl_fp16:.4f}", flush=True)

    configs = {
        'B_all_2bit':              [2] * n_layers,
        'C_qwen_outlier_3b':       [3 if li in QWEN_OUTLIERS else 2 for li in range(n_layers)],
        'D_qwen_outlier_4b':       [4 if li in QWEN_OUTLIERS else 2 for li in range(n_layers)],
        'E_layer0_only_4b':        [4 if li == 0 else 2 for li in range(n_layers)],
        'F_qwen_outlier_8b':       [8 if li in QWEN_OUTLIERS else 2 for li in range(n_layers)],
        'G_wrong_outliers_4b':     [4 if li in WRONG_OUTLIERS else 2 for li in range(n_layers)],  # control
    }

    # Fit per (layer, bits) pair
    unique_pairs = set()
    for name, bl in configs.items():
        for li, b in enumerate(bl):
            unique_pairs.add((li, b))
    print(f"\n[Fit] {len(unique_pairs)} unique (layer, bits) pairs...", flush=True)

    centroid_cache = {}
    t_fit = time.time()
    for idx, (li, b) in enumerate(sorted(unique_pairs)):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  Fitting {idx+1}/{len(unique_pairs)}: L{li} @ {b}-bit", flush=True)
        centroid_cache[(li, b)] = fit_lloyd_for_layer_bits(model, tok, calib_text, li, b)
    print(f"  Fit done in {time.time()-t_fit:.1f}s", flush=True)

    # Run each config
    results_config = {}
    for name, bits_list in configs.items():
        avg_b = avg_bits(bits_list)
        outlier_layers = [li for li in range(n_layers) if bits_list[li] != 2]
        print(f"\n[{name}] avg={avg_b:.3f}, outlier_layers={outlier_layers}", flush=True)

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
    base_ppl = results_config['B_all_2bit']['ppl']
    for name, r in results_config.items():
        vs_base = (r['ppl'] - base_ppl) / base_ppl * 100
        print(f"{name:<25} | {r['avg_bits']:>10.3f} | {r['ppl']:>10.4f} | {r['delta_vs_fp16_pct']:>+11.2f}%  (vs 2-bit: {vs_base:+.2f}%)")

    out = {
        'model': MODEL_NAME,
        'qwen_outlier_layers': QWEN_OUTLIERS,
        'wrong_outliers': WRONG_OUTLIERS,
        'ppl_fp16': ppl_fp16,
        'configs': results_config,
        'runtime_sec': time.time() - t_start,
    }
    out_file = OUT_DIR / 'exp_next4_qwen_outlier_preservation.json'
    with open(out_file, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_file}")
    print(f"Total runtime: {out['runtime_sec']:.1f}s ({out['runtime_sec']/60:.1f}m)")


if __name__ == '__main__':
    main()
