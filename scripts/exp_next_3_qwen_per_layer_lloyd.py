#!/usr/bin/env python3
"""
Next-3: Per-layer L² Lloyd failure breakdown on Qwen2.5-7B
===========================================================

Cross-model validation of Exp4 (which ran on Mistral-7B).

Qwen has a MUCH smaller Lloyd failure in v3 (1.05× vs Mistral's 5.06×).
Does Qwen also have a "layer 2 catastrophe" pattern, just at smaller scale?
Or is the failure distributed uniformly?

Prediction based on Exp1:
  Qwen has only 1 outlier head (vs Mistral's 5). So we expect:
  - Smaller max ΔPPL than Mistral
  - Fewer catastrophic layers
  - Possibly different outlier layer location (Exp1 top was layer 19 for Qwen-7B)
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
SHORT = 'qwen2.5-7b'
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


def apply_lloyd_to_tensor(x, centroids_per_dim):
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
    def __init__(self, centroids_per_head, n_kv, head_dim):
        self.centroids_per_head = centroids_per_head
        self.n_kv = n_kv
        self.head_dim = head_dim

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x = output.view(B, T, self.n_kv, self.head_dim)
        x_q = torch.zeros_like(x)
        for hk in range(self.n_kv):
            head_data = x[:, :, hk, :]
            centroids = self.centroids_per_head[hk]
            x_q[:, :, hk, :] = apply_lloyd_to_tensor(head_data, centroids)
        return x_q.view(B, T, self.n_kv * self.head_dim)


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


def fit_lloyd_for_layer(model, tok, layer_idx, calib_text, bits=BITS):
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
    n_q = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_q
    K = K.reshape(1, -1, n_kv, head_dim)[0]

    centroids_per_head = []
    for hk in range(n_kv):
        head_K = K[:, hk, :].astype(np.float32)
        centroids = np.zeros((head_dim, 2 ** bits))
        for j in range(head_dim):
            centroids[j] = lloyd_max_1d_fit(head_K[:, j], bits, n_iter=30)
        centroids_per_head.append(centroids)
    return centroids_per_head


def main():
    print("=" * 60)
    print(f"Exp Next-3: Per-layer Lloyd breakdown on {MODEL_NAME}")
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

    calib_text, eval_text = get_texts(tok)
    eval_enc = tok(eval_text, return_tensors='pt', truncation=True, max_length=N_EVAL_TOKENS)
    eval_ids = eval_enc['input_ids'].to(DEVICE)
    print(f"  Eval tokens: {eval_ids.shape[1]}", flush=True)

    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    print(f"  n_layers={n_layers}, n_kv={n_kv}, head_dim={head_dim}", flush=True)

    # Baseline
    baseline_ppl, _ = compute_ppl(model, eval_ids)
    print(f"  Baseline PPL: {baseline_ppl:.4f}", flush=True)

    print(f"\n  Per-layer Lloyd substitution (32 layers):", flush=True)
    per_layer = []
    for li in range(n_layers):
        t_li = time.time()
        centroids = fit_lloyd_for_layer(model, tok, li, calib_text, bits=BITS)
        hook = LloydQuantizerHook(centroids, n_kv, head_dim)
        handle = model.model.layers[li].self_attn.k_proj.register_forward_hook(hook)
        ppl_li, loss_li = compute_ppl(model, eval_ids)
        handle.remove()

        delta = ppl_li - baseline_ppl
        ratio = ppl_li / baseline_ppl
        marker = '!!!' if ratio > 1.1 else '   '
        print(f"  {marker} Layer {li:3d}: PPL={ppl_li:8.3f}  ΔPPL={delta:+7.3f}  ratio={ratio:5.3f} ({time.time()-t_li:.1f}s)", flush=True)
        per_layer.append({
            'layer': li,
            'ppl': ppl_li,
            'loss': loss_li,
            'delta_ppl': delta,
            'ratio': ratio,
        })

    sorted_delta = sorted(per_layer, key=lambda x: -x['delta_ppl'])
    print(f"\n  Top-5 catastrophic layers (Qwen):")
    for r in sorted_delta[:5]:
        print(f"    Layer {r['layer']:3d}: ΔPPL = {r['delta_ppl']:+.3f}, ratio = {r['ratio']:.3f}")

    print(f"\n  Comparison with Mistral (Exp4):")
    print(f"    Mistral top catastrophic: layer 2 (+0.555), 4 (+0.521), 6 (+0.304)")
    print(f"    Qwen top catastrophic: "
          f"layer {sorted_delta[0]['layer']} (+{sorted_delta[0]['delta_ppl']:.3f})")

    out = {
        'model': MODEL_NAME,
        'n_layers': n_layers,
        'n_kv': n_kv,
        'head_dim': head_dim,
        'bits': BITS,
        'baseline_ppl': baseline_ppl,
        'per_layer': per_layer,
        'sorted_by_delta': sorted_delta,
        'max_delta': float(max(r['delta_ppl'] for r in per_layer)),
        'max_ratio': float(max(r['ratio'] for r in per_layer)),
        'n_catastrophic_10pct': sum(1 for r in per_layer if r['ratio'] > 1.1),
        'runtime_sec': time.time() - t_start,
    }
    out_file = OUT_DIR / 'exp_next3_qwen_per_layer_lloyd.json'
    with open(out_file, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_file}")
    print(f"Total runtime: {out['runtime_sec']:.1f}s ({out['runtime_sec']/60:.1f}m)")


if __name__ == '__main__':
    main()
