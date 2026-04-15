#!/usr/bin/env python3
"""
Next-8: Per-head sensitivity on Mistral-7B
=============================================

Previous experiments were per-layer. This extends to per-(layer, kv_head)
granularity to find the most sensitive individual heads.

For each (layer, kv_head):
  1. Fit Lloyd centroids at 2-bit for this specific head
  2. Apply quantization ONLY to this head (others FP16)
  3. Measure PPL delta
  4. Rank all (layer, kv_head) by ΔPPL

Also test "per-head sensitivity allocation":
  - Give top-N most sensitive HEADS extra bits (rest at 2-bit)
  - Compare vs per-layer allocation with same avg bits

For Mistral-7B: 32 layers × 8 kv_heads = 256 heads.
Expected runtime: ~256 × 2 sec per PPL eval = ~9 min + fitting + final configs
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
BITS = 2

OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')


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


class SingleHeadLloydHook:
    """Quantizes ONLY a specific kv_head at the target layer, leaves others FP16."""
    def __init__(self, target_kv_head, centroids, n_kv, head_dim):
        self.target = target_kv_head
        self.centroids = centroids  # (head_dim, n_levels)
        self.n_kv = n_kv
        self.head_dim = head_dim

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x = output.view(B, T, self.n_kv, self.head_dim)
        # Extract only target head
        data = x[:, :, self.target, :].float().cpu().numpy()
        shape = data.shape
        data_flat = data.reshape(-1, self.head_dim)
        c = self.centroids
        for j in range(self.head_dim):
            boundaries = (c[j, :-1] + c[j, 1:]) / 2
            idx = np.searchsorted(boundaries, data_flat[:, j])
            data_flat[:, j] = c[j, idx]
        data_q = torch.from_numpy(data_flat.reshape(shape)).to(output.device).to(output.dtype)
        # Replace only target head
        x_new = x.clone()
        x_new[:, :, self.target, :] = data_q
        return x_new.view(B, T, self.n_kv * self.head_dim)


class MultiHeadMixedHook:
    """Per-head bit allocation: each kv_head can have its own centroids.
    If centroids is None, leave FP16."""
    def __init__(self, centroids_per_head, n_kv, head_dim):
        # centroids_per_head: list of n_kv, each either None (FP16) or (head_dim, n_levels)
        self.centroids = centroids_per_head
        self.n_kv = n_kv
        self.head_dim = head_dim

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x = output.view(B, T, self.n_kv, self.head_dim)
        x_out = x.clone()
        for hk in range(self.n_kv):
            if self.centroids[hk] is None:
                continue  # keep FP16
            data = x[:, :, hk, :].float().cpu().numpy()
            shape = data.shape
            data_flat = data.reshape(-1, self.head_dim)
            c = self.centroids[hk]
            for j in range(self.head_dim):
                boundaries = (c[j, :-1] + c[j, 1:]) / 2
                idx = np.searchsorted(boundaries, data_flat[:, j])
                data_flat[:, j] = c[j, idx]
            data_q = torch.from_numpy(data_flat.reshape(shape)).to(output.device).to(output.dtype)
            x_out[:, :, hk, :] = data_q
        return x_out.view(B, T, self.n_kv * self.head_dim)


def fit_per_head(model, tok, calib_text, layer_idx, bits):
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
        return " ".join(["Calib."] * 5000), " ".join(["Eval."] * 5000)


def main():
    print("=" * 60)
    print(f"Next-8: Per-head sensitivity on {MODEL_NAME}")
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
    print(f"  Total heads to test: {n_layers * n_kv}", flush=True)

    calib_text, eval_text = get_texts(tok)
    eval_enc = tok(eval_text, return_tensors='pt', truncation=True, max_length=N_EVAL_TOKENS)
    eval_ids = eval_enc['input_ids'].to(DEVICE)

    ppl_fp16, _ = compute_ppl(model, eval_ids)
    print(f"  FP16 PPL: {ppl_fp16:.4f}", flush=True)

    # ======================================
    # Phase 1: Fit per-layer Lloyd centroids at 2-bit
    # ======================================
    print(f"\n{'='*40}")
    print("Phase 1: Fit Lloyd centroids per layer")
    print('='*40, flush=True)

    per_layer_centroids = {}
    t_fit = time.time()
    for li in range(n_layers):
        per_layer_centroids[li] = fit_per_head(model, tok, calib_text, li, BITS)
        if (li + 1) % 8 == 0:
            print(f"  Fit progress: {li+1}/{n_layers}", flush=True)
    print(f"  Fit done in {time.time()-t_fit:.1f}s", flush=True)

    # ======================================
    # Phase 2: Per-head PPL substitution
    # ======================================
    print(f"\n{'='*40}")
    print("Phase 2: Per-head PPL substitution")
    print('='*40, flush=True)

    per_head = []
    t_phase2 = time.time()
    total = n_layers * n_kv
    done = 0
    for li in range(n_layers):
        for hk in range(n_kv):
            done += 1
            hook = SingleHeadLloydHook(hk, per_layer_centroids[li][hk], n_kv, head_dim)
            handle = model.model.layers[li].self_attn.k_proj.register_forward_hook(hook)
            ppl, loss = compute_ppl(model, eval_ids)
            handle.remove()

            delta = ppl - ppl_fp16
            per_head.append({
                'layer': li, 'kv_head': hk,
                'ppl': ppl, 'delta_ppl': delta, 'ratio': ppl / ppl_fp16,
            })

            if done % 16 == 0 or delta > 0.1:
                marker = '!!!' if delta > 0.1 else '   '
                print(f"  {marker} L{li:3d}/H{hk} ({done}/{total}): Δ={delta:+.4f}", flush=True)

    print(f"  Phase 2 done in {time.time()-t_phase2:.1f}s", flush=True)

    # Sort
    sorted_heads = sorted(per_head, key=lambda x: -x['delta_ppl'])
    print(f"\n  Top-10 most sensitive (layer, kv_head) pairs:")
    for r in sorted_heads[:10]:
        print(f"    L{r['layer']:3d}/H{r['kv_head']}: Δ={r['delta_ppl']:+.4f}")

    # ======================================
    # Phase 3: Per-head sensitivity allocation
    # ======================================
    print(f"\n{'='*40}")
    print("Phase 3: Per-head vs per-layer allocation")
    print('='*40, flush=True)

    # Fit additional centroids at 3-bit for top heads
    top_heads = [(r['layer'], r['kv_head']) for r in sorted_heads[:30]]  # top-30
    top_layers = sorted(set(li for li, _ in top_heads))

    # Fit 3-bit centroids for layers that contain top heads
    print(f"  Fitting 3-bit centroids for {len(top_layers)} layers with top heads...", flush=True)
    per_layer_3bit = {}
    for li in top_layers:
        per_layer_3bit[li] = fit_per_head(model, tok, calib_text, li, 3)

    # Configs:
    def make_per_head_alloc(top_k):
        """Return per-layer list of (centroids_per_head_dict_or_None)."""
        target_heads_3bit = set((r['layer'], r['kv_head']) for r in sorted_heads[:top_k])
        per_layer = []
        for li in range(n_layers):
            layer_centroids = []
            for hk in range(n_kv):
                if (li, hk) in target_heads_3bit:
                    # Use 3-bit centroids
                    if li in per_layer_3bit:
                        layer_centroids.append(per_layer_3bit[li][hk])
                    else:
                        layer_centroids.append(per_layer_centroids[li][hk])  # fallback
                else:
                    layer_centroids.append(per_layer_centroids[li][hk])  # 2-bit
            per_layer.append(layer_centroids)
        return per_layer

    def compute_per_head_avg_bits(top_k):
        total_bits = (n_layers * n_kv - top_k) * 2 + top_k * 3
        return total_bits / (n_layers * n_kv)

    configs = {
        'per_head_top5_3b':   make_per_head_alloc(5),
        'per_head_top10_3b':  make_per_head_alloc(10),
        'per_head_top20_3b':  make_per_head_alloc(20),
        'per_head_top30_3b':  make_per_head_alloc(30),
    }

    results = {}
    for name, per_layer_cents in configs.items():
        handles = []
        for li in range(n_layers):
            hook = MultiHeadMixedHook(per_layer_cents[li], n_kv, head_dim)
            h = model.model.layers[li].self_attn.k_proj.register_forward_hook(hook)
            handles.append(h)
        t0 = time.time()
        ppl, loss = compute_ppl(model, eval_ids)
        elapsed = time.time() - t0
        for h in handles:
            h.remove()

        # Compute avg bits
        top_k_num = int(name.replace('per_head_top', '').replace('_3b', ''))
        avg_b = compute_per_head_avg_bits(top_k_num)

        results[name] = {
            'top_k_heads': top_k_num,
            'avg_bits': avg_b,
            'ppl': ppl,
            'delta_vs_fp16_pct': (ppl - ppl_fp16) / ppl_fp16 * 100,
        }
        print(f"  {name:<25} topK={top_k_num} avg={avg_b:.3f} PPL={ppl:.4f} Δ={((ppl-ppl_fp16)/ppl_fp16*100):+.2f}% ({elapsed:.1f}s)", flush=True)

    # Compare with per-layer allocation at similar budgets
    # per_head_top30 (30 heads * 1 extra bit / 256 heads = 0.117 extra) avg 2.117
    # vs per_layer top15 (15 layers * 1 extra bit / 32 layers = 0.469) avg 2.469
    # Different scales, but we can compare
    print(f"\n  Comparison: per-head vs per-layer allocation")
    print(f"  (at similar avg bit budgets)")
    per_layer_reference = {
        'top5_3b': {'avg_bits': 2.156, 'ppl': 10.36},   # from Next-5
        'top10_3b': {'avg_bits': 2.312, 'ppl': 8.24},
        'top15_3b': {'avg_bits': 2.469, 'ppl': 7.58},
    }
    print(f"  per-layer reference (from Next-5):")
    for name, r in per_layer_reference.items():
        print(f"    {name}: avg={r['avg_bits']:.3f} PPL={r['ppl']:.2f}")

    out = {
        'model': MODEL_NAME,
        'n_layers': n_layers,
        'n_kv': n_kv,
        'head_dim': head_dim,
        'ppl_fp16': ppl_fp16,
        'per_head_breakdown': per_head,
        'sorted_by_delta': sorted_heads,
        'top_30_heads': [{'layer': r['layer'], 'kv_head': r['kv_head'], 'delta_ppl': r['delta_ppl']} for r in sorted_heads[:30]],
        'per_head_allocation_results': results,
        'per_layer_reference_from_next5': per_layer_reference,
        'runtime_sec': time.time() - t_start,
    }
    out_file = OUT_DIR / 'exp_next8_per_head_sensitivity.json'
    with open(out_file, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_file}")
    print(f"Total runtime: {out['runtime_sec']:.1f}s ({out['runtime_sec']/60:.1f}m)")


if __name__ == '__main__':
    main()
