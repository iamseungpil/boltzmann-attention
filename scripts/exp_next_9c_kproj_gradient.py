#!/usr/bin/env python3
"""
Next-9c: Correct cascade factor via ∂loss/∂k_proj_output
=========================================================

FIX for Next-9/9b: The cascade factor g_{l,h} must capture the sensitivity
of PPL to KEY perturbation, not attention output perturbation.

Previous (Next-9/9b, wrong):
  g_{l,h} = ||∂loss/∂attn_out_{l,h}||²  -- upstream gradient only
  Missing: ∂(attn_out)/∂(K) = softmax Jacobian

Fix (Next-9c):
  g_{l,h} = ||∂loss/∂k_proj_output_{l,h}||²  -- direct sensitivity

This is equivalent to asking: "How much does the loss change if we
perturb the KEY at (layer l, kv head h)?". Exactly what Master Equation needs.

Also adds alternative: use Exp4's direct per-layer ΔPPL measurements
as the empirical sensitivity (no backward pass needed, more accurate).
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
DEVICE = 'cuda:0'
DTYPE = torch.bfloat16
N_CALIB_TOKENS = 1024
N_EVAL_TOKENS = 2048

AVG_BITS_TARGETS = [2.156, 2.3, 2.5]
B_FLOOR = 2
B_MAX = 6

# Exp4 Mistral per-layer ΔPPL (from exp4_per_layer_lloyd_breakdown.json)
# Direct measurement of Lloyd quantization impact per layer
EXP4_MISTRAL_DELTA_PPL = {
    0: 0.005, 1: 0.120, 2: 0.555, 3: 0.287, 4: 0.521, 5: 0.206,
    6: 0.304, 7: 0.166, 8: 0.152, 9: 0.160, 10: 0.070, 11: 0.079,
    12: 0.037, 13: 0.039, 14: 0.034, 15: 0.047, 16: 0.030, 17: 0.050,
    18: 0.025, 19: 0.024, 20: 0.067, 21: 0.067, 22: 0.155, 23: 0.122,
    24: 0.046, 25: 0.010, 26: -0.004, 27: 0.103, 28: 0.032, 29: 0.096,
    30: 0.116, 31: 0.028,
}

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


def measure_cascade_via_kproj(model, input_ids, n_layers, n_kv, n_q, head_dim):
    """
    CORRECT cascade measurement: gradient of loss wrt k_proj output.
    This captures full sensitivity of PPL to key perturbation at layer l.
    """
    grad_captures = {}

    def make_fh(li_local):
        def fh(mod, inputs, output):
            # output is k_proj output: (B, T, n_kv * head_dim)
            if output.requires_grad:
                def bh(grad):
                    grad_captures[li_local] = grad.detach().cpu().float().numpy()
                output.register_hook(bh)
        return fh

    handles = []
    for li in range(n_layers):
        k_proj = model.model.layers[li].self_attn.k_proj
        handles.append(k_proj.register_forward_hook(make_fh(li)))

    model.zero_grad()
    out = model(input_ids, use_cache=False)
    logits = out.logits
    targets = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        logits[:, :-1, :].contiguous().reshape(-1, logits.size(-1)).float(),
        targets.reshape(-1),
        reduction='mean'
    )
    loss.backward()

    for h in handles:
        h.remove()

    g_table = np.zeros((n_layers, n_kv), dtype=np.float32)
    for li, grad in grad_captures.items():
        T = grad.shape[1]
        grad_reshaped = grad.reshape(1, T, n_kv, head_dim)  # (1, T, n_kv, d)
        for hk in range(n_kv):
            # Sum squared over (T, d)
            g_val = np.mean(np.sum(grad_reshaped[0, :, hk, :] ** 2, axis=-1))
            g_table[li, hk] = float(g_val)

    return g_table


def collect_kqa(model, input_ids, n_layers):
    captured = {}
    handles = []
    def kh(li):
        def h(m, i, o):
            captured.setdefault(li, {})['k'] = o.detach().cpu().float().numpy()
        return h
    def ah(li):
        def h(m, i, o):
            if isinstance(o, tuple) and len(o) >= 2 and o[1] is not None:
                captured.setdefault(li, {})['attn'] = o[1].detach().cpu().float().numpy()
        return h
    def qh(li):
        def h(m, i, o):
            captured.setdefault(li, {})['q'] = o.detach().cpu().float().numpy()
        return h

    for li in range(n_layers):
        mod = model.model.layers[li].self_attn
        handles.append(mod.k_proj.register_forward_hook(kh(li)))
        handles.append(mod.q_proj.register_forward_hook(qh(li)))
        handles.append(mod.register_forward_hook(ah(li)))

    with torch.no_grad():
        _ = model(input_ids, output_attentions=True, use_cache=False)
    for h in handles:
        h.remove()
    return captured


def water_filling_global(importance, total_budget, b_floor=2, b_max=6):
    n = len(importance)
    imp = np.array(importance, dtype=np.float64)
    imp = np.maximum(imp, 1e-12)
    bits = np.full(n, b_floor, dtype=int)
    spent = n * b_floor
    if spent > total_budget:
        return bits
    while spent < total_budget:
        valid = bits < b_max
        if not valid.any():
            break
        gains = np.where(
            valid,
            imp * (4.0 ** (-bits.astype(float)) - 4.0 ** (-(bits + 1).astype(float))),
            -np.inf
        )
        j_best = int(np.argmax(gains))
        bits[j_best] += 1
        spent += 1
    return bits


def fit_pca_l2_lloyd(K, bits):
    K = K.astype(np.float32)
    K_mean = K.mean(axis=0)
    K_c = K - K_mean
    cov = (K_c.T @ K_c) / max(K.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    V = eigvecs[:, order]
    K_pca = K_c @ V
    d = K.shape[1]
    centroids = np.zeros((d, 2 ** bits), dtype=np.float32)
    for j in range(d):
        centroids[j] = lloyd_max_1d_fit(K_pca[:, j], bits, n_iter=20).astype(np.float32)
    return {
        'K_mean': K_mean,
        'V': V.astype(np.float32),
        'centroids': centroids,
        'bits': bits,
    }


class PCAL2LloydHook:
    def __init__(self, head_quantizers, n_kv, head_dim):
        self.hq = head_quantizers
        self.n_kv = n_kv
        self.head_dim = head_dim

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x_bf = output.view(B, T, self.n_kv, self.head_dim)
        x_np = x_bf.float().cpu().numpy()
        x_q = np.zeros_like(x_np)
        for hk in range(self.n_kv):
            q = self.hq[hk]
            data = x_np[:, :, hk, :]
            shape = data.shape
            K_flat = data.reshape(-1, self.head_dim).astype(np.float32)
            K_c = K_flat - q['K_mean']
            K_pca = K_c @ q['V']
            K_pca_q = np.zeros_like(K_pca)
            c = q['centroids']
            for j in range(self.head_dim):
                boundaries = (c[j, :-1] + c[j, 1:]) / 2
                idx = np.searchsorted(boundaries, K_pca[:, j])
                K_pca_q[:, j] = c[j, idx]
            K_recon = K_pca_q @ q['V'].T + q['K_mean']
            x_q[:, :, hk, :] = K_recon.reshape(shape)
        result = torch.from_numpy(x_q).to(output.device).to(output.dtype)
        return result.view(B, T, self.n_kv * self.head_dim)


def get_texts(tok):
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        texts = [t for t in ds['text'] if len(t.strip()) > 100]
        return '\n\n'.join(texts[:300]), '\n\n'.join(texts[300:600])
    except Exception:
        return " ".join(["Calib."] * 5000), " ".join(["Eval."] * 5000)


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


def make_config_bits(layer_weights, n_layers, n_kv, avg_bits, b_floor=2, b_max=6):
    """
    Given per-layer importance weights (layer_weights[l]),
    distribute bits per layer first (equally among heads of that layer).
    """
    total_budget = int(round(n_layers * n_kv * avg_bits))
    # Per-layer importance expanded to per-head (uniform within layer)
    importance_flat = []
    index_map = []
    for li in range(n_layers):
        for hk in range(n_kv):
            importance_flat.append(float(layer_weights[li]))
            index_map.append((li, hk))
    bits_flat = water_filling_global(
        np.array(importance_flat), total_budget, b_floor=b_floor, b_max=b_max
    )
    bits_table = np.zeros((n_layers, n_kv), dtype=int)
    for k, (li, hk) in enumerate(index_map):
        bits_table[li, hk] = bits_flat[k]
    return bits_table


def main():
    print("=" * 70)
    print("Next-9c: Cascade via ∂loss/∂k_proj + Exp4 direct sensitivity")
    print("=" * 70, flush=True)
    t_start = time.time()

    print("\nLoading model...", flush=True)
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
    head_dim = model.config.hidden_size // n_q
    print(f"  n_layers={n_layers}, n_kv={n_kv}, n_q={n_q}, head_dim={head_dim}", flush=True)

    calib_text, eval_text = get_texts(tok)
    calib_enc = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB_TOKENS)
    calib_ids = calib_enc['input_ids'].to(DEVICE)
    eval_enc = tok(eval_text, return_tensors='pt', truncation=True, max_length=N_EVAL_TOKENS)
    eval_ids = eval_enc['input_ids'].to(DEVICE)

    # Baseline
    print("\n[Baseline] FP16...", flush=True)
    ppl_fp16, _ = compute_ppl(model, eval_ids)
    print(f"  FP16 PPL: {ppl_fp16:.4f}", flush=True)

    # Measure CORRECT g via k_proj gradient
    print("\n[Measure g via ∂loss/∂k_proj]", flush=True)
    t0 = time.time()
    g_kproj_table = measure_cascade_via_kproj(model, calib_ids, n_layers, n_kv, n_q, head_dim)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)
    g_kproj_per_layer = g_kproj_table.sum(axis=1)
    top5_g = np.argsort(g_kproj_per_layer)[::-1][:5]
    print(f"  Top 5 layers by g (k_proj grad): {top5_g.tolist()}", flush=True)
    print(f"    values: {[f'{g_kproj_per_layer[l]:.2e}' for l in top5_g]}", flush=True)

    # Collect keys
    print("\n[Collect K for quantizer fit]", flush=True)
    captured = collect_kqa(model, calib_ids, n_layers)
    print(f"  Collected {len(captured)} layers", flush=True)

    K_table = {}
    for li in range(n_layers):
        data = captured.get(li, {})
        if 'k' not in data:
            continue
        T_c = data['k'].shape[1]
        K_all = data['k'].reshape(T_c, n_kv, head_dim).astype(np.float32)
        for hk in range(n_kv):
            K_table[(li, hk)] = K_all[:, hk, :]

    # Exp4 direct sensitivity as per-layer weights
    exp4_layer_weights = np.array(
        [max(0, EXP4_MISTRAL_DELTA_PPL.get(li, 0)) + 1e-6 for li in range(n_layers)],
        dtype=np.float32
    )
    print(f"\n[Exp4 direct sensitivities] Top 5 layers: "
          f"{np.argsort(exp4_layer_weights)[::-1][:5].tolist()}", flush=True)

    # g_kproj per-layer (sum over heads)
    g_kproj_layer_weights = g_kproj_per_layer.astype(np.float32)

    # Configs to test
    test_configs = []
    for avg in AVG_BITS_TARGETS:
        # Config 1: g_kproj-based
        test_configs.append(
            ('g_kproj_avg' + str(avg), g_kproj_layer_weights, avg)
        )
        # Config 2: Exp4 direct sensitivity
        test_configs.append(
            ('exp4_sensitivity_avg' + str(avg), exp4_layer_weights, avg)
        )

    results = {
        'model': MODEL_NAME,
        'ppl_fp16': ppl_fp16,
        'g_kproj_table': g_kproj_table.tolist(),
        'g_kproj_per_layer': g_kproj_per_layer.tolist(),
        'exp4_layer_weights': exp4_layer_weights.tolist(),
        'configs': {},
    }

    for cfg_name, layer_weights, avg_bits in test_configs:
        print(f"\n[Config {cfg_name}] avg_bits={avg_bits}", flush=True)

        bits_table = make_config_bits(
            layer_weights, n_layers, n_kv, avg_bits, B_FLOOR, B_MAX
        )
        actual_avg = bits_table.mean()
        unique_bits, counts = np.unique(bits_table, return_counts=True)
        dist = dict(zip(unique_bits.tolist(), counts.tolist()))
        print(f"  Bit distribution: {dist} (actual avg: {actual_avg:.3f})", flush=True)

        layer_bits_sum = bits_table.sum(axis=1)
        top5 = np.argsort(layer_bits_sum)[::-1][:5]
        print(f"  Top 5 layers by bits: {top5.tolist()} (total: {layer_bits_sum[top5].tolist()})", flush=True)

        # Fit quantizers
        t_fit = time.time()
        per_layer_head_data = {li: [None] * n_kv for li in range(n_layers)}
        for (li, hk), K in K_table.items():
            b = int(bits_table[li, hk])
            try:
                hd = fit_pca_l2_lloyd(K, b)
                per_layer_head_data[li][hk] = hd
            except Exception:
                pass
        print(f"  Fit in {time.time()-t_fit:.1f}s", flush=True)

        # Fill missing
        for li in range(n_layers):
            for hk in range(n_kv):
                if per_layer_head_data[li][hk] is None:
                    per_layer_head_data[li][hk] = {
                        'K_mean': np.zeros(head_dim, dtype=np.float32),
                        'V': np.eye(head_dim, dtype=np.float32),
                        'centroids': np.stack([
                            np.linspace(-3, 3, 2**B_FLOOR, dtype=np.float32)
                            for _ in range(head_dim)
                        ]),
                        'bits': B_FLOOR,
                    }

        # Measure PPL
        handles = []
        for li in range(n_layers):
            hook = PCAL2LloydHook(per_layer_head_data[li], n_kv, head_dim)
            h = model.model.layers[li].self_attn.k_proj.register_forward_hook(hook)
            handles.append(h)

        t0 = time.time()
        try:
            ppl, loss = compute_ppl(model, eval_ids)
        except Exception as e:
            ppl, loss = float('inf'), float('inf')
            print(f"  PPL FAILED: {e}", flush=True)
        finally:
            for h in handles:
                h.remove()

        delta = (ppl - ppl_fp16) / ppl_fp16 * 100
        print(f"  PPL: {ppl:.4f} (Δ vs FP16: {delta:+.2f}%) [eval: {time.time()-t0:.1f}s]", flush=True)

        results['configs'][cfg_name] = {
            'avg_bits_target': avg_bits,
            'avg_bits_actual': float(actual_avg),
            'ppl': ppl,
            'delta_vs_fp16_pct': delta,
            'bit_distribution': {int(k): int(v) for k, v in dist.items()},
            'top_5_layers_by_bits': top5.tolist(),
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  FP16: {ppl_fp16:.4f}")
    print(f"  Next-4 E (hand-picked L2-6 @ 3b, avg 2.156): 6.9505  ← main reference")
    print()

    for cfg_name, cfg_data in results['configs'].items():
        if 'ppl' in cfg_data:
            vs_E = (cfg_data['ppl'] - 6.9505) / 6.9505 * 100
            marker = "✅" if cfg_data['ppl'] < 6.9505 else "❌"
            print(f"  {marker} {cfg_name:<30} avg={cfg_data['avg_bits_actual']:.3f}  "
                  f"PPL={cfg_data['ppl']:.4f}  (vs Next-4 E: {vs_E:+.2f}%)")

    results['runtime_sec'] = time.time() - t_start
    out_file = OUT_DIR / 'exp_next9c_kproj_gradient.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")
    print(f"Total runtime: {results['runtime_sec']:.1f}s ({results['runtime_sec']/60:.1f}m)")


if __name__ == '__main__':
    main()
