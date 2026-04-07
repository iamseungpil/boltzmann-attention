#!/usr/bin/env python3
"""
Next-10: CWF extended — Qwen cross-verification + avg_bits sweep
=================================================================

Two objectives:
  1. Qwen-7B cross-verification of CWF method
     - Qwen has smaller Lloyd failure (v3 ratio 1.05× vs Mistral 5.06×)
     - Does CWF still work? Or is it Mistral-specific?
  2. Extended avg_bits sweep on Mistral to probe v3 WF floor=2 boundary
     - v3 best: WF floor=2 = 5.82 at 2-bit
     - Can CWF at higher avg_bits match or beat 5.82?

Method: Pre-RoPE PCA + L² Lloyd + Cascade-aware Water-Filling (CWF)
  Importance = sensitivity[l] × tr(M[l,h])
  Sensitivity from Exp4/Next-3 direct ΔPPL measurements
  Allocation via greedy WF with floor=2
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

DEVICE = 'cuda:0'
DTYPE = torch.bfloat16
N_CALIB_TOKENS = 1024
N_EVAL_TOKENS = 2048

B_FLOOR = 2
B_MAX = 6

OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Direct sensitivities from Exp4 (Mistral) and Next-3 (Qwen)
EXP4_MISTRAL_DELTA_PPL = {
    0: 0.005, 1: 0.120, 2: 0.555, 3: 0.287, 4: 0.521, 5: 0.206,
    6: 0.304, 7: 0.166, 8: 0.152, 9: 0.160, 10: 0.070, 11: 0.079,
    12: 0.037, 13: 0.039, 14: 0.034, 15: 0.047, 16: 0.030, 17: 0.050,
    18: 0.025, 19: 0.024, 20: 0.067, 21: 0.067, 22: 0.155, 23: 0.122,
    24: 0.046, 25: 0.010, 26: -0.004, 27: 0.103, 28: 0.032, 29: 0.096,
    30: 0.116, 31: 0.028,
}

# From Next-3 (Qwen per-layer Lloyd substitution)
NEXT3_QWEN_DELTA_PPL = {
    0: 0.3583, 1: 0.0613, 2: 0.0590, 3: 0.0414, 4: 0.2101, 5: 0.1951,
    6: 0.0353, 7: -0.0075, 8: 0.0395, 9: 0.0048, 10: -0.0409, 11: 0.0167,
    12: 0.0646, 13: 0.0287, 14: 0.0089, 15: 0.0420, 16: 0.0332, 17: 0.0796,
    18: 0.0098, 19: 0.0356, 20: -0.0368, 21: 0.0177, 22: 0.0804, 23: 0.0269,
    24: 0.0714, 25: 0.0486, 26: 0.2194, 27: 0.0051,
}

MODELS = [
    ('mistralai/Mistral-7B-v0.3', 'mistral-7b', EXP4_MISTRAL_DELTA_PPL,
     [2.0, 2.156, 2.3, 2.5, 2.75, 3.0, 3.25, 3.5]),
    ('Qwen/Qwen2.5-7B', 'qwen-7b', NEXT3_QWEN_DELTA_PPL,
     [2.0, 2.156, 2.3, 2.5, 2.75, 3.0]),
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

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


def collect_k_and_fisher(model, input_ids, n_layers, n_kv, n_q, head_dim):
    captured = {}
    handles = []
    def kh(li):
        def h(m, i, o):
            captured.setdefault(li, {})['k'] = o.detach().cpu().float().numpy()
        return h
    def qh(li):
        def h(m, i, o):
            captured.setdefault(li, {})['q'] = o.detach().cpu().float().numpy()
        return h
    def ah(li):
        def h(m, i, o):
            if isinstance(o, tuple) and len(o) >= 2 and o[1] is not None:
                captured.setdefault(li, {})['attn'] = o[1].detach().cpu().float().numpy()
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

    T = input_ids.shape[1]
    n_q_per_kv = n_q // n_kv
    trace_table = np.zeros((n_layers, n_kv), dtype=np.float32)
    K_table = {}

    for li in range(n_layers):
        data = captured.get(li, {})
        if not all(k in data for k in ['k', 'q', 'attn']):
            continue
        T_c = data['k'].shape[1]
        K_all = data['k'].reshape(T_c, n_kv, head_dim).astype(np.float32)
        Q_all = data['q'].reshape(T_c, n_q, head_dim).astype(np.float32)
        attn_all = data['attn'][0].astype(np.float32)

        for hk in range(n_kv):
            K = K_all[:, hk, :]
            q_heads = list(range(hk * n_q_per_kv, (hk+1) * n_q_per_kv))
            Q = Q_all[:, q_heads, :].mean(axis=1)
            attn_mean = attn_all[q_heads, :, :].mean(axis=0)
            s_t = (attn_mean * (1.0 - attn_mean)).sum(axis=1)
            M = ((Q * s_t[:, None]).T @ Q) / max(T_c, 1)
            trace_table[li, hk] = float(np.trace(M))
            K_table[(li, hk)] = K

    return K_table, trace_table


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


def make_bits_table(layer_weights, n_layers, n_kv, avg_bits, trace_table,
                    use_trace_weight=True, b_floor=2, b_max=6):
    """Build bits allocation from layer_weights × (optional) trace."""
    total_budget = int(round(n_layers * n_kv * avg_bits))
    importance_flat = []
    index_map = []
    for li in range(n_layers):
        for hk in range(n_kv):
            layer_w = float(layer_weights[li])
            if use_trace_weight:
                imp = layer_w * float(trace_table[li, hk])
            else:
                imp = layer_w
            importance_flat.append(imp)
            index_map.append((li, hk))
    bits_flat = water_filling_global(
        np.array(importance_flat), total_budget, b_floor, b_max
    )
    bits_table = np.zeros((n_layers, n_kv), dtype=int)
    for k, (li, hk) in enumerate(index_map):
        bits_table[li, hk] = bits_flat[k]
    return bits_table


def run_model(model_name, short_name, delta_ppl_dict, avg_bits_list):
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}", flush=True)
    t_start = time.time()

    print("Loading...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=DTYPE, device_map=DEVICE,
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t_start:.1f}s", flush=True)

    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_q
    print(f"  n_layers={n_layers}, n_kv={n_kv}, head_dim={head_dim}", flush=True)

    calib_text, eval_text = get_texts(tok)
    calib_enc = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB_TOKENS)
    calib_ids = calib_enc['input_ids'].to(DEVICE)
    eval_enc = tok(eval_text, return_tensors='pt', truncation=True, max_length=N_EVAL_TOKENS)
    eval_ids = eval_enc['input_ids'].to(DEVICE)

    # Baseline FP16
    print("\n[Baseline] FP16 PPL...", flush=True)
    ppl_fp16, _ = compute_ppl(model, eval_ids)
    print(f"  FP16 PPL: {ppl_fp16:.4f}", flush=True)

    # Collect keys + Fisher traces
    print("\n[Phase 1] Collecting K + Fisher metrics...", flush=True)
    t0 = time.time()
    K_table, trace_table = collect_k_and_fisher(model, calib_ids, n_layers, n_kv, n_q, head_dim)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    # Prepare sensitivity weights
    sens = np.zeros(n_layers, dtype=np.float32)
    for li in range(n_layers):
        sens[li] = max(0.0, delta_ppl_dict.get(li, 0.0)) + 1e-6

    print(f"  Top 5 layers by sensitivity: "
          f"{np.argsort(sens)[::-1][:5].tolist()}", flush=True)

    results = {
        'model': model_name,
        'ppl_fp16': ppl_fp16,
        'n_layers': n_layers,
        'n_kv': n_kv,
        'head_dim': head_dim,
        'configs': {},
    }

    # Test each avg_bits target
    for avg_bits in avg_bits_list:
        cfg_name = f'cwf_avg{avg_bits}'
        print(f"\n[{cfg_name}]", flush=True)

        bits_table = make_bits_table(
            sens, n_layers, n_kv, avg_bits, trace_table,
            use_trace_weight=True, b_floor=B_FLOOR, b_max=B_MAX
        )
        actual_avg = bits_table.mean()
        unique_bits, counts = np.unique(bits_table, return_counts=True)
        dist = dict(zip(unique_bits.tolist(), counts.tolist()))
        print(f"  Bit dist: {dist}, avg: {actual_avg:.3f}", flush=True)

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

        fit_time = time.time() - t_fit

        # Install hooks and measure PPL
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
        print(f"  PPL: {ppl:.4f} (Δ: {delta:+.2f}%)  [fit: {fit_time:.0f}s, eval: {time.time()-t0:.1f}s]", flush=True)

        results['configs'][cfg_name] = {
            'avg_bits_target': avg_bits,
            'avg_bits_actual': float(actual_avg),
            'ppl': ppl,
            'delta_vs_fp16_pct': delta,
            'bit_distribution': {int(k): int(v) for k, v in dist.items()},
        }

    results['runtime_sec'] = time.time() - t_start
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results


def main():
    print("=" * 70)
    print("Next-10: CWF Extended — Qwen cross-verification + avg_bits sweep")
    print("=" * 70, flush=True)
    t_total = time.time()

    all_results = {}
    for model_name, short_name, delta_dict, avg_list in MODELS:
        try:
            res = run_model(model_name, short_name, delta_dict, avg_list)
            all_results[short_name] = res
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results[short_name] = {'error': str(e)}

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for short, res in all_results.items():
        if 'error' in res:
            print(f"\n{short}: ERROR {res['error']}")
            continue
        print(f"\n{short} (FP16={res['ppl_fp16']:.4f}):")
        print(f"{'Config':<20} | {'avg_bits':>10} | {'PPL':>10} | {'Δ FP16':>10}")
        print('-' * 60)
        for cfg_name, cfg in res['configs'].items():
            if 'ppl' in cfg:
                print(f"  {cfg_name:<18} | {cfg['avg_bits_actual']:>10.3f} | "
                      f"{cfg['ppl']:>10.4f} | {cfg['delta_vs_fp16_pct']:>+9.2f}%")

    # Reference comparisons
    print("\n=== Reference points ===")
    print("  Mistral v3 Uniform 2b: 6.4614")
    print("  Mistral v3 WF floor=2 (best): 5.8222")
    print("  Qwen v3 Uniform 2b: 7.9804")
    print("  Qwen v3 WF floor=2: 7.0985")

    # Save
    all_results['_meta'] = {
        'total_runtime_sec': time.time() - t_total,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    out_file = OUT_DIR / 'exp_next10_cwf_extended.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_file}")
    print(f"Total runtime: {all_results['_meta']['total_runtime_sec']:.0f}s "
          f"({all_results['_meta']['total_runtime_sec']/60:.1f}m)")


if __name__ == '__main__':
    main()
