#!/usr/bin/env python3
"""
Theory validation experiments (T1-T5).
Tests theoretical predictions from the strengthened theory.

T3: Attention-aware quantizer vs standard quantizers
T4: Query-weighted PCA vs standard PCA
T5: Key+Value joint quantization

Usage:
  CUDA_VISIBLE_DEVICES=1 python run_theory_experiments.py --model Qwen/Qwen2.5-7B --exp t3 t4
"""
from __future__ import annotations
import argparse, json, time, gc, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import sqrtm

warnings.filterwarnings("ignore")


def uniform_quant_col(col, bits):
    nl = 2 ** bits
    vmin, vmax = col.min(), col.max()
    if vmax - vmin < 1e-10: return col.copy()
    s = (vmax - vmin) / (nl - 1)
    q = np.clip(np.round((col - vmin) / s).astype(int), 0, nl - 1)
    return q * s + vmin


def lloyd_max_col(col, bits, n_iter=10):
    """Standard Lloyd-Max quantizer (no weighting).
    Returns reconstructed column."""
    nl = 2 ** bits
    vmin, vmax = col.min(), col.max()
    if vmax - vmin < 1e-10: return col.copy()
    percentiles = np.linspace(0, 100, nl + 2)[1:-1]
    centroids = np.percentile(col, percentiles)
    for _ in range(n_iter):
        dists = np.abs(col[:, None] - centroids[None, :])
        assignments = np.argmin(dists, axis=1)
        for k in range(nl):
            mask = assignments == k
            if mask.sum() > 0:
                centroids[k] = col[mask].mean()
    dists = np.abs(col[:, None] - centroids[None, :])
    assignments = np.argmin(dists, axis=1)
    return centroids[assignments]


def weighted_bit_allocation(weights, total_bits, min_bits=2, max_bits=8):
    """Reverse water-filling: allocate more bits to dimensions with larger weights.
    Standard high-rate result: optimal b_j = b_avg + (1/2)·log2(σ_j² / GM(σ²))
    For attention-weighted, we use w_j = sqrt(Σ_Q,jj) · sqrt(Σ_K,jj) as effective σ_j.
    Returns integer bit allocation per dimension summing close to total_bits."""
    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 1e-10)
    d = len(weights)
    b_avg = total_bits / d
    log_w = np.log2(weights ** 2)
    gm_log = log_w.mean()
    bits_real = b_avg + 0.5 * (log_w - gm_log)
    bits_int = np.clip(np.round(bits_real).astype(int), min_bits, max_bits)
    # Adjust to match total budget
    diff = int(total_bits - bits_int.sum())
    if diff != 0:
        order = np.argsort(-(bits_real - bits_int)) if diff > 0 else np.argsort(bits_real - bits_int)
        for i in order[:abs(diff)]:
            bits_int[i] += 1 if diff > 0 else -1
            bits_int[i] = max(min_bits, min(max_bits, bits_int[i]))
    return bits_int


def attn_aware_quantize_block(K_block, bits_per_dim):
    """Real attention-aware quantization: uses different bit count per dimension.
    K_block: (n_samples, d_head)
    bits_per_dim: array of length d_head with int bits for each dimension.
    Returns reconstructed K_block."""
    out = np.empty_like(K_block)
    for j in range(K_block.shape[1]):
        out[:, j] = lloyd_max_col(K_block[:, j], int(bits_per_dim[j]))
    return out


def make_theory_hook(layer_idx, bits, n_kv, d_head, method,
                     pca_bases=None, qw_pca_bases=None,
                     query_weights=None, random_rot=None):
    """Hook for theory experiments."""
    def hook_fn(module, input, output):
        k = output[0] if isinstance(output, tuple) else output
        k_np = k.detach().cpu().float().numpy()
        orig_shape = k_np.shape
        k_flat = k_np.reshape(-1, n_kv, d_head)

        for h in range(n_kv):
            Kh = k_flat[:, h, :]

            # Choose rotation
            if method == 'qw_pca' and qw_pca_bases and (layer_idx, h) in qw_pca_bases:
                R = qw_pca_bases[(layer_idx, h)]
            elif method in ('standard_pca', 'attn_quant') and pca_bases and (layer_idx, h) in pca_bases:
                R = pca_bases[(layer_idx, h)]
            elif method == 'random_rot' and random_rot is not None:
                R = random_rot
            else:
                R = None

            Kr = Kh @ R if R is not None else Kh.copy()

            # Choose quantizer
            for j in range(d_head):
                if method == 'attn_quant' and query_weights and (layer_idx, h) in query_weights:
                    w = query_weights[(layer_idx, h)][j]
                    Kr[:, j] = weighted_lloyd_col(Kr[:, j], bits, weight=w)
                else:
                    Kr[:, j] = uniform_quant_col(Kr[:, j], bits)

            k_flat[:, h, :] = Kr @ R.T if R is not None else Kr

        return torch.tensor(k_flat.reshape(orig_shape), dtype=k.dtype, device=k.device)
    return hook_fn


def calibrate(model, device, n_kv, n_layers, d_head, layers):
    """Extract key AND query covariances."""
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, trust_remote_code=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join([t for t in ds["text"] if t.strip()][:50])
    cal_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=2048).to(device)

    key_capture = {}
    query_capture = {}
    hooks = []

    def make_hook(li):
        def fn(mod, args_t, kwargs_t):
            hs = args_t[0] if args_t else kwargs_t.get('hidden_states')
            if hs is not None:
                n_heads = model.config.num_attention_heads
                k = mod.k_proj(hs)[0].detach().cpu().float().numpy().reshape(-1, n_kv, d_head)
                q = mod.q_proj(hs)[0].detach().cpu().float().numpy().reshape(-1, n_heads, d_head)
                for h in range(n_kv):
                    key_capture[(li, h)] = k[:, h, :]
                # GQA: store ALL G query heads per KV group separately
                # (mean(Cov) ≠ Cov(mean) — Jensen, ensure correct GQA covariance)
                G = n_heads // n_kv
                for h in range(n_kv):
                    # Stack all G heads of this KV group: shape (G*seq, d_head)
                    q_group_stacked = q[:, h*G:(h+1)*G, :].reshape(-1, d_head)
                    query_capture[(li, h)] = q_group_stacked
        return fn

    for l in range(n_layers):
        hooks.append(layers[l].self_attn.register_forward_pre_hook(make_hook(l), with_kwargs=True))

    with torch.no_grad():
        model(cal_ids, use_cache=False)
    for hook in hooks:
        hook.remove()

    # Compute PCA bases
    pca_bases = {}
    eigenvalues = {}
    for (l, h), K in key_capture.items():
        Kc = K - K.mean(0)
        cov = Kc.T @ Kc / K.shape[0]
        eigvals, eigvecs = np.linalg.eigh(cov)
        pca_bases[(l, h)] = eigvecs
        eigenvalues[(l, h)] = np.maximum(eigvals, 1e-10)

    # Compute query covariance and query-weighted PCA + measurements
    qw_pca_bases = {}
    qw_inverse = {}  # Σ_Q^{-1/2} for proper qw_pca application in k-space
    sigma_q_half = {}  # Σ_Q^{1/2} for forward transform
    query_weights = {}
    sigma_q_diag = {}  # diagonal of Σ_Q in PCA basis (for water-filling)
    sigma_k_diag = {}  # diagonal of Σ_K in PCA basis
    kappa_q = {}
    kappa_k = {}
    for (l, h) in key_capture:
        Q = query_capture[(l, h)]
        Qc = Q - Q.mean(0)
        Sigma_Q = Qc.T @ Qc / Q.shape[0]
        Sigma_Q = (Sigma_Q + Sigma_Q.T) / 2 + 1e-6 * np.eye(d_head)

        K = key_capture[(l, h)]
        Kc = K - K.mean(0)
        Sigma_K = Kc.T @ Kc / K.shape[0]
        Sigma_K = (Sigma_K + Sigma_K.T) / 2 + 1e-6 * np.eye(d_head)

        # Measure condition numbers (no longer hand-picked)
        eigQ = np.linalg.eigvalsh(Sigma_Q)
        eigK = np.linalg.eigvalsh(Sigma_K)
        kappa_q[(l, h)] = float(eigQ.max() / max(eigQ.min(), 1e-10))
        kappa_k[(l, h)] = float(eigK.max() / max(eigK.min(), 1e-10))

        # Query-weighted PCA: eigvecs of Σ_Q^{1/2} · Σ_K · Σ_Q^{1/2}
        # Note: V_qw rotates in k̃ = Σ_Q^{1/2}·k space, so to apply on raw k
        # we need: k -> Σ_Q^{1/2}·k (transform in) -> V_qw^T·... -> Σ_Q^{-1/2} (transform out)
        # Store both Σ_Q^{1/2} and Σ_Q^{-1/2} for the hook to apply correctly.
        try:
            Sq_half = sqrtm(Sigma_Q).real
            Sq_half = (Sq_half + Sq_half.T) / 2
            Sq_inv = np.linalg.pinv(Sq_half)
            M = Sq_half @ Sigma_K @ Sq_half
            M = (M + M.T) / 2
            _, V_qw = np.linalg.eigh(M)
            qw_pca_bases[(l, h)] = V_qw
            sigma_q_half[(l, h)] = Sq_half
            qw_inverse[(l, h)] = Sq_inv
        except Exception:
            qw_pca_bases[(l, h)] = pca_bases[(l, h)]
            sigma_q_half[(l, h)] = np.eye(d_head)
            qw_inverse[(l, h)] = np.eye(d_head)

        # Per-dimension query weights (sqrt of diagonal of Σ_Q in PCA space)
        V = pca_bases[(l, h)]
        Sigma_Q_pca = V.T @ Sigma_Q @ V
        Sigma_K_pca = V.T @ Sigma_K @ V
        sq_diag = np.maximum(np.diag(Sigma_Q_pca), 1e-10)
        sk_diag = np.maximum(np.diag(Sigma_K_pca), 1e-10)
        sigma_q_diag[(l, h)] = sq_diag
        sigma_k_diag[(l, h)] = sk_diag
        query_weights[(l, h)] = np.sqrt(sq_diag)

    np.random.seed(42)
    R_random = np.linalg.qr(np.random.randn(d_head, d_head))[0]

    # Print spectrum summary (no more hand-picked κ)
    kq_vals = np.array(list(kappa_q.values()))
    kk_vals = np.array(list(kappa_k.values()))
    print(f"  Measured κ(Σ_Q):  median={np.median(kq_vals):.2f}  mean={kq_vals.mean():.2f}  max={kq_vals.max():.2f}")
    print(f"  Measured κ(Σ_K):  median={np.median(kk_vals):.2f}  mean={kk_vals.mean():.2f}  max={kk_vals.max():.2f}")

    return {
        'pca_bases': pca_bases, 'eigenvalues': eigenvalues,
        'qw_pca_bases': qw_pca_bases, 'query_weights': query_weights,
        'sigma_q_half': sigma_q_half, 'qw_inverse': qw_inverse,
        'sigma_q_diag': sigma_q_diag, 'sigma_k_diag': sigma_k_diag,
        'kappa_q': kappa_q, 'kappa_k': kappa_k,
        'R_random': R_random, 'key_capture': key_capture, 'query_capture': query_capture,
    }


def eval_ppl(model, all_ids, layers, n_layers, n_kv, d_head, device,
             bits, method, cal_data, max_tokens=50000):
    """Evaluate PPL with a given theory method."""
    chunk_len, n_chunks = 2048, (min(all_ids.shape[1], max_tokens) - 1) // 2048
    hks = []
    for l in range(n_layers):
        hks.append(layers[l].self_attn.k_proj.register_forward_hook(
            make_theory_hook(l, bits, n_kv, d_head, method,
                             pca_bases=cal_data['pca_bases'],
                             qw_pca_bases=cal_data['qw_pca_bases'],
                             query_weights=cal_data['query_weights'],
                             random_rot=cal_data['R_random'])))
    nll, tok = 0.0, 0
    with torch.no_grad():
        for ci in range(n_chunks):
            s = ci * chunk_len
            e = s + chunk_len + 1
            if e > min(all_ids.shape[1], max_tokens): break
            inp = all_ids[:, s:e].to(device)
            out = model(inp[:, :-1], use_cache=False)
            nll += nn.CrossEntropyLoss(reduction='sum')(out.logits[0], inp[0, 1:]).item()
            tok += inp.shape[1] - 1
    for hk in hks:
        hk.remove()
    ppl = np.exp(nll / tok)
    print(f"    {method}_{bits}b: PPL={ppl:.4f} ({tok} tokens)")
    return float(ppl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--exp", type=str, nargs="+", default=["t3", "t4"])
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    device = "cuda:0"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"\n{'#'*70}")
    print(f"# Theory Experiments: {args.model}")
    print(f"# Experiments: {args.exp}")
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

    # Calibrate (extracts both K and Q covariances)
    print("\n  Calibrating (K + Q covariances)...")
    cal_data = calibrate(model, device, n_kv, n_layers, d_head, layers)
    print(f"  Done: {len(cal_data['pca_bases'])} heads")

    results = {'model': args.model, 'experiments': {}}

    for exp in args.exp:
        print(f"\n{'='*60}")
        print(f"Experiment {exp.upper()}")
        print(f"{'='*60}")

        if exp == 't3':
            # T3: Attention-aware quantizer comparison
            for bits in args.bits:
                print(f"\n  --- {bits}-bit ---")
                methods = ['standard_pca', 'attn_quant']
                for method in methods:
                    ppl = eval_ppl(model, all_ids, layers, n_layers, n_kv, d_head,
                                   device, bits, method, cal_data)
                    results['experiments'][f't3_{method}_{bits}bit'] = ppl

        elif exp == 't4':
            # T4: Query-weighted PCA vs standard PCA
            for bits in args.bits:
                print(f"\n  --- {bits}-bit ---")
                for method in ['standard_pca', 'qw_pca']:
                    ppl = eval_ppl(model, all_ids, layers, n_layers, n_kv, d_head,
                                   device, bits, method, cal_data)
                    results['experiments'][f't4_{method}_{bits}bit'] = ppl

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.model.replace("/", "_")
    outf = Path(args.output_dir) / f"theory_{tag}_{ts}.json"
    with open(outf, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outf}")

    del model; torch.cuda.empty_cache(); gc.collect()


if __name__ == '__main__':
    main()
