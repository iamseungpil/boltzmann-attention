#!/usr/bin/env python3
"""
Idea 4: L∞ vs L² measurement for Lloyd-Max vs Uniform quantization.
Validates Proposition 3: "Softmax converts L² to effective L∞."

For each (layer, head):
  1. Extract calibration keys, fit PCA
  2. Quantize with Uniform 2-bit and Lloyd-Max 2-bit
  3. Measure: L² MSE, L∞ max-error, attention-weighted error
  4. Compare: does Lloyd have worse L∞ despite better L²?

Then correlate with per-layer sensitivity (ΔPPL) to show L∞ predicts PPL better than L².

Usage:
  CUDA_VISIBLE_DEVICES=3 python exp_linf_vs_l2.py --model mistralai/Mistral-7B-v0.3
"""
import argparse, json, time, gc, os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import numpy as np
import torch
from pathlib import Path

DTYPE = torch.bfloat16
CALIB_TOKENS = 2048
BITS = 2


def lloyd_max_1d(col, bits, n_iter=30):
    n_levels = 2 ** bits
    pcts = np.linspace(0, 100, n_levels + 2)[1:-1]
    centroids = np.sort(np.percentile(col, pcts))
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


def uniform_quant(col, bits):
    nl = 2 ** bits
    vmin, vmax = col.min(), col.max()
    if vmax - vmin < 1e-10:
        return col.copy()
    s = (vmax - vmin) / (nl - 1)
    q = np.clip(np.round((col - vmin) / s).astype(int), 0, nl - 1)
    return q * s + vmin


def lloyd_quant(col, centroids):
    boundaries = (centroids[:-1] + centroids[1:]) / 2
    idx = np.searchsorted(boundaries, col)
    return centroids[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="results/linf_vs_l2")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    short = args.model.split("/")[-1].replace(".", "_")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"Loading {args.model}...")
    tok_kw = {"trust_remote_code": True}
    mdl_kw = {"torch_dtype": DTYPE, "trust_remote_code": True}
    if args.hf_token:
        tok_kw["token"] = args.hf_token
        mdl_kw["token"] = args.hf_token

    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kw)
    model = AutoModelForCausalLM.from_pretrained(args.model, **mdl_kw).to(device).eval()

    cfg = model.config
    n_kv = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    n_heads = cfg.num_attention_heads
    n_layers = cfg.num_hidden_layers
    d_head = cfg.hidden_size // n_heads
    G = n_heads // n_kv

    # Calibration data
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    calib_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=CALIB_TOKENS).to(device)

    # Capture K and Q
    print("Capturing K and Q...")
    k_data, q_data = {}, {}
    hooks = []
    def kh(li):
        def fn(mod, inp, out):
            k_data[li] = out.detach().cpu().float().numpy()
        return fn
    def qh(li):
        def fn(mod, inp, out):
            q_data[li] = out.detach().cpu().float().numpy()
        return fn
    for li in range(n_layers):
        hooks.append(model.model.layers[li].self_attn.k_proj.register_forward_hook(kh(li)))
        hooks.append(model.model.layers[li].self_attn.q_proj.register_forward_hook(qh(li)))
    with torch.no_grad():
        model(calib_ids, use_cache=False)
    for h in hooks:
        h.remove()

    # Per-head measurement
    print(f"Measuring L∞ vs L² for {n_layers}×{n_kv} heads...")
    results = []

    for li in range(n_layers):
        k_np = k_data[li].reshape(-1, n_kv, d_head)
        q_np = q_data[li].reshape(-1, n_heads, d_head)

        for hk in range(n_kv):
            K = k_np[:, hk, :].astype(np.float32)
            Q_group = q_np[:, hk*G:(hk+1)*G, :].mean(axis=1).astype(np.float32)

            # PCA
            K_mean = K.mean(0)
            K_c = K - K_mean
            cov = (K_c.T @ K_c) / max(K.shape[0] - 1, 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            V = eigvecs[:, np.argsort(eigvals)[::-1]]
            K_pca = K_c @ V

            # Quantize each dim
            uni_recon = np.zeros_like(K_pca)
            lloyd_recon = np.zeros_like(K_pca)

            for j in range(d_head):
                col = K_pca[:, j]
                # Uniform
                uni_recon[:, j] = uniform_quant(col, BITS)
                # Lloyd-Max
                centroids = lloyd_max_1d(col, BITS)
                lloyd_recon[:, j] = lloyd_quant(col, centroids)

            # Errors in PCA space
            uni_err = K_pca - uni_recon
            lloyd_err = K_pca - lloyd_recon

            # L² MSE (per-head average)
            uni_l2 = float(np.mean(uni_err ** 2))
            lloyd_l2 = float(np.mean(lloyd_err ** 2))

            # L∞ max-error (per-head max across all tokens and dims)
            uni_linf = float(np.max(np.abs(uni_err)))
            lloyd_linf = float(np.max(np.abs(lloyd_err)))

            # Attention-weighted error: |q^T δk|
            # Reconstruct in original space
            uni_dk = uni_err @ V.T  # error in original space
            lloyd_dk = lloyd_err @ V.T

            # q^T δk for each token
            uni_attn_err = np.abs(np.sum(Q_group * uni_dk, axis=1))  # (T,)
            lloyd_attn_err = np.abs(np.sum(Q_group * lloyd_dk, axis=1))

            uni_attn_l2 = float(np.mean(uni_attn_err ** 2))
            lloyd_attn_l2 = float(np.mean(lloyd_attn_err ** 2))
            uni_attn_linf = float(np.max(uni_attn_err))
            lloyd_attn_linf = float(np.max(lloyd_attn_err))

            results.append({
                "layer": li, "head": hk,
                "uniform": {
                    "l2_mse": round(uni_l2, 8),
                    "linf_max": round(uni_linf, 6),
                    "attn_l2": round(uni_attn_l2, 8),
                    "attn_linf": round(uni_attn_linf, 6),
                },
                "lloyd": {
                    "l2_mse": round(lloyd_l2, 8),
                    "linf_max": round(lloyd_linf, 6),
                    "attn_l2": round(lloyd_attn_l2, 8),
                    "attn_linf": round(lloyd_attn_linf, 6),
                },
                "lloyd_wins_l2": lloyd_l2 < uni_l2,
                "uniform_wins_linf": uni_linf < lloyd_linf,
                "l2_ratio": round(lloyd_l2 / max(uni_l2, 1e-12), 4),
                "linf_ratio": round(lloyd_linf / max(uni_linf, 1e-12), 4),
            })

    # Aggregate
    n_heads_total = len(results)
    lloyd_wins_l2 = sum(1 for r in results if r["lloyd_wins_l2"])
    uni_wins_linf = sum(1 for r in results if r["uniform_wins_linf"])

    l2_ratios = [r["l2_ratio"] for r in results]
    linf_ratios = [r["linf_ratio"] for r in results]

    summary = {
        "model": args.model,
        "bits": BITS,
        "n_heads": n_heads_total,
        "lloyd_wins_l2": f"{lloyd_wins_l2}/{n_heads_total}",
        "uniform_wins_linf": f"{uni_wins_linf}/{n_heads_total}",
        "l2_ratio_mean": round(float(np.mean(l2_ratios)), 4),
        "l2_ratio_median": round(float(np.median(l2_ratios)), 4),
        "linf_ratio_mean": round(float(np.mean(linf_ratios)), 4),
        "linf_ratio_median": round(float(np.median(linf_ratios)), 4),
    }

    print(f"\n{'='*60}")
    print(f"RESULTS: {args.model} ({BITS}-bit)")
    print(f"{'='*60}")
    print(f"Lloyd wins L² (MSE): {lloyd_wins_l2}/{n_heads_total}")
    print(f"Uniform wins L∞ (max-error): {uni_wins_linf}/{n_heads_total}")
    print(f"L² ratio (Lloyd/Uniform): mean={np.mean(l2_ratios):.4f}, median={np.median(l2_ratios):.4f}")
    print(f"L∞ ratio (Lloyd/Uniform): mean={np.mean(linf_ratios):.4f}, median={np.median(linf_ratios):.4f}")
    print(f"\nIf Lloyd wins L² but Uniform wins L∞ → Proposition 3 CONFIRMED")
    print(f"Confirmation rate: {sum(1 for r in results if r['lloyd_wins_l2'] and r['uniform_wins_linf'])}/{n_heads_total}")

    # Save
    out_path = out_dir / f"{short}_linf_vs_l2_{BITS}bit.json"
    out_path.write_text(json.dumps({"summary": summary, "per_head": results}, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
