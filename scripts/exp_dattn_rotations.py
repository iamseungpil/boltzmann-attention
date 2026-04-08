#!/usr/bin/env python3
"""
D_attn measurement across 4 rotation types (no quantization needed).

D_attn(R) = Σ_j [R^T Σ_Q R]_jj × [R^T Σ_K R]_jj

Tests rearrangement inequality prediction:
  D_attn(Anti-PCA) < D_attn(Random) < D_attn(PCA)

Also computes D_mse(R) = trace(Σ_K) (should be rotation-invariant).

Usage:
  python exp_dattn_rotations.py --model mistralai/Mistral-7B-v0.3
  (runs on CPU, no GPU needed for the D_attn computation itself)
"""
import argparse, json, time, os, gc
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import numpy as np
import torch
from pathlib import Path

DTYPE = torch.bfloat16
CALIB_TOKENS = 4096  # more than 2K for stable covariance estimation


def compute_dattn(Sigma_K, Sigma_Q, R):
    """D_attn = Σ_j [R^T Σ_Q R]_jj × [R^T Σ_K R]_jj"""
    RtQR = R.T @ Sigma_Q @ R
    RtKR = R.T @ Sigma_K @ R
    return float(np.sum(np.diag(RtQR) * np.diag(RtKR)))


def compute_dmse(Sigma_K, R):
    """D_mse = trace(R^T Σ_K R) = trace(Σ_K) — should be rotation-invariant."""
    return float(np.trace(R.T @ Sigma_K @ R))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="results/dattn_rotations")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    short = args.model.split("/")[-1].replace(".", "_")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"{'='*60}")
    print(f"D_attn ROTATION ANALYSIS: {args.model}")
    print(f"{'='*60}")

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

    # Calibration
    calib_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_text = "\n\n".join([t for t in calib_ds["text"] if t.strip()])
    calib_ids = tokenizer.encode(calib_text, return_tensors="pt", truncation=False)[:, :CALIB_TOKENS].to(device)

    # Extract K and Q
    print("\n[Extracting K and Q...]")
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

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Extracted {n_layers} layers")

    # Compute covariances and D_attn per head
    results = {"model": args.model, "n_layers": n_layers, "n_kv": n_kv, "d_head": d_head,
               "per_head": [], "aggregate": {}}

    all_dattn = {"norot": [], "pca": [], "anti_pca": [], "random_mean": []}
    all_dmse = {"norot": [], "pca": []}
    random_seeds = [42, 123, 456]

    for li in range(n_layers):
        k_np = k_data[li].reshape(-1, n_kv, d_head)
        q_np = q_data[li].reshape(-1, n_heads, d_head)

        for hk in range(n_kv):
            K = k_np[:, hk, :].astype(np.float64)
            Q_group = q_np[:, hk*G:(hk+1)*G, :].mean(axis=1).astype(np.float64)

            # Covariances
            K_c = K - K.mean(0)
            Q_c = Q_group - Q_group.mean(0)
            N = K.shape[0]
            Sigma_K = (K_c.T @ K_c) / (N - 1) + 1e-8 * np.eye(d_head)
            Sigma_Q = (Q_c.T @ Q_c) / (N - 1) + 1e-8 * np.eye(d_head)

            # PCA rotation
            ev_K, V_K = np.linalg.eigh(Sigma_K)
            V_K = V_K[:, ::-1]  # descending
            ev_K = ev_K[::-1]

            # Query variance in PCA basis
            Sigma_Q_pca = V_K.T @ Sigma_Q @ V_K
            sigma2_q = np.diag(Sigma_Q_pca)

            # Rank correlation
            from scipy.stats import spearmanr
            rho, _ = spearmanr(ev_K, sigma2_q)

            # --- 4 rotations ---

            # 1. NoRot (identity)
            R_norot = np.eye(d_head)
            dattn_norot = compute_dattn(Sigma_K, Sigma_Q, R_norot)
            dmse_norot = compute_dmse(Sigma_K, R_norot)

            # 2. PCA
            R_pca = V_K
            dattn_pca = compute_dattn(Sigma_K, Sigma_Q, R_pca)
            dmse_pca = compute_dmse(Sigma_K, R_pca)

            # 3. Anti-PCA: PCA columns reordered so high λ_K pairs with low σ²_Q
            pca_order = np.arange(d_head)  # already sorted by descending λ_K
            q_order = np.argsort(sigma2_q)  # ascending σ²_Q
            # Anti-PCA: PCA dim with largest λ_K gets column position with smallest σ²_Q
            anti_perm = np.zeros(d_head, dtype=int)
            anti_perm[q_order] = pca_order  # place PCA dims into q-ascending slots
            R_anti = V_K[:, anti_perm]
            dattn_anti = compute_dattn(Sigma_K, Sigma_Q, R_anti)

            # 4. Random (average over seeds)
            dattn_randoms = []
            for seed in random_seeds:
                np.random.seed(seed * 10000 + li * 100 + hk)
                R_rand = np.linalg.qr(np.random.randn(d_head, d_head))[0]
                dattn_randoms.append(compute_dattn(Sigma_K, Sigma_Q, R_rand))
            dattn_random_mean = float(np.mean(dattn_randoms))

            # Collect
            all_dattn["norot"].append(dattn_norot)
            all_dattn["pca"].append(dattn_pca)
            all_dattn["anti_pca"].append(dattn_anti)
            all_dattn["random_mean"].append(dattn_random_mean)
            all_dmse["norot"].append(dmse_norot)
            all_dmse["pca"].append(dmse_pca)

            results["per_head"].append({
                "layer": li, "head": hk,
                "rho_lambda_sigma_q": round(rho, 4),
                "dattn_norot": round(dattn_norot, 6),
                "dattn_pca": round(dattn_pca, 6),
                "dattn_anti_pca": round(dattn_anti, 6),
                "dattn_random": round(dattn_random_mean, 6),
                "dmse_norot": round(dmse_norot, 6),
                "dmse_pca": round(dmse_pca, 6),
                "dmse_invariant": abs(dmse_norot - dmse_pca) < 1e-4 * dmse_norot,
            })

    # Aggregate
    for rot in ["norot", "pca", "anti_pca", "random_mean"]:
        vals = all_dattn[rot]
        results["aggregate"][f"dattn_{rot}_mean"] = round(float(np.mean(vals)), 6)
        results["aggregate"][f"dattn_{rot}_sum"] = round(float(np.sum(vals)), 4)

    # D_mse invariance check
    dmse_diff = [abs(a - b) / max(a, 1e-10) for a, b in zip(all_dmse["norot"], all_dmse["pca"])]
    results["aggregate"]["dmse_invariant_pct"] = round(sum(1 for d in dmse_diff if d < 0.001) / len(dmse_diff) * 100, 1)
    results["aggregate"]["dmse_max_diff_pct"] = round(max(dmse_diff) * 100, 4)

    # Rearrangement check
    n_heads_total = len(all_dattn["pca"])
    anti_wins = sum(1 for a, p in zip(all_dattn["anti_pca"], all_dattn["pca"]) if a < p)
    pca_beats_random = sum(1 for p, r in zip(all_dattn["pca"], all_dattn["random_mean"]) if p < r)

    results["aggregate"]["anti_pca_wins_pca"] = f"{anti_wins}/{n_heads_total}"
    results["aggregate"]["pca_beats_random"] = f"{pca_beats_random}/{n_heads_total}"

    # Print summary
    print(f"\n{'='*60}")
    print(f"D_attn RESULTS: {args.model}")
    print(f"{'='*60}")
    print(f"D_mse rotation-invariant: {results['aggregate']['dmse_invariant_pct']}% of heads")
    print(f"D_mse max diff: {results['aggregate']['dmse_max_diff_pct']}%")
    print(f"\nD_attn (sum across all heads):")
    print(f"  NoRot:     {results['aggregate']['dattn_norot_sum']:.4f}")
    print(f"  PCA:       {results['aggregate']['dattn_pca_sum']:.4f}")
    print(f"  Anti-PCA:  {results['aggregate']['dattn_anti_pca_sum']:.4f}")
    print(f"  Random:    {results['aggregate']['dattn_random_mean_sum']:.4f}")
    print(f"\nRearrangement inequality:")
    print(f"  Anti-PCA < PCA: {anti_wins}/{n_heads_total} heads")
    print(f"  PCA < Random:   {pca_beats_random}/{n_heads_total} heads")

    # Theoretical prediction
    if results['aggregate']['dattn_anti_pca_sum'] < results['aggregate']['dattn_random_mean_sum'] < results['aggregate']['dattn_pca_sum']:
        print(f"\n  ✓ Rearrangement prediction CONFIRMED: Anti-PCA < Random < PCA")
    else:
        ordering = sorted([
            ("Anti-PCA", results['aggregate']['dattn_anti_pca_sum']),
            ("Random", results['aggregate']['dattn_random_mean_sum']),
            ("PCA", results['aggregate']['dattn_pca_sum']),
            ("NoRot", results['aggregate']['dattn_norot_sum']),
        ], key=lambda x: x[1])
        print(f"\n  Actual ordering: {' < '.join(f'{n}({v:.4f})' for n, v in ordering)}")

    # Save
    out_path = out_dir / f"{short}_dattn.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
