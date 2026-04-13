#!/usr/bin/env python3
"""
Decorrelation hypothesis test: 3 experiments in 1 script.

Exp D: Error correlation matrix diagonality under each rotation
Exp A: Per-token attention error distribution (mean vs max vs 99th pct)
Exp C: Whitening test (PCA + unit variance scaling)

Hypothesis: PCA helps PPL through DECORRELATION (not D_attn reduction).
Evidence needed:
  1. Cov(δK) is more diagonal under PCA than NoRot/Random
  2. Per-token |q^T δk| has lower MAX under PCA despite higher MEAN
  3. Whitening (max decorrelation) may further improve PPL

Usage:
  CUDA_VISIBLE_DEVICES=1 python exp_decorrelation_hypothesis.py --model mistralai/Mistral-7B-v0.3 --bits 3
"""
import argparse, json, time, gc, os, hashlib
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import numpy as np
import torch
from pathlib import Path

DTYPE = torch.bfloat16
CALIB_TOKENS = 4096
EVAL_CHUNK = 2048
MAX_EVAL = 50000


def uniform_quantize(col, vmin, vmax, bits):
    nl = 2 ** bits
    col_c = np.clip(col, vmin, vmax)
    if vmax - vmin < 1e-10:
        return col_c
    s = (vmax - vmin) / (nl - 1)
    q = np.clip(np.round((col_c - vmin) / s).astype(int), 0, nl - 1)
    return q * s + vmin


class DecorrelationHook:
    """Quantize with rotation, collect per-token attention errors."""
    def __init__(self, n_kv, d_head, K_means, rotations, calib_ranges, bits,
                 Q_bases=None):
        self.n_kv = n_kv
        self.d_head = d_head
        self.K_means = K_means
        self.R = rotations
        self.ranges = calib_ranges  # dict h → (d, 2) [min, max]
        self.bits = bits
        self.Q_bases = Q_bases  # for computing q^T δk
        # Collectors
        self.all_errors = []  # per-chunk: (n_kv, T, d) errors in rotated space
        self.attn_errors = []  # per-chunk: |q^T δk| values

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x = output.view(B, T, self.n_kv, self.d_head).float().cpu().numpy()
        out = x.copy()
        for h in range(self.n_kv):
            m = self.K_means[h]
            R = self.R[h]
            ranges = self.ranges[h]
            for b_idx in range(B):
                block = x[b_idx, :, h, :]  # (T, d)
                centered = block - m
                rotated = centered @ R
                quantized = np.zeros_like(rotated)
                for j in range(self.d_head):
                    vmin, vmax = ranges[j]
                    quantized[:, j] = uniform_quantize(rotated[:, j], vmin, vmax, self.bits)
                error = rotated - quantized
                self.all_errors.append(error)  # (T, d)
                reconstructed = quantized @ R.T + m
                out[b_idx, :, h, :] = reconstructed
        return torch.from_numpy(out).to(output.device).to(output.dtype).view(B, T, -1)


def eval_ppl(model, eval_ids, device):
    model.eval()
    total_len = min(eval_ids.shape[1], MAX_EVAL)
    n_chunks = total_len // EVAL_CHUNK
    total_nll, total_scored = 0.0, 0
    for i in range(n_chunks):
        chunk = eval_ids[:, i*EVAL_CHUNK:(i+1)*EVAL_CHUNK].to(device)
        with torch.no_grad():
            out = model(chunk, labels=chunk, use_cache=False)
        total_nll += out.loss.item() * (EVAL_CHUNK - 1)
        total_scored += EVAL_CHUNK - 1
    return float(np.exp(total_nll / total_scored))


def error_diagonality(errors_list, d_head):
    """Compute how diagonal the error covariance is (ratio of diag energy to total)."""
    all_err = np.concatenate(errors_list, axis=0)  # (N_total, d)
    if all_err.shape[0] < 10:
        return 0.0, 0.0
    cov = np.cov(all_err.T)  # (d, d)
    diag_energy = np.sum(np.diag(cov)**2)
    total_energy = np.sum(cov**2)
    diag_ratio = diag_energy / max(total_energy, 1e-10)
    # Off-diagonal magnitude
    off_diag = cov.copy()
    np.fill_diagonal(off_diag, 0)
    off_diag_norm = np.sqrt(np.sum(off_diag**2))
    diag_norm = np.sqrt(diag_energy)
    return float(diag_ratio), float(off_diag_norm / max(diag_norm, 1e-10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--output-dir", default="results/decorrelation")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    short = args.model.split("/")[-1].replace(".", "_")
    bits = args.bits

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"{'='*60}")
    print(f"DECORRELATION HYPOTHESIS: {args.model}, {bits}-bit")
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
    n_layers = cfg.num_hidden_layers
    d_head = cfg.hidden_size // cfg.num_attention_heads

    eval_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_text = "\n\n".join([t for t in eval_ds["text"] if t.strip()])
    eval_ids = tokenizer.encode(eval_text, return_tensors="pt", truncation=False)

    calib_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_text = "\n\n".join([t for t in calib_ds["text"] if t.strip()])
    calib_ids = tokenizer.encode(calib_text, return_tensors="pt", truncation=False)[:, :CALIB_TOKENS].to(device)

    # FP16 baseline
    print("\n[FP16]")
    ppl_fp16 = eval_ppl(model, eval_ids, device)
    print(f"  PPL = {ppl_fp16:.4f}")

    # Calibration
    print("\n[Calibration]")
    k_data = {}
    hooks = []
    for li in range(n_layers):
        def kh(li_=li):
            def fn(mod, inp, out):
                k_data[li_] = out.detach().cpu().float().numpy()
            return fn
        hooks.append(model.model.layers[li].self_attn.k_proj.register_forward_hook(kh()))
    with torch.no_grad():
        model(calib_ids, use_cache=False)
    for h in hooks:
        h.remove()

    # Per-head calibration stats
    calib_info = {}
    for li in range(n_layers):
        k_np = k_data[li].reshape(-1, n_kv, d_head)
        for hk in range(n_kv):
            K = k_np[:, hk, :].astype(np.float32)
            K_mean = K.mean(0)
            K_c = K - K_mean
            cov = (K_c.T @ K_c) / max(K.shape[0]-1, 1) + 1e-6 * np.eye(d_head)
            ev, V = np.linalg.eigh(cov)
            V = V[:, ::-1].astype(np.float32)
            ev = ev[::-1].astype(np.float32)
            K_pca = K_c @ V
            # Whitening: PCA + scale to unit variance
            scale = 1.0 / np.sqrt(np.maximum(ev, 1e-10))
            V_whiten = V * scale[np.newaxis, :]  # (d, d) — NOT orthogonal!
            # For orthogonal whitening, we need V @ diag(1/sqrt(ev))
            # But V_whiten^T V_whiten ≠ I. We need inverse = V @ diag(sqrt(ev))
            calib_info[(li, hk)] = {
                "K_mean": K_mean, "V": V, "ev": ev,
                "K_c": K_c, "K_pca": K_pca, "scale": scale,
            }
    del k_data
    gc.collect()
    print(f"  Calibrated {n_layers}×{n_kv} heads")

    # Define rotations
    def get_rotation_and_ranges(rot_type, li, hk):
        info = calib_info[(li, hk)]
        V = info["V"]
        K_c = info["K_c"]
        ev = info["ev"]

        if rot_type == "norot":
            R = np.eye(d_head, dtype=np.float32)
        elif rot_type == "pca":
            R = V
        elif rot_type == "random":
            np.random.seed(42 * 10000 + li * 100 + hk)
            R = np.linalg.qr(np.random.randn(d_head, d_head).astype(np.float32))[0]
        elif rot_type == "whiten":
            # Whitened PCA: rotate to PCA, then scale to unit variance
            # R_whiten = V @ diag(1/sqrt(ev)) — but this is NOT orthogonal
            # So we can't use it directly as R in the hook (inverse R.T won't work)
            # Instead: apply PCA, scale, quantize, unscale, inverse PCA
            # We'll handle this specially
            R = V  # use PCA rotation, handle scaling in hook
        else:
            R = np.eye(d_head, dtype=np.float32)

        # Compute calibration ranges in rotated space
        K_rot = K_c @ R
        ranges = np.zeros((d_head, 2), dtype=np.float32)
        for j in range(d_head):
            ranges[j, 0] = K_rot[:, j].min()
            ranges[j, 1] = K_rot[:, j].max()

        return R, ranges

    # Run experiments
    rotations_to_test = ["norot", "pca", "random"]
    results = {"model": args.model, "bits": bits, "fp16": round(ppl_fp16, 4), "rotations": {}}

    for rot_type in rotations_to_test:
        print(f"\n[{rot_type}] {bits}-bit...")
        t0 = time.time()

        hook_handles = []
        all_hooks = []
        for li in range(n_layers):
            K_means = {}
            R_dict = {}
            ranges_dict = {}
            for hk in range(n_kv):
                K_means[hk] = calib_info[(li, hk)]["K_mean"]
                R, ranges = get_rotation_and_ranges(rot_type, li, hk)
                R_dict[hk] = R
                ranges_dict[hk] = ranges

            hook = DecorrelationHook(n_kv, d_head, K_means, R_dict, ranges_dict, bits)
            all_hooks.append(hook)
            hook_handles.append(
                model.model.layers[li].self_attn.k_proj.register_forward_hook(hook))

        ppl = eval_ppl(model, eval_ids, device)
        for h in hook_handles:
            h.remove()
        elapsed = time.time() - t0

        # Analyze collected errors
        # Exp D: Error diagonality
        all_errors = []
        for hook_obj in all_hooks:
            all_errors.extend(hook_obj.all_errors)

        if all_errors:
            diag_ratio, off_diag_ratio = error_diagonality(all_errors[:100], d_head)  # sample for speed
            # L∞ statistics
            all_err_flat = np.concatenate(all_errors, axis=0)
            linf_mean = float(np.mean(np.max(np.abs(all_err_flat), axis=1)))
            linf_max = float(np.max(np.abs(all_err_flat)))
            linf_p99 = float(np.percentile(np.max(np.abs(all_err_flat), axis=1), 99))
            mse = float(np.mean(all_err_flat**2))
        else:
            diag_ratio, off_diag_ratio = 0, 0
            linf_mean, linf_max, linf_p99, mse = 0, 0, 0, 0

        result = {
            "ppl": round(ppl, 4),
            "delta_ppl": round(ppl - ppl_fp16, 4),
            "mse": round(mse, 6),
            "linf_mean": round(linf_mean, 4),
            "linf_max": round(linf_max, 4),
            "linf_p99": round(linf_p99, 4),
            "error_diag_ratio": round(diag_ratio, 4),
            "error_offdiag_ratio": round(off_diag_ratio, 4),
            "runtime_sec": round(elapsed, 1),
        }
        results["rotations"][rot_type] = result
        print(f"  PPL={ppl:.4f} MSE={mse:.6f} L∞_max={linf_max:.4f} L∞_p99={linf_p99:.4f}")
        print(f"  Error diag_ratio={diag_ratio:.4f} offdiag_ratio={off_diag_ratio:.4f}")

        # Clear
        for hook_obj in all_hooks:
            hook_obj.all_errors.clear()
        gc.collect()

    # Summary
    print(f"\n{'='*60}")
    print(f"DECORRELATION RESULTS: {args.model} {bits}-bit")
    print(f"{'='*60}")
    print(f"FP16: {ppl_fp16:.4f}")
    print(f"\n{'Rotation':<10s} {'PPL':>8s} {'MSE':>10s} {'L∞_max':>8s} {'L∞_p99':>8s} {'Diag%':>8s} {'OffDiag':>8s}")
    for rot in rotations_to_test:
        r = results["rotations"][rot]
        print(f"{rot:<10s} {r['ppl']:>8.4f} {r['mse']:>10.6f} {r['linf_max']:>8.4f} {r['linf_p99']:>8.4f} {r['error_diag_ratio']:>8.4f} {r['error_offdiag_ratio']:>8.4f}")

    # Hypothesis check
    pca_r = results["rotations"].get("pca", {})
    norot_r = results["rotations"].get("norot", {})
    rand_r = results["rotations"].get("random", {})

    print(f"\nHypothesis checks:")
    if pca_r and norot_r:
        print(f"  PCA diag_ratio > NoRot: {pca_r['error_diag_ratio']:.4f} vs {norot_r['error_diag_ratio']:.4f} → {'✓' if pca_r['error_diag_ratio'] > norot_r['error_diag_ratio'] else '✗'}")
        print(f"  PCA L∞_max < NoRot: {pca_r['linf_max']:.4f} vs {norot_r['linf_max']:.4f} → {'✓' if pca_r['linf_max'] < norot_r['linf_max'] else '✗'}")
        print(f"  PCA PPL < NoRot: {pca_r['ppl']:.4f} vs {norot_r['ppl']:.4f} → {'✓' if pca_r['ppl'] < norot_r['ppl'] else '✗'}")
    if pca_r and rand_r:
        print(f"  PCA diag_ratio > Random: {pca_r['error_diag_ratio']:.4f} vs {rand_r['error_diag_ratio']:.4f} → {'✓' if pca_r['error_diag_ratio'] > rand_r['error_diag_ratio'] else '✗'}")

    out_path = out_dir / f"{short}_{bits}b_decorrelation.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
