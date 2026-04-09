#!/usr/bin/env python3
"""
CORRECTED Lloyd vs Uniform comparison.
Fixes mean-centering bug that inflated Lloyd PPL catastrophically.

Bug was: eval hook applied PCA rotation WITHOUT subtracting calibration mean.
Fix: subtract K_mean before PCA, add K_mean after inverse PCA (matching coworker's code).

Also fixes Uniform fairness: uses calibration min/max (not per-chunk adaptive).

Intent: Determine whether Lloyd (MSE-optimal) or Uniform is truly better for PPL.
Hypothesis: With correct mean-centering, Lloyd should be competitive with or better than Uniform.
Verification: PPL comparison on 49K test, 3 models × 3 bit-widths.

Evaluation manifest:
  Dataset: wikitext-2-raw-v1, test split
  Eval: 49K tokens, 2048-token non-overlapping chunks
  Calibration: train split, first 2048 tokens
  PPL = exp(total_NLL / total_scored_tokens)
  All stats (K_mean, PCA basis, Lloyd centroids, Uniform range) fixed at calibration time.

Usage:
  CUDA_VISIBLE_DEVICES=0 python exp_correct_lloyd_vs_uniform.py --model mistralai/Mistral-7B-v0.3
"""
import argparse, json, time, gc, os, hashlib
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import numpy as np
import torch
from pathlib import Path

DTYPE = torch.bfloat16
CALIB_TOKENS = 2048
EVAL_CHUNK = 2048
MAX_EVAL = 50000


def lloyd_1d(col, bits, n_iter=30):
    """Lloyd-Max scalar quantizer for 1D data."""
    n_levels = 2 ** bits
    if n_levels <= 1:
        return np.array([col.mean()], dtype=np.float32)
    pcts = np.linspace(0, 100, n_levels + 2)[1:-1]
    centroids = np.sort(np.percentile(col, pcts)).astype(np.float64)
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
    return centroids.astype(np.float32)


def lloyd_quantize(col, centroids):
    """Quantize using pre-fitted Lloyd centroids."""
    if len(centroids) <= 1:
        return np.full_like(col, centroids[0])
    boundaries = (centroids[:-1] + centroids[1:]) / 2
    idx = np.searchsorted(boundaries, col)
    return centroids[idx]


def uniform_quantize_fixed(col, vmin, vmax, bits):
    """Uniform quantizer with FIXED range from calibration (not per-chunk)."""
    nl = 2 ** bits
    col_clamped = np.clip(col, vmin, vmax)
    if vmax - vmin < 1e-10:
        return col_clamped
    s = (vmax - vmin) / (nl - 1)
    q = np.clip(np.round((col_clamped - vmin) / s).astype(int), 0, nl - 1)
    return q * s + vmin


def uniform_quantize_adaptive(col, bits):
    """Uniform quantizer with per-chunk adaptive range (BUGGY baseline)."""
    nl = 2 ** bits
    vmin, vmax = col.min(), col.max()
    if vmax - vmin < 1e-10:
        return col.copy()
    s = (vmax - vmin) / (nl - 1)
    q = np.clip(np.round((col - vmin) / s).astype(int), 0, nl - 1)
    return q * s + vmin


class CorrectQuantHook:
    """
    Quantization hook with CORRECT mean centering.

    Pipeline:
      1. Subtract calibration K_mean from eval keys
      2. Apply PCA rotation (V)
      3. Quantize in PCA space (Lloyd or Uniform with calibration stats)
      4. Inverse PCA rotation
      5. Add back K_mean
    """
    def __init__(self, n_kv, d_head, K_means, V_bases,
                 quantizer_type, bits,
                 lloyd_centroids=None,
                 uniform_ranges=None,
                 use_sink=False, n_sink=1):
        self.n_kv = n_kv
        self.d_head = d_head
        self.K_means = K_means          # dict h → (d_head,) float32
        self.V_bases = V_bases          # dict h → (d_head, d_head) float32
        self.qtype = quantizer_type     # "lloyd", "uniform_fixed", "uniform_adaptive"
        self.bits = bits
        self.lloyd_c = lloyd_centroids  # dict h → (d_head, n_levels) float32
        self.uni_ranges = uniform_ranges  # dict h → (d_head, 2) [min, max]
        self.use_sink = use_sink
        self.n_sink = n_sink

        # Track eval-time L∞ for diagnostics
        self.max_errors = []

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x = output.view(B, T, self.n_kv, self.d_head).float().cpu().numpy()
        out = x.copy()

        for h in range(self.n_kv):
            K_mean = self.K_means[h]
            V = self.V_bases[h]
            data = x[:, :, h, :]  # (B, T, d)

            for b_idx in range(B):
                block = data[b_idx]  # (T, d)
                start_t = self.n_sink if self.use_sink else 0

                # Sink tokens: keep as-is
                out[b_idx, :start_t, h, :] = block[:start_t]

                if start_t >= T:
                    continue

                tokens = block[start_t:]  # (T-start, d)

                # STEP 1: Subtract calibration mean (THE FIX)
                centered = tokens - K_mean

                # STEP 2: PCA rotation
                rotated = centered @ V

                # STEP 3: Quantize
                quantized = np.zeros_like(rotated)
                for j in range(self.d_head):
                    col = rotated[:, j]
                    if self.qtype == "lloyd":
                        c = self.lloyd_c[h][j]
                        quantized[:, j] = lloyd_quantize(col, c)
                    elif self.qtype == "uniform_fixed":
                        vmin, vmax = self.uni_ranges[h][j]
                        quantized[:, j] = uniform_quantize_fixed(col, vmin, vmax, self.bits)
                    elif self.qtype == "uniform_adaptive":
                        quantized[:, j] = uniform_quantize_adaptive(col, self.bits)

                # Track L∞ for diagnostics
                err = np.abs(rotated - quantized)
                self.max_errors.append(float(err.max()))

                # STEP 4: Inverse PCA
                reconstructed = quantized @ V.T

                # STEP 5: Add back mean (THE FIX)
                out[b_idx, start_t:, h, :] = reconstructed + K_mean

        return torch.from_numpy(out).to(output.device).to(output.dtype).view(B, T, -1)


def eval_ppl(model, eval_ids, device, chunk_len=EVAL_CHUNK, max_tokens=MAX_EVAL):
    model.eval()
    total_len = min(eval_ids.shape[1], max_tokens)
    n_chunks = total_len // chunk_len
    total_nll, total_scored = 0.0, 0
    for i in range(n_chunks):
        chunk = eval_ids[:, i*chunk_len:(i+1)*chunk_len].to(device)
        with torch.no_grad():
            out = model(chunk, labels=chunk, use_cache=False)
        scored = chunk_len - 1
        total_nll += out.loss.item() * scored
        total_scored += scored
    return float(np.exp(total_nll / total_scored)), n_chunks, total_scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--output-dir", default="results/correct_lloyd")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    short = args.model.split("/")[-1].replace(".", "_")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"{'='*60}")
    print(f"CORRECTED Lloyd vs Uniform: {args.model}")
    print(f"Bits: {args.bits}")
    print(f"Protocol: 49K test, calibration-fixed stats, mean-centering FIXED")
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
    print(f"  n_layers={n_layers}, n_kv={n_kv}, d_head={d_head}")

    # Load data
    eval_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_text = "\n\n".join([t for t in eval_ds["text"] if t.strip()])
    eval_ids = tokenizer.encode(eval_text, return_tensors="pt", truncation=False)
    eval_tokens = min(eval_ids.shape[1], MAX_EVAL)
    print(f"  Eval: {eval_ids.shape[1]} tokens → {eval_tokens} used")

    calib_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_text = "\n\n".join([t for t in calib_ds["text"] if t.strip()])
    calib_ids = tokenizer.encode(calib_text, return_tensors="pt", truncation=False)[:, :CALIB_TOKENS].to(device)

    # Verify no data overlap
    calib_hash = hashlib.md5(calib_ids.cpu().numpy().tobytes()).hexdigest()[:8]
    eval_hash = hashlib.md5(eval_ids[:, :CALIB_TOKENS].numpy().tobytes()).hexdigest()[:8]
    assert calib_hash != eval_hash, "FATAL: calibration and eval data overlap!"
    print(f"  Data: calib_hash={calib_hash}, eval_hash={eval_hash} (no overlap)")

    # FP16 baseline
    print("\n[FP16 baseline]")
    ppl_fp16, nc, ns = eval_ppl(model, eval_ids, device)
    print(f"  PPL = {ppl_fp16:.4f} ({nc} chunks, {ns} tokens)")

    # Calibration: extract keys
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

    # Compute PCA bases, means, and quantizer stats per bit-width
    calib_stats = {}  # (li, hk) → {K_mean, V, lloyd_centroids[bits], uni_ranges[bits]}
    for li in range(n_layers):
        k_np = k_data[li].reshape(-1, n_kv, d_head)
        for hk in range(n_kv):
            K = k_np[:, hk, :].astype(np.float32)
            K_mean = K.mean(0)
            K_c = K - K_mean
            cov = (K_c.T @ K_c) / max(K.shape[0] - 1, 1) + 1e-6 * np.eye(d_head)
            ev, V = np.linalg.eigh(cov)
            V = V[:, ::-1].astype(np.float32)  # descending eigenvalue order

            K_pca = K_c @ V  # mean-subtracted, PCA-rotated calibration data

            stats = {"K_mean": K_mean, "V": V, "lloyd": {}, "uni_range": {}}
            for bits in args.bits:
                # Lloyd centroids per dim
                lloyd_c = np.zeros((d_head, 2**bits), dtype=np.float32)
                for j in range(d_head):
                    lloyd_c[j] = lloyd_1d(K_pca[:, j], bits)
                stats["lloyd"][bits] = lloyd_c

                # Uniform range per dim (from calibration, NOT per-chunk)
                uni_range = np.zeros((d_head, 2), dtype=np.float32)
                for j in range(d_head):
                    uni_range[j, 0] = K_pca[:, j].min()
                    uni_range[j, 1] = K_pca[:, j].max()
                stats["uni_range"][bits] = uni_range

            calib_stats[(li, hk)] = stats

    del k_data
    gc.collect()
    print(f"  Calibrated {n_layers}×{n_kv} heads")

    # Configs
    configs = [
        ("uniform_fixed", "Uniform (calibration range)"),
        ("uniform_adaptive", "Uniform (per-chunk, BUGGY baseline)"),
        ("lloyd", "Lloyd (CORRECTED mean-centering)"),
    ]

    all_results = {"model": args.model, "fp16": round(ppl_fp16, 4), "bits_results": {}}

    for bits in args.bits:
        print(f"\n{'='*40}")
        print(f"  {bits}-bit experiments")
        print(f"{'='*40}")

        bit_results = {}

        for qtype, desc in configs:
            cfg_name = f"{qtype}_{bits}b"
            out_path = out_dir / f"{short}_{cfg_name}.json"

            if out_path.exists():
                cached = json.loads(out_path.read_text())
                print(f"\n  [{cfg_name}] {desc} → PPL={cached['ppl']} (cached)")
                bit_results[qtype] = cached
                continue

            print(f"\n  [{cfg_name}] {desc}...")
            t0 = time.time()

            hook_handles = []
            all_hook_objects = []
            for li in range(n_layers):
                K_means = {hk: calib_stats[(li, hk)]["K_mean"] for hk in range(n_kv)}
                V_bases = {hk: calib_stats[(li, hk)]["V"] for hk in range(n_kv)}

                lloyd_c = None
                uni_ranges = None
                if qtype == "lloyd":
                    lloyd_c = {}
                    for hk in range(n_kv):
                        lloyd_c[hk] = {}
                        c = calib_stats[(li, hk)]["lloyd"][bits]
                        for j in range(d_head):
                            lloyd_c[hk][j] = c[j]
                elif qtype == "uniform_fixed":
                    uni_ranges = {hk: calib_stats[(li, hk)]["uni_range"][bits] for hk in range(n_kv)}

                hook_obj = CorrectQuantHook(
                    n_kv, d_head, K_means, V_bases,
                    quantizer_type=qtype, bits=bits,
                    lloyd_centroids=lloyd_c,
                    uniform_ranges=uni_ranges,
                )
                all_hook_objects.append(hook_obj)
                hook_handles.append(
                    model.model.layers[li].self_attn.k_proj.register_forward_hook(hook_obj))

            ppl, nc, ns = eval_ppl(model, eval_ids, device)
            for h in hook_handles:
                h.remove()
            elapsed = time.time() - t0

            # Collect L∞ stats
            all_linf = []
            for ho in all_hook_objects:
                all_linf.extend(ho.max_errors)
            avg_linf = float(np.mean(all_linf)) if all_linf else 0
            max_linf = float(np.max(all_linf)) if all_linf else 0

            delta = ppl - ppl_fp16
            result = {
                "ppl": round(ppl, 4),
                "delta": round(delta, 4),
                "bits": bits,
                "quantizer": qtype,
                "desc": desc,
                "eval_linf_avg": round(avg_linf, 6),
                "eval_linf_max": round(max_linf, 6),
                "runtime_sec": round(elapsed, 1),
            }
            out_path.write_text(json.dumps(result, indent=2))
            bit_results[qtype] = result
            print(f"    PPL = {ppl:.4f} (Δ={delta:+.4f}, L∞_avg={avg_linf:.4f}, {elapsed:.1f}s)")

        all_results["bits_results"][str(bits)] = bit_results

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.model}")
    print(f"{'='*60}")
    print(f"FP16: {ppl_fp16:.4f}")
    print(f"{'Bits':<6} {'Uniform(fixed)':<18} {'Uniform(adaptive)':<20} {'Lloyd(corrected)':<18} {'Lloyd wins?'}")
    for bits in args.bits:
        br = all_results["bits_results"][str(bits)]
        uf = br.get("uniform_fixed", {}).get("ppl", "—")
        ua = br.get("uniform_adaptive", {}).get("ppl", "—")
        ll = br.get("lloyd", {}).get("ppl", "—")
        wins = "✓" if isinstance(ll, float) and isinstance(uf, float) and ll < uf else "✗" if isinstance(ll, float) and isinstance(uf, float) else "?"
        print(f"{bits}b     {uf:<18} {ua:<20} {ll:<18} {wins}")

    # Save full
    full_path = out_dir / f"{short}_full.json"
    full_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {full_path}")


if __name__ == "__main__":
    main()
