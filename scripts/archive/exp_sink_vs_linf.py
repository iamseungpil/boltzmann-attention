#!/usr/bin/env python3
"""
Sink vs L∞ separation experiment.
Disentangles two hypotheses for Lloyd-Max PPL catastrophe:

Hypothesis A (Sink): Catastrophe is caused by quantizing attention-sink tokens
  → Fix: keep first N_SINK tokens in FP16, quantize rest
Hypothesis B (L∞): Catastrophe is caused by Lloyd's large tail errors (L∞)
  → Fix: use Uniform (bounded L∞) instead of Lloyd

Configs:
  1. Baseline:     ALL tokens Uniform 2-bit (PCA)           → current v3 baseline
  2. ALL Lloyd:    ALL tokens Lloyd 2-bit (PCA)              → catastrophic baseline
  3. Sink+Lloyd:   First 4 tokens FP16, rest Lloyd 2-bit     → tests Hypothesis A
  4. NoSink+Uni:   First 4 tokens quantized, rest Uniform    → tests sink importance for Uniform
  5. Sink+Uniform: First 4 tokens FP16, rest Uniform 2-bit   → combined (should be best)

Interpretation:
  If Config 3 (Sink+Lloyd) ≈ Config 1 (all Uniform): Sink is the main cause
  If Config 3 (Sink+Lloyd) ≈ Config 2 (all Lloyd):   L∞ is the main cause
  If Config 3 is between: Both contribute

Usage:
  CUDA_VISIBLE_DEVICES=3 python exp_sink_vs_linf.py --model mistralai/Mistral-7B-v0.3
"""
import argparse, json, time, gc, os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import numpy as np
import torch
from pathlib import Path

DTYPE = torch.bfloat16
CALIB_TOKENS = 2048
EVAL_CHUNK = 2048
MAX_EVAL_TOKENS = 50000
N_SINK = 4  # number of sink tokens to protect
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


class SinkAwareHook:
    """Quantization hook that can protect sink tokens and choose quantizer per-dim."""
    def __init__(self, pca_bases, lloyd_centroids, n_kv, d_head,
                 quant_type="uniform", n_sink=0):
        self.pca_bases = pca_bases      # dict: head_idx → V matrix
        self.lloyd_centroids = lloyd_centroids  # dict: head_idx → (d_head, n_levels)
        self.n_kv = n_kv
        self.d_head = d_head
        self.quant_type = quant_type    # "uniform", "lloyd", "none"
        self.n_sink = n_sink            # first n_sink tokens kept FP16

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x = output.view(B, T, self.n_kv, self.d_head)
        x_np = x.float().cpu().numpy()
        out = np.zeros_like(x_np)

        for h in range(self.n_kv):
            V = self.pca_bases[h]
            data = x_np[:, :, h, :]  # (B, T, d)

            for b in range(B):
                for t in range(T):
                    k = data[b, t, :]  # (d,)

                    # Sink protection: keep FP16
                    if t < self.n_sink:
                        out[b, t, h, :] = k
                        continue

                    # PCA rotate
                    k_pca = k @ V  # (d,)

                    # Quantize
                    if self.quant_type == "uniform":
                        k_q = np.array([uniform_quant(np.array([k_pca[j]]), BITS)[0]
                                       for j in range(self.d_head)])
                    elif self.quant_type == "lloyd":
                        c = self.lloyd_centroids[h]
                        k_q = np.array([lloyd_quant(np.array([k_pca[j]]), c[j])[0]
                                       for j in range(self.d_head)])
                    else:  # "none"
                        k_q = k_pca

                    # PCA inverse rotate
                    out[b, t, h, :] = k_q @ V.T

        return torch.from_numpy(out).to(output.device).to(output.dtype).view(B, T, -1)


class FastSinkAwareHook:
    """Vectorized version — much faster."""
    def __init__(self, pca_bases, lloyd_centroids_per_head, n_kv, d_head,
                 quant_type="uniform", n_sink=0):
        self.V = pca_bases          # dict: h → (d, d)
        self.lloyd_c = lloyd_centroids_per_head  # dict: h → (d, n_levels)
        self.n_kv = n_kv
        self.d_head = d_head
        self.qt = quant_type
        self.n_sink = n_sink

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x = output.view(B, T, self.n_kv, self.d_head).float().cpu().numpy()
        out = x.copy()

        for h in range(self.n_kv):
            V = self.V[h]
            data = x[:, :, h, :]         # (B, T, d)
            flat = data.reshape(-1, self.d_head)  # (B*T, d)

            # PCA rotate all tokens
            rotated = flat @ V            # (B*T, d)

            # Quantize
            if self.qt == "uniform":
                quantized = np.zeros_like(rotated)
                for j in range(self.d_head):
                    quantized[:, j] = uniform_quant(rotated[:, j], BITS)
            elif self.qt == "lloyd":
                quantized = np.zeros_like(rotated)
                c = self.lloyd_c[h]
                for j in range(self.d_head):
                    boundaries = (c[j, :-1] + c[j, 1:]) / 2
                    idx = np.searchsorted(boundaries, rotated[:, j])
                    quantized[:, j] = c[j, idx]
            else:
                quantized = rotated

            # Inverse PCA
            reconstructed = quantized @ V.T
            reconstructed = reconstructed.reshape(B, T, self.d_head)

            # Restore sink tokens (overwrite with original)
            if self.n_sink > 0:
                reconstructed[:, :self.n_sink, :] = data[:, :self.n_sink, :]

            out[:, :, h, :] = reconstructed

        return torch.from_numpy(out).to(output.device).to(output.dtype).view(B, T, -1)


def eval_ppl(model, eval_ids, device, chunk_len=EVAL_CHUNK, max_tokens=MAX_EVAL_TOKENS):
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
    parser.add_argument("--output-dir", default="results/sink_vs_linf")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    short = args.model.split("/")[-1].replace(".", "_")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"{'='*60}")
    print(f"SINK vs L∞ SEPARATION: {args.model}")
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
    print(f"  n_layers={n_layers}, n_kv={n_kv}, d_head={d_head}")

    # Load eval data (TEST split, full)
    eval_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_text = "\n\n".join([t for t in eval_ds["text"] if t.strip()])
    eval_ids = tokenizer.encode(eval_text, return_tensors="pt", truncation=False)
    print(f"  Eval: {eval_ids.shape[1]} tokens, using {min(eval_ids.shape[1], MAX_EVAL_TOKENS)}")

    # Calibration data (TRAIN split)
    calib_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_text = "\n\n".join([t for t in calib_ds["text"] if t.strip()])
    calib_all = tokenizer.encode(calib_text, return_tensors="pt", truncation=False)
    calib_ids = calib_all[:, :CALIB_TOKENS].to(device)

    # Stage 1: FP16 baseline
    print("\n[FP16 baseline]")
    ppl_fp16, nc, ns = eval_ppl(model, eval_ids, device)
    print(f"  FP16 PPL = {ppl_fp16:.4f} ({nc} chunks, {ns} tokens)")

    # Stage 2: Calibrate PCA + Lloyd per layer per head
    print("\n[Calibration]")
    k_data = {}
    hooks = []
    def kh(li):
        def fn(mod, inp, out):
            k_data[li] = out.detach().cpu().float().numpy()
        return fn
    for li in range(n_layers):
        hooks.append(model.model.layers[li].self_attn.k_proj.register_forward_hook(kh(li)))
    with torch.no_grad():
        model(calib_ids, use_cache=False)
    for h in hooks:
        h.remove()

    pca_bases = {}   # (layer, head) → V
    lloyd_c = {}     # (layer, head) → (d, n_levels)
    for li in range(n_layers):
        k_np = k_data[li].reshape(-1, n_kv, d_head)
        for hk in range(n_kv):
            K = k_np[:, hk, :].astype(np.float32)
            K_mean = K.mean(0)
            K_c = K - K_mean
            cov = (K_c.T @ K_c) / max(K.shape[0]-1, 1)
            _, V = np.linalg.eigh(cov)
            V = V[:, ::-1].astype(np.float32)  # descending eigenvalue order
            pca_bases[(li, hk)] = V

            # Fit Lloyd on PCA'd data
            K_pca = K_c @ V
            centroids = np.zeros((d_head, 2**BITS), dtype=np.float32)
            for j in range(d_head):
                centroids[j] = lloyd_max_1d(K_pca[:, j], BITS).astype(np.float32)
            lloyd_c[(li, hk)] = centroids

    del k_data
    gc.collect()
    print(f"  Calibrated {n_layers}×{n_kv} heads")

    # Stage 3: Run configs
    configs = [
        ("1_all_uniform",    "uniform", 0,      "ALL tokens Uniform 2-bit"),
        ("2_all_lloyd",      "lloyd",   0,      "ALL tokens Lloyd 2-bit"),
        ("3_sink4_lloyd",    "lloyd",   N_SINK, "Sink 4 FP16 + rest Lloyd"),
        ("4_nosink_uniform", "uniform", 0,      "ALL tokens Uniform (no sink protection)"),
        ("5_sink4_uniform",  "uniform", N_SINK, "Sink 4 FP16 + rest Uniform"),
    ]

    results = {"model": args.model, "fp16_ppl": round(ppl_fp16, 4), "n_sink": N_SINK, "configs": {}}

    for cfg_name, quant_type, n_sink, description in configs:
        print(f"\n[{cfg_name}] {description}")
        t0 = time.time()

        # Install hooks
        hook_handles = []
        for li in range(n_layers):
            pca_b = {hk: pca_bases[(li, hk)] for hk in range(n_kv)}
            lloyd_centroids = {hk: lloyd_c[(li, hk)] for hk in range(n_kv)}
            hook = FastSinkAwareHook(pca_b, lloyd_centroids, n_kv, d_head,
                                     quant_type=quant_type, n_sink=n_sink)
            hook_handles.append(
                model.model.layers[li].self_attn.k_proj.register_forward_hook(hook))

        ppl, nc, ns = eval_ppl(model, eval_ids, device)
        for h in hook_handles:
            h.remove()
        elapsed = time.time() - t0

        results["configs"][cfg_name] = {
            "ppl": round(ppl, 4),
            "quant_type": quant_type,
            "n_sink": n_sink,
            "description": description,
            "runtime_sec": round(elapsed, 1),
        }
        print(f"  PPL = {ppl:.4f} ({elapsed:.1f}s)")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.model}")
    print(f"{'='*60}")
    print(f"FP16:                    {ppl_fp16:.4f}")
    for cfg_name, cfg_data in sorted(results["configs"].items()):
        delta = (cfg_data["ppl"] / ppl_fp16 - 1) * 100
        print(f"  {cfg_name:25s} PPL={cfg_data['ppl']:.4f}  ({delta:+.1f}% vs FP16)")

    print(f"\n--- INTERPRETATION ---")
    c1 = results["configs"]["1_all_uniform"]["ppl"]
    c2 = results["configs"]["2_all_lloyd"]["ppl"]
    c3 = results["configs"]["3_sink4_lloyd"]["ppl"]
    c5 = results["configs"]["5_sink4_uniform"]["ppl"]

    lloyd_gap = c2 - c1
    sink_fix = c2 - c3
    sink_pct = sink_fix / lloyd_gap * 100 if lloyd_gap > 0 else 0

    print(f"Lloyd catastrophe gap:     {lloyd_gap:.4f} (Config2 - Config1)")
    print(f"Sink protection fixes:     {sink_fix:.4f} ({sink_pct:.1f}% of gap)")
    print(f"Remaining (L∞ component):  {c3 - c1:.4f} ({100-sink_pct:.1f}% of gap)")

    if sink_pct > 70:
        print("→ SINK is the PRIMARY cause of Lloyd catastrophe")
    elif sink_pct < 30:
        print("→ L∞ tail error is the PRIMARY cause")
    else:
        print("→ BOTH contribute significantly")

    # Save
    out_path = out_dir / f"{short}_sink_vs_linf.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
