#!/usr/bin/env python3
"""
Phase 0 v2: pos0 attention diagnostic (FIXED — uses model.attentions, not hook).
Tests: pos0 mass → Mode A (sink) vs Mode B (tail) → Lloyd+sink vs Grid prediction.

4 strategies: Grid, Grid+Sink, Lloyd, Lloyd+Sink
1 diagnostic: pos0 attention mass from model(..., output_attentions=True)

Usage:
  CUDA_VISIBLE_DEVICES=2 python exp_pos0_diagnostic_v2.py --model mistralai/Mistral-Nemo-Base-2407
  CUDA_VISIBLE_DEVICES=3 python exp_pos0_diagnostic_v2.py --model meta-llama/Llama-3.1-8B --hf-token $HF_TOKEN
"""
import argparse, json, time, gc, os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import numpy as np
import torch
from pathlib import Path

DTYPE = torch.bfloat16
CALIB_TOKENS = 2048
EVAL_CHUNK = 2048
MAX_EVAL = 50000
BITS = 2
N_SINK = 1


def uniform_quant(col, bits):
    nl = 2 ** bits
    vmin, vmax = col.min(), col.max()
    if vmax - vmin < 1e-10:
        return col.copy()
    s = (vmax - vmin) / (nl - 1)
    q = np.clip(np.round((col - vmin) / s).astype(int), 0, nl - 1)
    return q * s + vmin


def lloyd_1d(col, bits, n_iter=25):
    n_levels = 2 ** bits
    if n_levels <= 1:
        return np.array([float(col.mean())], dtype=np.float32)
    pcts = np.linspace(0, 100, n_levels + 2)[1:-1]
    centroids = np.sort(np.percentile(col, pcts)).astype(np.float64)
    for _ in range(n_iter):
        b = (centroids[:-1] + centroids[1:]) / 2
        idx = np.searchsorted(b, col)
        new_c = centroids.copy()
        for k in range(n_levels):
            m = idx == k
            if m.sum() > 0:
                new_c[k] = col[m].mean()
        if np.max(np.abs(new_c - centroids)) < 1e-6:
            break
        centroids = new_c
    return centroids.astype(np.float32)


class QuantHook:
    def __init__(self, n_kv, d_head, pca_bases, lloyd_centroids=None,
                 use_lloyd=False, use_sink=False, n_sink=1):
        self.n_kv = n_kv
        self.d_head = d_head
        self.V = pca_bases
        self.lloyd_c = lloyd_centroids
        self.use_lloyd = use_lloyd
        self.use_sink = use_sink
        self.n_sink = n_sink

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x = output.view(B, T, self.n_kv, self.d_head).float().cpu().numpy()
        out = x.copy()
        for h in range(self.n_kv):
            V = self.V[h]
            data = x[:, :, h, :]
            for b_idx in range(B):
                block = data[b_idx]
                start_t = self.n_sink if self.use_sink else 0
                if start_t >= T:
                    continue
                tokens = block[start_t:]
                rotated = tokens @ V
                quantized = np.zeros_like(rotated)
                if self.use_lloyd and self.lloyd_c and h in self.lloyd_c:
                    c = self.lloyd_c[h]
                    for j in range(self.d_head):
                        if j in c and len(c[j]) > 1:
                            bd = (c[j][:-1] + c[j][1:]) / 2
                            idx = np.searchsorted(bd, rotated[:, j])
                            quantized[:, j] = c[j][idx]
                        else:
                            quantized[:, j] = uniform_quant(rotated[:, j], BITS)
                else:
                    for j in range(self.d_head):
                        quantized[:, j] = uniform_quant(rotated[:, j], BITS)
                out[b_idx, start_t:, h, :] = quantized @ V.T
                out[b_idx, :start_t, h, :] = block[:start_t]
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


def measure_pos0_mass(model, calib_ids, n_layers):
    """Measure pos0 attention mass using model(..., output_attentions=True)."""
    # Need eager attention for output_attentions
    with torch.no_grad():
        out = model(calib_ids, output_attentions=True, use_cache=False)

    attentions = out.attentions  # tuple of (B, n_heads, T, T) per layer
    pos0_per_layer = []
    for li in range(min(n_layers, len(attentions))):
        attn = attentions[li]  # (B, n_heads, T, T)
        # pos0 mass = attention weight on position 0, averaged across heads and query positions
        pos0_mass = attn[0, :, :, 0].mean().item()  # mean across heads and queries
        pos0_per_layer.append(pos0_mass)

    return pos0_per_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="results/pos0_diagnostic_v2")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    short = args.model.split("/")[-1].replace(".", "_")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"{'='*60}")
    print(f"POS0 DIAGNOSTIC v2: {args.model}")
    print(f"{'='*60}")

    tok_kw = {"trust_remote_code": True}
    # MUST use eager attention for output_attentions
    mdl_kw = {"torch_dtype": DTYPE, "trust_remote_code": True, "attn_implementation": "eager"}
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
    print(f"  n_layers={n_layers}, n_kv={n_kv}, n_heads={n_heads}, d_head={d_head}")

    # Data
    eval_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_text = "\n\n".join([t for t in eval_ds["text"] if t.strip()])
    eval_ids = tokenizer.encode(eval_text, return_tensors="pt", truncation=False)

    calib_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_text = "\n\n".join([t for t in calib_ds["text"] if t.strip()])
    calib_ids = tokenizer.encode(calib_text, return_tensors="pt", truncation=False)[:, :CALIB_TOKENS].to(device)

    # Step 1: pos0 attention mass (FIXED: use model output, not hook)
    print("\n[Step 1] Measuring pos0 attention mass...")
    pos0_per_layer = measure_pos0_mass(model, calib_ids, n_layers)
    pos0_mean = float(np.mean(pos0_per_layer))
    pos0_max = float(np.max(pos0_per_layer))
    pos0_max_layer = int(np.argmax(pos0_per_layer))

    mode_prediction = "Mode A (sink)" if pos0_mean > 0.3 else "Mode B (tail)" if pos0_mean < 0.2 else "Intermediate"
    strategy_prediction = "lloyd_sink" if pos0_mean > 0.3 else "grid" if pos0_mean < 0.2 else "uncertain"

    print(f"  pos0 mass: mean={pos0_mean:.4f}, max={pos0_max:.4f} (layer {pos0_max_layer})")
    print(f"  Prediction: {mode_prediction} → recommend {strategy_prediction}")

    # Step 2: FP16 baseline
    print("\n[Step 2] FP16 baseline...")
    ppl_fp16 = eval_ppl(model, eval_ids, device)
    print(f"  FP16 PPL = {ppl_fp16:.4f}")

    # Step 3: Calibrate PCA + Lloyd
    print("\n[Step 3] Calibration...")
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

    pca_all, lloyd_all = {}, {}
    for li in range(n_layers):
        k_np = k_data[li].reshape(-1, n_kv, d_head)
        for hk in range(n_kv):
            K = k_np[:, hk, :].astype(np.float32)
            Kc = K - K.mean(0)
            cov = (Kc.T @ Kc) / max(K.shape[0]-1, 1) + 1e-6 * np.eye(d_head)
            _, V = np.linalg.eigh(cov)
            V = V[:, ::-1].astype(np.float32)
            pca_all[(li, hk)] = V
            K_pca = Kc @ V
            lloyd_c = {}
            for j in range(d_head):
                lloyd_c[j] = lloyd_1d(K_pca[:, j], BITS)
            lloyd_all[(li, hk)] = lloyd_c
    del k_data
    gc.collect()
    print(f"  Calibrated {n_layers}×{n_kv} heads")

    # Step 4: 4 strategies
    strategies = [
        ("grid",       False, False, "PCA + Grid (Uniform)"),
        ("grid_sink",  False, True,  "PCA + Grid + Sink"),
        ("lloyd",      True,  False, "PCA + Lloyd"),
        ("lloyd_sink", True,  True,  "PCA + Lloyd + Sink"),
    ]

    results = {
        "model": args.model, "short": short,
        "n_layers": n_layers, "n_kv": n_kv, "d_head": d_head,
        "fp16_ppl": round(ppl_fp16, 4),
        "pos0_mass_mean": round(pos0_mean, 4),
        "pos0_mass_max": round(pos0_max, 4),
        "pos0_max_layer": pos0_max_layer,
        "mode_prediction": mode_prediction,
        "strategy_prediction": strategy_prediction,
        "strategies": {},
    }

    for sname, use_lloyd, use_sink, desc in strategies:
        print(f"\n[{sname}] {desc}...")
        t0 = time.time()
        hook_handles = []
        for li in range(n_layers):
            pca_b = {hk: pca_all[(li, hk)] for hk in range(n_kv)}
            lloyd_c = {hk: lloyd_all[(li, hk)] for hk in range(n_kv)} if use_lloyd else None
            hook = QuantHook(n_kv, d_head, pca_b, lloyd_c,
                           use_lloyd=use_lloyd, use_sink=use_sink, n_sink=N_SINK)
            hook_handles.append(model.model.layers[li].self_attn.k_proj.register_forward_hook(hook))
        ppl = eval_ppl(model, eval_ids, device)
        for h in hook_handles:
            h.remove()
        elapsed = time.time() - t0
        results["strategies"][sname] = {
            "ppl": round(ppl, 4), "delta": round(ppl - ppl_fp16, 4), "runtime_sec": round(elapsed, 1)
        }
        print(f"  PPL = {ppl:.4f} (Δ={ppl-ppl_fp16:+.4f}, {elapsed:.1f}s)")

    # Oracle
    best_s = min(results["strategies"], key=lambda s: results["strategies"][s]["ppl"])
    results["oracle_best"] = best_s
    results["oracle_ppl"] = results["strategies"][best_s]["ppl"]
    results["prediction_correct"] = (strategy_prediction == best_s)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.model}")
    print(f"pos0 mass = {pos0_mean:.4f} → {mode_prediction} → predict {strategy_prediction}")
    print(f"Oracle best = {best_s} (PPL={results['oracle_ppl']:.4f})")
    print(f"Prediction {'CORRECT ✓' if results['prediction_correct'] else 'WRONG ✗'}")
    print(f"{'='*60}")

    full_path = out_dir / f"{short}_diagnostic_v2.json"
    full_path.write_text(json.dumps(results, indent=2))
    print(f"Saved: {full_path}")


if __name__ == "__main__":
    main()
