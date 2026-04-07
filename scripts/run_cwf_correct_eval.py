#!/usr/bin/env python3
"""
Protocol-locked CWF re-evaluation.
Fixes the develop branch bug: 2K train → 49K test.

EVALUATION MANIFEST (frozen before execution):
  - Dataset: wikitext, wikitext-2-raw-v1
  - Eval split: TEST (never train)
  - Calibration split: TRAIN (first 1024 tokens, offset by seed)
  - Tokenizer: model default, trust_remote_code=True
  - Chunk length: 2048 tokens, non-overlapping
  - Max eval tokens: min(total_test_tokens, 50000)
  - Scored tokens per chunk: chunk_len - 1 (exclude first token from loss)
  - PPL = exp(total_NLL / total_scored_tokens)  [NOT mean of per-chunk PPLs]
  - BOS: not added (tokenizer default)
  - Padding: none (no batching, sequential chunks)
  - Dtype: bfloat16
  - model.eval(), torch.no_grad()
  - Fresh calibration per seed (no reuse from develop branch)

STAGES:
  Stage 1: Calibration (TRAIN split only) → PCA bases, Lloyd centroids, Fisher traces
  Stage 2: Per-layer sensitivity (TEST split, one-layer-at-a-time substitution)
  Stage 3: CWF allocation + final PPL (TEST split, full eval)

Usage:
  CUDA_VISIBLE_DEVICES=3 python run_cwf_correct_eval.py --model mistralai/Mistral-7B-v0.3
  CUDA_VISIBLE_DEVICES=3 python run_cwf_correct_eval.py --model Qwen/Qwen2.5-7B
"""
import argparse, json, time, gc, os, sys, hashlib
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import numpy as np
import torch
from pathlib import Path

# ── FROZEN PROTOCOL CONSTANTS ──
EVAL_SPLIT = "test"
CALIB_SPLIT = "train"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
CHUNK_LEN = 2048
MAX_EVAL_TOKENS = 50000
CALIB_TOKENS = 1024
DTYPE = torch.bfloat16
B_FLOOR = 2
B_MAX = 6
CWF_BUDGETS = [2.0, 2.5, 3.0, 3.5]
SEED = 42


def lloyd_max_1d(col, bits, n_iter=30):
    """Gaussian Lloyd-Max scalar quantizer for one dimension."""
    n_levels = 2 ** bits
    if n_levels < 2:
        return np.array([col.mean()])
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


def water_filling_global(importance, total_budget, b_floor=2, b_max=6):
    """Greedy bit allocation: assign +1 bit to highest-marginal-gain head."""
    n = len(importance)
    imp = np.maximum(np.array(importance, dtype=np.float64), 1e-12)
    bits = np.full(n, b_floor, dtype=int)
    spent = n * b_floor
    assert spent <= total_budget, f"Budget {total_budget} < floor requirement {spent}"
    while spent < total_budget:
        valid = bits < b_max
        if not valid.any():
            break
        gains = np.where(valid, imp * (4.0**(-bits) - 4.0**(-(bits+1))), -np.inf)
        j = int(np.argmax(gains))
        bits[j] += 1
        spent += 1
    assert bits.sum() == total_budget, f"Budget mismatch: {bits.sum()} != {total_budget}"
    return bits


def fit_pca_lloyd(K, bits):
    """PCA rotation + Lloyd-Max per dimension. Returns quantizer dict."""
    K = K.astype(np.float32)
    K_mean = K.mean(axis=0)
    K_c = K - K_mean
    cov = (K_c.T @ K_c) / max(K.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    V = eigvecs[:, order]
    K_pca = K_c @ V
    d = K.shape[1]
    centroids = np.zeros((d, 2**bits), dtype=np.float32)
    for j in range(d):
        centroids[j] = lloyd_max_1d(K_pca[:, j], bits).astype(np.float32)
    return {'K_mean': K_mean, 'V': V.astype(np.float32), 'centroids': centroids, 'bits': bits}


class QuantHook:
    """Forward hook: PCA rotate → Lloyd quantize → inverse rotate."""
    def __init__(self, head_quantizers, n_kv, d_head):
        self.hq = head_quantizers
        self.n_kv = n_kv
        self.d_head = d_head

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x = output.view(B, T, self.n_kv, self.d_head).float().cpu().numpy()
        out = np.zeros_like(x)
        for h in range(self.n_kv):
            q = self.hq[h]
            data = x[:, :, h, :].reshape(-1, self.d_head).astype(np.float32)
            K_c = data - q['K_mean']
            K_pca = K_c @ q['V']
            K_q = np.zeros_like(K_pca)
            c = q['centroids']
            for j in range(self.d_head):
                bd = (c[j, :-1] + c[j, 1:]) / 2
                idx = np.searchsorted(bd, K_pca[:, j])
                K_q[:, j] = c[j, idx]
            out[:, :, h, :] = (K_q @ q['V'].T + q['K_mean']).reshape(x[:, :, h, :].shape)
        return torch.from_numpy(out).to(output.device).to(output.dtype).view(B, T, -1)


def uniform_quant_hook(n_kv, d_head, bits):
    """Simple per-channel min-max uniform quantization hook."""
    def hook_fn(module, inputs, output):
        B, T, _ = output.shape
        x = output.view(B, T, n_kv, d_head).float().cpu().numpy()
        out = np.zeros_like(x)
        for h in range(n_kv):
            for j in range(d_head):
                col = x[:, :, h, j].flatten()
                vmin, vmax = col.min(), col.max()
                if vmax - vmin < 1e-10:
                    out[:, :, h, j] = x[:, :, h, j]
                    continue
                nl = 2**bits
                s = (vmax - vmin) / (nl - 1)
                q = np.clip(np.round((col - vmin) / s).astype(int), 0, nl-1)
                out[:, :, h, j] = (q * s + vmin).reshape(x[:, :, h, j].shape)
        return torch.from_numpy(out).to(output.device).to(output.dtype).view(B, T, -1)
    return hook_fn


def pca_uniform_hook(pca_bases, n_kv, d_head, bits):
    """PCA rotation + per-channel uniform quantization."""
    def hook_fn(module, inputs, output):
        B, T, _ = output.shape
        x = output.view(B, T, n_kv, d_head).float().cpu().numpy()
        out = np.zeros_like(x)
        for h in range(n_kv):
            V = pca_bases[h]
            data = x[:, :, h, :].reshape(-1, d_head)
            Kr = data @ V
            for j in range(d_head):
                col = Kr[:, j]
                vmin, vmax = col.min(), col.max()
                if vmax - vmin < 1e-10:
                    continue
                nl = 2**bits
                s = (vmax - vmin) / (nl - 1)
                q = np.clip(np.round((col - vmin) / s).astype(int), 0, nl-1)
                Kr[:, j] = q * s + vmin
            out[:, :, h, :] = (Kr @ V.T).reshape(x[:, :, h, :].shape)
        return torch.from_numpy(out).to(output.device).to(output.dtype).view(B, T, -1)
    return hook_fn


def eval_ppl(model, eval_ids, device, chunk_len=CHUNK_LEN, max_tokens=MAX_EVAL_TOKENS):
    """Evaluate PPL: total_NLL / total_scored_tokens, then exp."""
    model.eval()
    total_len = min(eval_ids.shape[1], max_tokens)
    n_chunks = total_len // chunk_len
    total_nll, total_scored = 0.0, 0
    for i in range(n_chunks):
        chunk = eval_ids[:, i*chunk_len:(i+1)*chunk_len].to(device)
        with torch.no_grad():
            out = model(chunk, labels=chunk, use_cache=False)
        scored = chunk_len - 1  # first token has no prediction target
        total_nll += out.loss.item() * scored
        total_scored += scored
    ppl = float(np.exp(total_nll / total_scored))
    return ppl, n_chunks, total_scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="results/cwf_corrected")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="Skip Phase 2 if sensitivity JSON already exists")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    short = args.model.split("/")[-1].replace(".", "_")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Log manifest ──
    manifest = {
        "eval_split": EVAL_SPLIT, "calib_split": CALIB_SPLIT,
        "dataset": f"{DATASET_NAME}/{DATASET_CONFIG}",
        "chunk_len": CHUNK_LEN, "max_eval_tokens": MAX_EVAL_TOKENS,
        "calib_tokens": CALIB_TOKENS, "dtype": str(DTYPE),
        "seed": args.seed, "b_floor": B_FLOOR, "b_max": B_MAX,
        "cwf_budgets": CWF_BUDGETS, "model": args.model,
        "script": "run_cwf_correct_eval.py",
        "ppl_formula": "exp(total_NLL / total_scored_tokens)",
    }
    print(f"{'='*60}")
    print(f"PROTOCOL-LOCKED CWF EVALUATION")
    print(f"Model: {args.model}")
    print(f"Eval: {EVAL_SPLIT} split, {MAX_EVAL_TOKENS} max tokens, chunk={CHUNK_LEN}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}")

    # ── Load model ──
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

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
    print(f"  n_layers={n_layers}, n_kv={n_kv}, d_head={d_head}, GQA G={G}")

    # ── Load data (strict split separation) ──
    eval_ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=EVAL_SPLIT)
    eval_text = "\n\n".join([t for t in eval_ds["text"] if t.strip()])
    eval_ids = tokenizer.encode(eval_text, return_tensors="pt", truncation=False)
    eval_tokens = min(eval_ids.shape[1], MAX_EVAL_TOKENS)
    n_eval_chunks = eval_tokens // CHUNK_LEN
    print(f"  Eval: {eval_ids.shape[1]} total tokens, using {eval_tokens} ({n_eval_chunks} chunks)")

    calib_ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=CALIB_SPLIT)
    calib_text = "\n\n".join([t for t in calib_ds["text"] if t.strip()])
    calib_all = tokenizer.encode(calib_text, return_tensors="pt", truncation=False)
    offset = np.random.randint(0, max(1, calib_all.shape[1] - CALIB_TOKENS))
    calib_ids = calib_all[:, offset:offset+CALIB_TOKENS].to(device)
    print(f"  Calib: {CALIB_TOKENS} tokens from train (offset={offset})")

    # Verify no overlap
    calib_hash = hashlib.md5(calib_ids.cpu().numpy().tobytes()).hexdigest()[:8]
    eval_hash = hashlib.md5(eval_ids[:, :CALIB_TOKENS].numpy().tobytes()).hexdigest()[:8]
    assert calib_hash != eval_hash, "FATAL: calibration and eval data overlap!"
    print(f"  Data integrity: calib_hash={calib_hash}, eval_hash={eval_hash} (no overlap)")

    manifest["eval_total_tokens"] = int(eval_ids.shape[1])
    manifest["eval_used_tokens"] = eval_tokens
    manifest["eval_chunks"] = n_eval_chunks
    manifest["calib_offset"] = offset
    manifest["calib_hash"] = calib_hash

    layers = model.model.layers
    results = {"manifest": manifest, "configs": {}}

    # ── Stage 0: FP16 baseline ──
    print("\n[Stage 0] FP16 baseline...")
    ppl_fp16, nc, ns = eval_ppl(model, eval_ids, device)
    print(f"  FP16 PPL = {ppl_fp16:.4f} ({nc} chunks, {ns} scored tokens)")
    results["fp16"] = {"ppl": round(ppl_fp16, 4), "chunks": nc, "scored_tokens": ns}

    # ── Stage 1: Calibration (TRAIN only) ──
    print("\n[Stage 1] Calibration (PCA + Fisher)...")
    k_data, q_data = {}, {}
    hooks = []
    def make_calib_hook(li):
        def fn(mod, inp, out):
            k_data[li] = out.detach().cpu().float().numpy()
        return fn
    def make_q_hook(li):
        def fn(mod, inp, out):
            q_data[li] = out.detach().cpu().float().numpy()
        return fn
    for li in range(n_layers):
        hooks.append(layers[li].self_attn.k_proj.register_forward_hook(make_calib_hook(li)))
        hooks.append(layers[li].self_attn.q_proj.register_forward_hook(make_q_hook(li)))
    with torch.no_grad():
        model(calib_ids, use_cache=False)
    for h in hooks:
        h.remove()

    pca_bases_all = {}  # (layer, head) → V matrix
    fisher_traces = np.zeros((n_layers, n_kv), dtype=np.float64)
    for li in range(n_layers):
        k_np = k_data[li].reshape(-1, n_kv, d_head)
        q_np = q_data[li].reshape(-1, n_heads, d_head)
        for hk in range(n_kv):
            K = k_np[:, hk, :]
            Kc = K - K.mean(0)
            cov = (Kc.T @ Kc) / max(K.shape[0]-1, 1) + 1e-6 * np.eye(d_head)
            _, V = np.linalg.eigh(cov)
            pca_bases_all[(li, hk)] = V.astype(np.float32)
            q_group = q_np[:, hk*G:(hk+1)*G, :].mean(axis=1)
            fisher_traces[li, hk] = max(np.sum(q_group**2) / q_group.shape[0], 1e-10)
    print(f"  Calibrated {n_layers}×{n_kv} = {n_layers*n_kv} heads")

    del k_data, q_data
    gc.collect()

    # ── Stage 2: Per-layer sensitivity (TEST split, Exp4-style) ──
    sens_path = out_dir / f"{short}_sensitivity_s{args.seed}.json"
    if sens_path.exists() and args.skip_sensitivity:
        sens = json.loads(sens_path.read_text())
        delta_ppl = {int(k): v for k, v in sens["delta_ppl"].items()}
        print(f"\n[Stage 2] Sensitivity loaded from cache ({len(delta_ppl)} layers)")
    else:
        print(f"\n[Stage 2] Per-layer sensitivity ({n_layers} layers)...")
        delta_ppl = {}
        # Fit Lloyd-2bit for each head from calibration
        lloyd_fits = {}
        for li in range(n_layers):
            # Re-extract K from calibration for Lloyd fitting
            kh = []
            def calib_k(mod, inp, out):
                kh.append(out.detach().cpu().float().numpy())
            handle = layers[li].self_attn.k_proj.register_forward_hook(calib_k)
            with torch.no_grad():
                model(calib_ids, use_cache=False)
            handle.remove()
            k_np = kh[0].reshape(-1, n_kv, d_head)
            for hk in range(n_kv):
                lloyd_fits[(li, hk)] = fit_pca_lloyd(k_np[:, hk, :], bits=2)

        for li in range(n_layers):
            hq = {hk: lloyd_fits[(li, hk)] for hk in range(n_kv)}
            hook_obj = QuantHook(hq, n_kv, d_head)
            handle = layers[li].self_attn.k_proj.register_forward_hook(hook_obj)
            ppl_li, _, _ = eval_ppl(model, eval_ids, device)
            handle.remove()
            delta_ppl[li] = round(ppl_li - ppl_fp16, 6)
            print(f"  Layer {li:2d}: ΔPPL={delta_ppl[li]:+.4f}")

        sens_data = {"model": args.model, "seed": args.seed, "ppl_fp16": ppl_fp16,
                     "delta_ppl": delta_ppl, "protocol": "49K_test"}
        sens_path.write_text(json.dumps(sens_data, indent=2))

    # ── Stage 3: CWF + baselines, all on TEST ──
    print(f"\n[Stage 3] CWF sweep + baselines...")

    # Compute CWF importance
    importance = np.zeros(n_layers * n_kv, dtype=np.float64)
    for li in range(n_layers):
        sens = max(delta_ppl.get(li, 0.001), 0.001)
        for hk in range(n_kv):
            importance[li * n_kv + hk] = sens * fisher_traces[li, hk]

    # Re-extract calibration K for Lloyd fitting at various bits
    print("  Re-extracting calibration K for all layers...")
    all_calib_K = {}
    for li in range(n_layers):
        kh = []
        def grab_k(mod, inp, out, _kh=kh):
            _kh.append(out.detach().cpu().float().numpy())
        handle = layers[li].self_attn.k_proj.register_forward_hook(grab_k)
        with torch.no_grad():
            model(calib_ids, use_cache=False)
        handle.remove()
        k_np = kh[0].reshape(-1, n_kv, d_head)
        for hk in range(n_kv):
            all_calib_K[(li, hk)] = k_np[:, hk, :]

    configs_to_run = []

    # Baseline: v3 PCA+Uniform at 2-bit
    configs_to_run.append(("v3_pca_uniform_2b", "pca_uniform", 2, None))

    # Baseline: v3 PCA+WF(floor=2) at 2-bit (needs WF allocation)
    # (WF uses eigenvalues for per-dim allocation — different from CWF per-head allocation)
    # For simplicity, compare against PCA+Uniform as the main baseline

    # CWF configs
    for avg_b in CWF_BUDGETS:
        configs_to_run.append((f"cwf_avg{avg_b}", "cwf", avg_b, None))

    for cfg_name, method, bits_or_avg, extra in configs_to_run:
        cfg_path = out_dir / f"{short}_{cfg_name}_s{args.seed}.json"
        if cfg_path.exists():
            cached = json.loads(cfg_path.read_text())
            print(f"  {cfg_name}: PPL={cached['ppl']} (cached)")
            results["configs"][cfg_name] = cached
            continue

        print(f"  {cfg_name}...", end=" ", flush=True)
        t0 = time.time()
        hook_handles = []

        if method == "pca_uniform":
            for li in range(n_layers):
                pca_b = {hk: pca_bases_all[(li, hk)] for hk in range(n_kv)}
                hook_handles.append(
                    layers[li].self_attn.k_proj.register_forward_hook(
                        pca_uniform_hook(pca_b, n_kv, d_head, bits_or_avg)))

        elif method == "cwf":
            avg_bits = bits_or_avg
            total_head_budget = int(round(avg_bits * n_layers * n_kv))
            head_bits = water_filling_global(importance, total_head_budget, B_FLOOR, B_MAX)
            actual_avg = head_bits.mean()

            # Fit Lloyd per head at allocated bits
            head_quantizers = {}
            for li in range(n_layers):
                for hk in range(n_kv):
                    idx = li * n_kv + hk
                    b = int(head_bits[idx])
                    head_quantizers[(li, hk)] = fit_pca_lloyd(all_calib_K[(li, hk)], b)

            for li in range(n_layers):
                hq = {hk: head_quantizers[(li, hk)] for hk in range(n_kv)}
                hook_handles.append(
                    layers[li].self_attn.k_proj.register_forward_hook(
                        QuantHook(hq, n_kv, d_head)))

        ppl, nc, ns = eval_ppl(model, eval_ids, device)
        for h in hook_handles:
            h.remove()
        elapsed = time.time() - t0

        result = {
            "ppl": round(ppl, 4), "chunks": nc, "scored_tokens": ns,
            "runtime_sec": round(elapsed, 1), "method": method,
        }
        if method == "cwf":
            result["avg_bits_target"] = bits_or_avg
            result["avg_bits_actual"] = round(float(actual_avg), 4)
            result["bit_distribution"] = {
                "min": int(head_bits.min()), "max": int(head_bits.max()),
                "mean": round(float(head_bits.mean()), 3)
            }

        cfg_path.write_text(json.dumps(result, indent=2))
        results["configs"][cfg_name] = result
        print(f"PPL={ppl:.4f} ({elapsed:.1f}s)")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.model}")
    print(f"{'='*60}")
    print(f"FP16: {ppl_fp16:.4f}")
    for cfg_name, cfg_data in sorted(results["configs"].items()):
        extra = ""
        if "avg_bits_actual" in cfg_data:
            extra = f" (avg={cfg_data['avg_bits_actual']:.3f}b)"
        print(f"  {cfg_name}: PPL={cfg_data['ppl']:.4f}{extra}")

    # Save full results
    full_path = out_dir / f"{short}_full_s{args.seed}.json"
    full_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {full_path}")


if __name__ == "__main__":
    main()
