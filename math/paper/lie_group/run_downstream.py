#!/usr/bin/env python3
"""
Downstream task evaluation with quantized KV cache.
Uses lm-evaluation-harness with k_proj forward hooks.

Usage:
  python run_downstream.py --model meta-llama/Llama-3.1-8B --method pre_pca_wf2 --bits 2 --tasks mmlu --device cuda:0
  python run_downstream.py --model meta-llama/Llama-3.1-8B --method fp16 --tasks mmlu --device cuda:0
  python run_downstream.py --model meta-llama/Llama-3.1-8B --method turbo --bits 2 --tasks gsm8k --device cuda:0
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

OUTDIR = Path(__file__).parent / "verification_results"

# ============================================================================
# Quantization (reuse from verify_v15)
# ============================================================================
def uniform_quant_col(col, bits):
    nl = 2 ** bits
    vmin, vmax = col.min(), col.max()
    if vmax - vmin < 1e-10:
        return col.copy()
    s = (vmax - vmin) / (nl - 1)
    q = np.clip(np.round((col - vmin) / s).astype(int), 0, nl - 1)
    return q * s + vmin


def compute_wf_alloc(variances, avg_bits, min_bits=2):
    d = len(variances)
    max_bits = 4
    log_var = np.log2(np.maximum(variances, 1e-10))
    mean_log_var = np.mean(log_var)
    b_alloc = avg_bits + 0.5 * (log_var - mean_log_var)
    b_alloc = np.clip(b_alloc, min_bits, max_bits)
    total_target = d * avg_bits
    b_alloc = b_alloc * (total_target / max(b_alloc.sum(), 1e-10))
    b_alloc = np.maximum(np.round(b_alloc), min_bits)
    b_alloc = np.minimum(b_alloc, max_bits)
    deficit = int(total_target - b_alloc.sum())
    if deficit > 0:
        headroom = max_bits - b_alloc
        idx = np.argsort(-headroom)
        for i in idx[:deficit]:
            if b_alloc[i] < max_bits:
                b_alloc[i] += 1
    elif deficit < 0:
        slack = b_alloc - min_bits
        idx = np.argsort(slack)
        for i in idx[:abs(deficit)]:
            if b_alloc[i] > min_bits:
                b_alloc[i] -= 1
    return b_alloc.astype(int)


# ============================================================================
# Hook factory
# ============================================================================
def make_k_quant_hook(layer_idx, bits, n_kv, d_head,
                      pca_bases=None, random_rot=None, wf_alloc=None):
    def hook_fn(module, input, output):
        k = output[0] if isinstance(output, tuple) else output
        k_np = k.detach().cpu().float().numpy()
        orig_shape = k_np.shape
        k_flat = k_np.reshape(-1, n_kv, d_head)

        for h in range(n_kv):
            Kh = k_flat[:, h, :]
            if pca_bases is not None and (layer_idx, h) in pca_bases:
                R = pca_bases[(layer_idx, h)]
            elif random_rot is not None:
                R = random_rot
            else:
                R = None

            Kr = Kh @ R if R is not None else Kh.copy()
            for j in range(d_head):
                b = int(wf_alloc[(layer_idx, h)][j]) if wf_alloc and (layer_idx, h) in wf_alloc else bits
                b = max(min(b, 4), 1)
                Kr[:, j] = uniform_quant_col(Kr[:, j], b)
            k_flat[:, h, :] = Kr @ R.T if R is not None else Kr

        return torch.tensor(k_flat.reshape(orig_shape), dtype=k.dtype, device=k.device)
    return hook_fn


# ============================================================================
# Calibration + hook installation
# ============================================================================
def calibrate_and_install_hooks(model, method, bits, device):
    """Calibrate PCA bases and install quantization hooks."""
    from transformers import AutoTokenizer
    from datasets import load_dataset

    cfg = model.config
    n_kv = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    n_layers = cfg.num_hidden_layers
    d_head = cfg.hidden_size // cfg.num_attention_heads
    layers = model.model.layers if hasattr(model, 'model') else model.transformer.h

    if method == 'fp16':
        return []  # No hooks

    # Calibration: extract pre-RoPE keys
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, trust_remote_code=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join([t for t in ds["text"] if t.strip()][:50])
    cal_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=2048).to(device)

    pre_capture = {}
    cal_hooks = []

    def make_cal_hook(li):
        def fn(mod, args_t, kwargs_t):
            hs = args_t[0] if args_t else kwargs_t.get('hidden_states')
            if hs is not None:
                k = mod.k_proj(hs)[0].detach().cpu().float().numpy().reshape(-1, n_kv, d_head)
                for h in range(n_kv):
                    pre_capture[(li, h)] = k[:, h, :]
        return fn

    for l in range(n_layers):
        cal_hooks.append(layers[l].self_attn.register_forward_pre_hook(make_cal_hook(l), with_kwargs=True))

    with torch.no_grad():
        model(cal_ids, use_cache=False)

    for hook in cal_hooks:
        hook.remove()

    # Compute PCA bases
    pca_bases = {}
    eigenvalues = {}
    for (l, h), K in pre_capture.items():
        Kc = K - K.mean(0)
        cov = Kc.T @ Kc / K.shape[0]
        eigvals, eigvecs = np.linalg.eigh(cov)
        pca_bases[(l, h)] = eigvecs
        eigenvalues[(l, h)] = np.maximum(eigvals, 1e-10)

    np.random.seed(42)
    R_random = np.linalg.qr(np.random.randn(d_head, d_head))[0]

    # Choose rotation and WF
    pca = None
    rand_rot = None
    wf_alloc = None

    if method in ('pre_pca_uni', 'pre_pca_wf2'):
        pca = pca_bases
    elif method == 'turbo':
        rand_rot = R_random

    if method == 'pre_pca_wf2':
        wf_alloc = {k: compute_wf_alloc(v, bits, min_bits=2) for k, v in eigenvalues.items()}

    # Install hooks
    hook_handles = []
    for l in range(n_layers):
        k_proj = layers[l].self_attn.k_proj
        hk = k_proj.register_forward_hook(
            make_k_quant_hook(l, bits, n_kv, d_head,
                              pca_bases=pca, random_rot=rand_rot, wf_alloc=wf_alloc))
        hook_handles.append(hk)

    print(f"  Installed {len(hook_handles)} hooks for method={method}, bits={bits}")
    return hook_handles


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--method", type=str, default="fp16",
                        choices=["fp16", "no_rot", "turbo", "pre_pca_uni", "pre_pca_wf2"])
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--tasks", type=str, nargs="+", default=["mmlu"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per task (for testing)")
    args = parser.parse_args()

    import lm_eval
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'#'*70}")
    print(f"# Downstream Eval: {args.model}")
    print(f"# Method: {args.method}, Bits: {args.bits}")
    print(f"# Tasks: {args.tasks}")
    print(f"{'#'*70}")

    t0 = time.time()

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True
    ).to(args.device).eval()

    # Install hooks
    hook_handles = calibrate_and_install_hooks(model, args.method, args.bits, args.device)

    # Run lm-eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)

    task_kwargs = {}
    for task in args.tasks:
        if args.num_fewshot is not None:
            task_kwargs[task] = {"num_fewshot": args.num_fewshot}

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
    )

    # Remove hooks
    for hk in hook_handles:
        hk.remove()

    # Print results
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Results: {args.model} | {args.method} {args.bits}bit")
    print(f"{'='*70}")

    output = {
        'model': args.model,
        'method': args.method,
        'bits': args.bits,
        'tasks': {},
        'elapsed': elapsed,
    }

    for task_name, task_results in results['results'].items():
        acc = task_results.get('acc,none', task_results.get('acc_norm,none', None))
        acc_stderr = task_results.get('acc_stderr,none', None)
        print(f"  {task_name}: acc={acc}")
        output['tasks'][task_name] = {
            'acc': acc,
            'acc_stderr': acc_stderr,
            'all_metrics': {k: v for k, v in task_results.items() if isinstance(v, (int, float))},
        }

    # Save
    OUTDIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.replace("/", "_")
    task_tag = "_".join(args.tasks)
    outfile = OUTDIR / f"downstream_{model_tag}_{args.method}_{args.bits}b_{task_tag}_{timestamp}.json"
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved: {outfile}")
    print(f"  Time: {elapsed:.1f}s")

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    main()
