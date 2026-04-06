#!/usr/bin/env python3
"""
Simple MMLU evaluation with quantized KV cache.
No external dependencies beyond transformers, lm-eval, torch, numpy.

Usage:
  CUDA_VISIBLE_DEVICES=0 python run_mmlu_simple.py --model Qwen/Qwen2.5-7B --method fp16 --tasks mmlu --num-fewshot 5
  CUDA_VISIBLE_DEVICES=0 python run_mmlu_simple.py --model Qwen/Qwen2.5-7B --method pre_pca --bits 2 --tasks mmlu --num-fewshot 5
"""
from __future__ import annotations
import argparse, gc, json, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
warnings.filterwarnings("ignore")


def uniform_quant_col(col, bits):
    nl = 2 ** bits
    vmin, vmax = col.min(), col.max()
    if vmax - vmin < 1e-10: return col.copy()
    s = (vmax - vmin) / (nl - 1)
    q = np.clip(np.round((col - vmin) / s).astype(int), 0, nl - 1)
    return q * s + vmin


def make_hook(layer_idx, bits, n_kv, d_head, pca_bases=None, random_rot=None):
    def hook_fn(module, input, output):
        k = output[0] if isinstance(output, tuple) else output
        k_np = k.detach().cpu().float().numpy()
        orig_shape = k_np.shape
        k_flat = k_np.reshape(-1, n_kv, d_head)
        for h in range(n_kv):
            Kh = k_flat[:, h, :]
            R = pca_bases.get((layer_idx, h)) if pca_bases else random_rot
            Kr = Kh @ R if R is not None else Kh.copy()
            for j in range(d_head):
                Kr[:, j] = uniform_quant_col(Kr[:, j], bits)
            k_flat[:, h, :] = Kr @ R.T if R is not None else Kr
        return torch.tensor(k_flat.reshape(orig_shape), dtype=k.dtype, device=k.device)
    return hook_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--method", type=str, default="fp16",
                        choices=["fp16", "no_rot", "random_rot", "pre_pca"])
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--tasks", type=str, nargs="+", default=["mmlu"])
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = "cuda:0"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"\n{'#'*60}")
    print(f"# MMLU: {args.model} | {args.method} {args.bits}b")
    print(f"{'#'*60}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()

    cfg = model.config
    n_kv = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    n_layers = cfg.num_hidden_layers
    d_head = cfg.hidden_size // cfg.num_attention_heads
    layers = model.model.layers if hasattr(model, 'model') else model.transformer.h

    hook_handles = []

    if args.method != "fp16":
        # Calibrate
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join([t for t in ds["text"] if t.strip()][:50])
        cal_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=2048).to(device)

        pre_capture = {}
        cal_hooks = []
        def make_cal(li):
            def fn(mod, a, kw):
                hs = a[0] if a else kw.get('hidden_states')
                if hs is not None:
                    k = mod.k_proj(hs)[0].detach().cpu().float().numpy().reshape(-1, n_kv, d_head)
                    for h in range(n_kv):
                        pre_capture[(li, h)] = k[:, h, :]
            return fn
        for l in range(n_layers):
            cal_hooks.append(layers[l].self_attn.register_forward_pre_hook(make_cal(l), with_kwargs=True))
        with torch.no_grad():
            model(cal_ids, use_cache=False)
        for h in cal_hooks:
            h.remove()

        pca_bases = {}
        for (l, h), K in pre_capture.items():
            Kc = K - K.mean(0)
            _, V = np.linalg.eigh(Kc.T @ Kc / K.shape[0])
            pca_bases[(l, h)] = V

        np.random.seed(42)
        R_random = np.linalg.qr(np.random.randn(d_head, d_head))[0]

        pca = pca_bases if args.method == "pre_pca" else None
        rand = R_random if args.method == "random_rot" else None

        for l in range(n_layers):
            hook_handles.append(layers[l].self_attn.k_proj.register_forward_hook(
                make_hook(l, args.bits, n_kv, d_head, pca_bases=pca, random_rot=rand)))
        print(f"  Installed {len(hook_handles)} hooks")

    # Run lm-eval
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
    results = lm_eval.simple_evaluate(model=lm, tasks=args.tasks, num_fewshot=args.num_fewshot, limit=args.limit)

    for hk in hook_handles:
        hk.remove()

    # Print and save
    output = {'model': args.model, 'method': args.method, 'bits': args.bits, 'tasks': {}}
    for task_name, task_results in results['results'].items():
        acc = task_results.get('acc,none', task_results.get('acc_norm,none'))
        print(f"  {task_name}: acc={acc}")
        output['tasks'][task_name] = {'acc': acc}

    outdir = Path("/scratch/boltzmann/results")
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.model.replace("/", "_")
    outf = outdir / f"mmlu_{tag}_{args.method}_{args.bits}b_{ts}.json"
    with open(outf, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {outf}")

    del model; torch.cuda.empty_cache(); gc.collect()

if __name__ == '__main__':
    main()
