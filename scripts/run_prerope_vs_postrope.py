#!/usr/bin/env python3
"""
ESSENTIAL experiment: Pre-RoPE PCA vs Post-RoPE PCA PPL comparison.
Tests the unique prediction of Theorem 1: Pre-RoPE PCA is optimal, Post-RoPE is not.

Usage:
  CUDA_VISIBLE_DEVICES=1 python run_prerope_vs_postrope.py --model Qwen/Qwen2.5-7B --bits 2 3 4
  CUDA_VISIBLE_DEVICES=2 python run_prerope_vs_postrope.py --model meta-llama/Llama-3.1-8B --bits 2 3 4
"""
import argparse, json, time, gc, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


def uniform_quant_col(col, bits):
    nl = 2 ** bits
    vmin, vmax = col.min(), col.max()
    if vmax - vmin < 1e-10:
        return col.copy()
    s = (vmax - vmin) / (nl - 1)
    q = np.clip(np.round((col - vmin) / s).astype(int), 0, nl - 1)
    return q * s + vmin


def make_k_quant_hook(layer_idx, bits, n_kv, d_head, pca_bases=None, random_rot=None):
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
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    device = "cuda:0"  # uses CUDA_VISIBLE_DEVICES externally
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"\n{'#'*70}")
    print(f"# Pre-RoPE vs Post-RoPE PCA: {args.model}")
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

    # Calibration: get Pre-RoPE keys
    cal_ids = all_ids[:, :2048].to(device)
    pre_capture = {}
    cal_hooks = []
    def make_cal_hook(li):
        def fn(mod, a, kw):
            hs = a[0] if a else kw.get('hidden_states')
            if hs is not None:
                k = mod.k_proj(hs)[0].detach().cpu().float().numpy().reshape(-1, n_kv, d_head)
                for h in range(n_kv):
                    pre_capture[(li, h)] = k[:, h, :]
        return fn
    for l in range(n_layers):
        cal_hooks.append(layers[l].self_attn.register_forward_pre_hook(make_cal_hook(l), with_kwargs=True))
    with torch.no_grad():
        out = model(cal_ids, use_cache=True)
    for hook in cal_hooks:
        hook.remove()

    # Pre-RoPE PCA
    pca_pre = {}
    for (l, h), K in pre_capture.items():
        Kc = K - K.mean(0)
        _, V = np.linalg.eigh(Kc.T @ Kc / K.shape[0])
        pca_pre[(l, h)] = V

    # Post-RoPE PCA (from past_key_values)
    pca_post = {}
    for l, item in enumerate(out.past_key_values):
        K = (item[0] if isinstance(item, tuple) else item.keys)[0].cpu().float().numpy()
        for h in range(n_kv):
            Kh = K[h]
            Kc = Kh - Kh.mean(0)
            _, V = np.linalg.eigh(Kc.T @ Kc / Kh.shape[0])
            pca_post[(l, h)] = V

    np.random.seed(42)
    R_random = np.linalg.qr(np.random.randn(d_head, d_head))[0]

    # PPL evaluation
    chunk_len, max_tokens = 2048, min(all_ids.shape[1], 50000)
    n_chunks = (max_tokens - 1) // chunk_len

    def eval_ppl(label, pca=None, rand_rot=None):
        hks = []
        for l in range(n_layers):
            hks.append(layers[l].self_attn.k_proj.register_forward_hook(
                make_k_quant_hook(l, bits, n_kv, d_head, pca_bases=pca, random_rot=rand_rot)))
        nll, tok = 0.0, 0
        with torch.no_grad():
            for ci in range(n_chunks):
                s = ci * chunk_len
                e = s + chunk_len + 1
                if e > max_tokens: break
                inp = all_ids[:, s:e].to(device)
                out = model(inp[:, :-1], use_cache=False)
                nll += nn.CrossEntropyLoss(reduction='sum')(out.logits[0], inp[0, 1:]).item()
                tok += inp.shape[1] - 1
        for hk in hks:
            hk.remove()
        ppl = np.exp(nll / tok)
        print(f"    {label}: PPL={ppl:.4f} ({tok} tokens)")
        return float(ppl)

    results = {'model': args.model, 'methods': {}}
    for bits in args.bits:
        print(f"\n  --- {bits}-bit ---")
        results['methods'][f'{bits}bit'] = {
            'no_rot': eval_ppl(f"NoRot_{bits}b"),
            'turbo': eval_ppl(f"Turbo_{bits}b", rand_rot=R_random),
            'post_rope_pca': eval_ppl(f"PostRoPE_PCA_{bits}b", pca=pca_post),
            'pre_rope_pca': eval_ppl(f"PreRoPE_PCA_{bits}b", pca=pca_pre),
        }
        r = results['methods'][f'{bits}bit']
        print(f"    Pre < Post? {r['pre_rope_pca'] < r['post_rope_pca']}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.model.replace("/", "_")
    outf = Path(args.output_dir) / f"prerope_vs_postrope_{tag}_{ts}.json"
    with open(outf, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outf}")

    del model; torch.cuda.empty_cache(); gc.collect()


if __name__ == '__main__':
    main()
