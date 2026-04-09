#!/usr/bin/env python3
"""
Needle Ranking Diagnostic: Does 2-bit preserve needle's rank in top-K?

This is the kill-criterion for CliffKV. If the needle token is NOT in
the top-K of 2-bit attention scores, selective refinement cannot save it.

For each NIAH trial:
1. Insert needle at a known position in haystack
2. Compute FP16 and 2-bit attention scores at the retrieval query
3. Check needle's rank under both

Usage:
  CUDA_VISIBLE_DEVICES=0 python exp_needle_ranking.py \
    --model mistralai/Mistral-7B-v0.3
"""
import argparse
import gc
import json
import math
import os
import time
import warnings
from pathlib import Path

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F

DTYPE = torch.bfloat16


def uniform_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-dim asymmetric uniform quantization."""
    n_lev = 2 ** bits
    c_min = x.amin(dim=-2, keepdim=True)
    c_max = x.amax(dim=-2, keepdim=True)
    rng = (c_max - c_min).clamp(min=1e-10)
    step = rng / (n_lev - 1)
    return torch.round((x - c_min) / step) * step + c_min


def make_niah_input(tokenizer, context_len: int, needle_depth: float,
                    needle_text: str = "The secret number is 42.",
                    query_text: str = "What is the secret number?"):
    """Create a NIAH prompt with needle inserted at given depth."""
    # Haystack filler
    filler = "This is a passage of filler text that serves as haystack content for the needle-in-a-haystack evaluation. It contains no useful information and is designed to pad the context. "

    needle_tokens = tokenizer.encode(needle_text, add_special_tokens=False)
    query_tokens = tokenizer.encode(query_text, add_special_tokens=False)

    # Build haystack
    filler_tokens = tokenizer.encode(filler * 100, add_special_tokens=False)

    # Calculate positions
    available = context_len - len(needle_tokens) - len(query_tokens) - 10
    needle_pos = int(available * needle_depth)

    # Assemble: [haystack_before] [needle] [haystack_after] [query]
    hay_before = filler_tokens[:needle_pos]
    hay_after = filler_tokens[:available - needle_pos]

    full_tokens = hay_before + needle_tokens + hay_after + query_tokens
    full_tokens = full_tokens[:context_len]

    input_ids = torch.tensor([full_tokens])
    needle_start = len(hay_before)
    needle_end = needle_start + len(needle_tokens)

    return input_ids, needle_start, needle_end


@torch.no_grad()
def check_needle_ranking(model, tokenizer, device,
                         context_len=4096, needle_depth=0.5,
                         bits_list=[2, 3, 4]):
    """Check needle token's rank in attention scores under quantization."""

    input_ids, needle_start, needle_end = make_niah_input(
        tokenizer, context_len, needle_depth)
    input_ids = input_ids.to(device)

    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, 'num_key_value_heads', n_heads)
    d_head = cfg.hidden_size // n_heads
    n_layers = cfg.num_hidden_layers
    G = n_heads // n_kv

    # Extract Q, K at all layers via hooks
    q_data = {}
    k_data = {}
    hooks = []

    def find_attn(model):
        modules = []
        for name, mod in model.named_modules():
            if 'Attention' in type(mod).__name__ and hasattr(mod, 'k_proj'):
                modules.append((name, mod))
        return modules

    attn_modules = find_attn(model)

    for li, (name, attn) in enumerate(attn_modules):
        def qh(li_=li):
            def fn(mod, inp, out):
                q_data[li_] = out.detach()
            return fn
        def kh(li_=li):
            def fn(mod, inp, out):
                k_data[li_] = out.detach()
            return fn
        hooks.append(attn.q_proj.register_forward_hook(qh()))
        hooks.append(attn.k_proj.register_forward_hook(kh()))

    model(input_ids, use_cache=False)
    for h in hooks:
        h.remove()

    S = input_ids.shape[1]
    query_pos = S - 1  # last token is the query

    results = {"context_len": context_len, "needle_depth": needle_depth,
               "needle_range": [needle_start, needle_end], "seq_len": S}

    for bits in [16] + bits_list:
        ranks_per_layer = []
        scores_per_layer = []

        for li, (name, attn) in enumerate(attn_modules):
            Q = q_data[li].view(1, S, n_heads, d_head).transpose(1, 2).float()
            K = k_data[li].view(1, S, n_kv, d_head).transpose(1, 2).float()

            # Apply RoPE
            pos_ids = torch.arange(S, device=device).unsqueeze(0)
            if hasattr(attn, 'rotary_emb'):
                dummy = torch.zeros(1, n_kv, S, d_head, device=device, dtype=Q.dtype)
                cos, sin = attn.rotary_emb(dummy, pos_ids)
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                Q, K = apply_rotary_pos_emb(Q.to(cos.dtype), K.to(cos.dtype), cos, sin)
                Q, K = Q.float(), K.float()

            # Quantize K if not FP16
            if bits < 16:
                K_q = uniform_quantize(K, bits)
            else:
                K_q = K

            # Expand for GQA
            K_exp = K_q.repeat_interleave(G, dim=1)

            # Get query vector at last position
            q_last = Q[:, :, query_pos:query_pos+1, :]  # (1, n_heads, 1, d)

            # Attention scores: (1, n_heads, 1, S)
            scores = (q_last @ K_exp.transpose(-1, -2)) / math.sqrt(d_head)
            scores = scores.squeeze(2)  # (1, n_heads, S)

            # For each head, find needle tokens' rank
            for head in range(n_heads):
                s = scores[0, head]  # (S,)
                # Needle score = max score among needle positions
                needle_score = s[needle_start:needle_end].max().item()
                # Rank = how many tokens have higher score
                rank = (s > needle_score).sum().item() + 1
                ranks_per_layer.append(rank)
                scores_per_layer.append(needle_score)

        # Aggregate: median and worst-case rank across heads and layers
        ranks = np.array(ranks_per_layer)
        key = f"{bits}bit" if bits < 16 else "fp16"
        results[key] = {
            "median_rank": int(np.median(ranks)),
            "mean_rank": round(float(np.mean(ranks)), 1),
            "p95_rank": int(np.percentile(ranks, 95)),
            "max_rank": int(ranks.max()),
            "min_rank": int(ranks.min()),
            "pct_in_top32": round(float((ranks <= 32).mean()) * 100, 1),
            "pct_in_top64": round(float((ranks <= 64).mean()) * 100, 1),
            "pct_in_top128": round(float((ranks <= 128).mean()) * 100, 1),
            "total_heads_layers": len(ranks),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="results/needle_ranking")
    parser.add_argument("--context-lens", nargs="+", type=int, default=[4096, 8192])
    parser.add_argument("--depths", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9])
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"NEEDLE RANKING DIAGNOSTIC: {args.model}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=DTYPE, trust_remote_code=True,
        attn_implementation="eager"
    ).to(args.device).eval()

    all_results = []

    for ctx in args.context_lens:
        for depth in args.depths:
            print(f"\n[ctx={ctx}, depth={depth}]", flush=True)
            r = check_needle_ranking(model, tokenizer, args.device,
                                     context_len=ctx, needle_depth=depth)
            all_results.append(r)

            # Print summary
            for key in ['fp16', '2bit', '3bit', '4bit']:
                if key in r:
                    d = r[key]
                    print(f"  {key}: median_rank={d['median_rank']}, "
                          f"top32={d['pct_in_top32']}%, "
                          f"top64={d['pct_in_top64']}%, "
                          f"max_rank={d['max_rank']}", flush=True)

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'ctx':>6} {'depth':>6} {'FP16 med':>9} {'2bit med':>9} {'2bit top64%':>12} {'3bit med':>9}")
    for r in all_results:
        fp16 = r.get('fp16', {})
        b2 = r.get('2bit', {})
        b3 = r.get('3bit', {})
        print(f"{r['context_len']:>6} {r['needle_depth']:>6.1f} "
              f"{fp16.get('median_rank', '?'):>9} {b2.get('median_rank', '?'):>9} "
              f"{b2.get('pct_in_top64', '?'):>12} {b3.get('median_rank', '?'):>9}")

    short = args.model.split("/")[-1].replace(".", "_")
    out_path = out_dir / f"{short}_needle_ranking.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
