#!/usr/bin/env python3
"""
Oracle Bit Allocation: Query-Direction Distortion vs MSE
=========================================================

Kill-criterion experiment: Does allocating more bits to tokens whose keys
are important for future query directions improve PPL over uniform allocation?

Single-layer intervention (layer 16), 50/50 split (1-bit / 3-bit), avg 2 bits.

Methods (all avg 2 bits/dim for intervened layer):
  A. FP16 — ceiling (no quantization)
  B. Uniform 2-bit — standard baseline
  C. Random 1/3 split — 50% tokens 1-bit, 50% 3-bit, random
  D. Importance-guided 1/3 split — 3-bit to top-50% by query-direction score
  E. Position-based 1/3 split — first 50% tokens get 3-bit (positional bias control)

Score: mean_{q>t} (q_postRoPE · k_postRoPE)^2  (normalized by # future queries)

Usage:
  CUDA_VISIBLE_DEVICES=0 python exp_oracle_bit_allocation.py \
    --model mistralai/Mistral-7B-v0.3 --target-layer 16
"""
import argparse
import gc
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE = torch.bfloat16
EVAL_CHUNK = 2048
MAX_EVAL = 50000


# ============================================================================
# Quantizers
# ============================================================================

def quantize_uniform(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-dim asymmetric uniform quantization."""
    if bits >= 16:
        return x
    n_lev = 2 ** bits
    c_min = x.amin(dim=0, keepdim=True)
    c_max = x.amax(dim=0, keepdim=True)
    rng = (c_max - c_min).clamp(min=1e-10)
    step = rng / (n_lev - 1)
    return torch.round((x - c_min) / step) * step + c_min


def quantize_per_token_variable(K: torch.Tensor, bit_assign: torch.Tensor) -> torch.Tensor:
    """Per-token variable-bit quantization.

    Args:
        K: (seq, d) key tensor for one head
        bit_assign: (seq,) int tensor with bit assignment per token (1, 2, 3, or 4)
    Returns:
        K_q: (seq, d) quantized key tensor
    """
    K_q = K.clone()
    for b in [1, 2, 3, 4]:
        mask = (bit_assign == b)
        if mask.any():
            tokens = K[mask]  # (n_tokens, d)
            K_q[mask] = quantize_uniform(tokens, b)
    return K_q


# ============================================================================
# Oracle Scoring
# ============================================================================

def compute_importance_scores(Q_post: torch.Tensor, K_post: torch.Tensor,
                              n_q_heads: int, n_kv_heads: int
                              ) -> torch.Tensor:
    """Compute per-token importance scores from post-RoPE Q and K.

    score(t, h) = mean_{q > t} (q · k_t)^2  (mean-normalized to avoid positional bias)

    Args:
        Q_post: (batch, n_q_heads, seq, d) post-RoPE query states
        K_post: (batch, n_kv_heads, seq, d) post-RoPE key states
    Returns:
        scores: (seq, n_kv_heads) importance scores
    """
    B, _, S, D = Q_post.shape
    G = n_q_heads // n_kv_heads  # GQA group size
    scores = torch.zeros(S, n_kv_heads, device=Q_post.device)

    for hk in range(n_kv_heads):
        k_h = K_post[0, hk]  # (S, D)

        # Aggregate query heads in this GQA group
        q_group = Q_post[0, hk*G:(hk+1)*G]  # (G, S, D)
        q_mean = q_group.mean(dim=0)  # (S, D) — average across G query heads

        # Compute (q_i · k_t)^2 for all pairs where i > t
        # qk: (S_q, S_k) = q_mean @ k_h^T
        qk = (q_mean @ k_h.T).float()  # (S, S)
        qk_sq = qk.pow(2)  # (S, S)

        # Causal mask: only future queries (i > t)
        # For token t, sum qk_sq[i, t] for i > t, then divide by (S - t - 1)
        causal = torch.triu(torch.ones(S, S, device=qk.device), diagonal=1)  # (S, S), 1 where i > t
        future_sum = (qk_sq * causal).sum(dim=0)  # (S,) — sum over query positions i for each key t
        future_count = causal.sum(dim=0).clamp(min=1)  # (S,) — number of future queries
        scores[:, hk] = future_sum / future_count  # mean-normalized

    return scores


def assign_bits_by_score(scores: torch.Tensor, method: str,
                         n_kv_heads: int, seq_len: int,
                         low_bit: int = 1, high_bit: int = 3) -> torch.Tensor:
    """Assign bits per token × KV head.

    Args:
        scores: (seq, n_kv_heads) importance scores
        method: 'uniform', 'random', 'guided', 'position'
        n_kv_heads: number of KV heads
        seq_len: sequence length
        low_bit: bits for bottom 50% tokens (default: 1)
        high_bit: bits for top 50% tokens (default: 3)
    Returns:
        bit_assign: (seq, n_kv_heads) int tensor
    """
    avg_bit = (low_bit + high_bit) / 2.0
    bit_assign = torch.full((seq_len, n_kv_heads), round(avg_bit), dtype=torch.int32)

    if method == 'uniform':
        bit_assign[:] = round(avg_bit)
        return bit_assign

    # 50/50 split: top 50% get high_bit, bottom 50% get low_bit
    half = seq_len // 2

    if method == 'random':
        for hk in range(n_kv_heads):
            perm = torch.randperm(seq_len)
            bit_assign[perm[:half], hk] = high_bit
            bit_assign[perm[half:], hk] = low_bit

    elif method == 'guided':
        for hk in range(n_kv_heads):
            _, idx = scores[:, hk].sort(descending=True)
            bit_assign[idx[:half], hk] = high_bit
            bit_assign[idx[half:], hk] = low_bit

    elif method == 'position':
        bit_assign[:half, :] = high_bit
        bit_assign[half:, :] = low_bit

    return bit_assign


# ============================================================================
# Single-Layer Intervention PPL Evaluation
# ============================================================================

def find_attn_modules(model):
    """Find attention modules."""
    modules = []
    for name, mod in model.named_modules():
        if 'Attention' in type(mod).__name__ and hasattr(mod, 'k_proj'):
            modules.append((name, mod))
    return modules


@torch.no_grad()
def evaluate_single_layer_intervention(
    model, tokenizer, device: str,
    target_layer: int, method: str,
    max_eval: int = MAX_EVAL,
    low_bit: int = 1, high_bit: int = 3,
) -> Dict:
    """Evaluate PPL with single-layer variable-bit K quantization.

    Steps:
    1. Run FP16 forward to extract post-RoPE Q, K at target layer
    2. Compute importance scores
    3. Assign bits per token
    4. Re-run with quantized K at target layer
    """
    from datasets import load_dataset
    eval_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_text = "\n\n".join([t for t in eval_ds["text"] if t.strip()])
    eval_ids = tokenizer.encode(eval_text, return_tensors="pt", truncation=False)

    total_len = min(eval_ids.shape[1], max_eval)
    n_chunks = total_len // EVAL_CHUNK
    eval_ids = eval_ids[:, :n_chunks * EVAL_CHUNK]

    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, 'num_key_value_heads', n_heads)
    d_head = cfg.hidden_size // n_heads

    attn_modules = find_attn_modules(model)
    target_attn = attn_modules[target_layer][1]

    total_nll = 0.0
    total_count = 0
    t0 = time.time()

    for ci in range(n_chunks):
        chunk = eval_ids[:, ci*EVAL_CHUNK:(ci+1)*EVAL_CHUNK].to(device)
        B, S = chunk.shape

        if method == 'fp16':
            # No intervention
            out = model(chunk, use_cache=False)
            logits = out.logits.float()
        else:
            # Step 1: Extract post-RoPE Q, K at target layer via hooks
            q_pre_data = {}
            k_pre_data = {}

            def q_hook(mod, inp, out):
                q_pre_data[0] = out.detach()
            def k_hook(mod, inp, out):
                k_pre_data[0] = out.detach()

            hq = target_attn.q_proj.register_forward_hook(q_hook)
            hk = target_attn.k_proj.register_forward_hook(k_hook)

            # First pass: FP16 to get Q, K (discard output)
            model(chunk, use_cache=False)

            hq.remove()
            hk.remove()

            # Reshape and apply RoPE
            Q_raw = q_pre_data[0].view(B, S, n_heads, d_head).transpose(1, 2).float()
            K_raw = k_pre_data[0].view(B, S, n_kv, d_head).transpose(1, 2).float()

            position_ids = torch.arange(S, device=device).unsqueeze(0)
            if hasattr(target_attn, 'rotary_emb'):
                dummy = torch.zeros(B, n_kv, S, d_head, device=device, dtype=Q_raw.dtype)
                cos, sin = target_attn.rotary_emb(dummy, position_ids)
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                Q_post, K_post = apply_rotary_pos_emb(Q_raw.to(cos.dtype), K_raw.to(cos.dtype), cos, sin)
                Q_post = Q_post.float()
                K_post = K_post.float()
            else:
                warnings.warn("rotary_emb not found; scoring with pre-RoPE states")
                Q_post, K_post = Q_raw, K_raw

            # Step 2: Compute importance scores
            scores = compute_importance_scores(Q_post, K_post, n_heads, n_kv)

            # Step 3: Assign bits
            effective_method = method
            effective_low = low_bit
            effective_high = high_bit
            if method == 'uniform_3bit':
                effective_method = 'uniform'
                effective_low = 3
                effective_high = 3
            bit_assign = assign_bits_by_score(scores, effective_method, n_kv, S,
                                                low_bit=effective_low, high_bit=effective_high)

            # Step 4: Quantize K at target layer with variable bits
            # Hook that applies variable-bit quantization to k_proj output
            k_pre_float = k_pre_data[0].float()  # (B, S, n_kv * d_head)

            def quant_hook(mod, inp, out):
                out_f = out.float()
                B_, S_, D_ = out_f.shape
                K_heads = out_f.view(B_, S_, n_kv, d_head)
                K_q = torch.zeros_like(K_heads)
                for hk in range(n_kv):
                    K_q[0, :, hk] = quantize_per_token_variable(
                        K_heads[0, :, hk], bit_assign[:, hk].to(device)
                    )
                return K_q.view(B_, S_, D_).to(out.dtype)

            hk2 = target_attn.k_proj.register_forward_hook(quant_hook)

            # Second pass: with quantized K
            out = model(chunk, use_cache=False)
            hk2.remove()
            logits = out.logits.float()

        # Compute NLL
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), reduction='sum')
        total_nll += loss.item()
        total_count += shift_labels.numel()

        if (ci + 1) % 4 == 0:
            ppl = math.exp(min(total_nll / max(total_count, 1), 100))
            print(f"    chunk {ci+1}/{n_chunks} ppl={ppl:.4f}", flush=True)

        del out, logits
        if method != 'fp16':
            del Q_raw, K_raw, Q_post, K_post, scores, bit_assign
        gc.collect()
        torch.cuda.empty_cache()

    ppl = math.exp(total_nll / max(total_count, 1))
    elapsed = time.time() - t0

    return {
        "method": method,
        "ppl": round(ppl, 4),
        "target_layer": target_layer,
        "n_chunks": n_chunks,
        "runtime_s": round(elapsed, 1),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Oracle Bit Allocation Experiment")
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-layer", type=int, default=16)
    parser.add_argument("--max-eval-tokens", type=int, default=MAX_EVAL)
    parser.add_argument("--output-dir", default="results/oracle_bit_alloc")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    short = args.model.split("/")[-1].replace(".", "_")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"{'='*60}")
    print(f"ORACLE BIT ALLOCATION: {args.model}")
    print(f"Target layer: {args.target_layer}, Max eval: {args.max_eval_tokens}")
    print(f"{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=DTYPE, trust_remote_code=True,
        attn_implementation="eager"
    ).to(device).eval()

    # Run two regimes: 1/3-bit split (avg 2.0) and 2/3-bit split (avg 2.5)
    splits = [
        {"name": "1_3", "low": 1, "high": 3, "avg": 2.0},
        {"name": "2_3", "low": 2, "high": 3, "avg": 2.5},
    ]

    all_results = {}

    for split in splits:
        print(f"\n{'='*60}")
        print(f"SPLIT: {split['low']}/{split['high']}-bit (avg {split['avg']})")
        print(f"{'='*60}", flush=True)

        methods = ['fp16', 'uniform', 'random', 'guided', 'position']
        # For 2/3 split, also add uniform-3bit as upper reference
        if split['low'] == 2:
            methods.append('uniform_3bit')
        split_results = {}

        for method in methods:
            tag = f"{method}_{split['name']}" if method not in ('fp16',) else method
            print(f"\n[{tag}]", flush=True)
            r = evaluate_single_layer_intervention(
                model, tokenizer, device,
                target_layer=args.target_layer,
                method=method,
                max_eval=args.max_eval_tokens,
                low_bit=split['low'], high_bit=split['high'],
            )
            r['split'] = split['name']
            r['avg_bits'] = split['avg']
            split_results[method] = r
            all_results[tag] = r
            print(f"  PPL = {r['ppl']}", flush=True)

        # Summary for this split
        print(f"\n--- {split['name']} split summary ---")
        print(f"{'Method':<15s} {'PPL':>10s} {'Time':>8s}")
        for m in methods:
            r = split_results[m]
            print(f"{m:<15s} {r['ppl']:>10.4f} {r['runtime_s']:>8.1f}s")

        # Kill criterion
        if split_results['guided']['ppl'] < split_results['random']['ppl']:
            delta = split_results['random']['ppl'] - split_results['guided']['ppl']
            print(f"\n  CONFIRMED: guided < random (Δ={delta:.4f})")
        else:
            print(f"\n  FAILED: guided >= random")

        # Key comparison: does guided beat uniform?
        if split_results['guided']['ppl'] < split_results['uniform']['ppl']:
            delta = split_results['uniform']['ppl'] - split_results['guided']['ppl']
            print(f"  BONUS: guided < uniform (Δ={delta:.4f}) *** STRONG SIGNAL ***")
        else:
            delta = split_results['guided']['ppl'] - split_results['uniform']['ppl']
            print(f"  Uniform still wins (uniform is {delta:.4f} better)")

    out_path = out_dir / f"{short}_layer{args.target_layer}_oracle_v2.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
