#!/usr/bin/env python3
"""
Query-Conditional PCA Subspace Selection for KV Cache Quantization.

Core idea: At attention time, instead of using all d=128 quantized PCA
dimensions, select top-m dimensions most relevant to the current query.

Algorithm:
  Write-time: Store keys in PCA coordinates at 2-bit (all d dims)
  Query-time: score_j = |q_pca_j|^2 * (lambda_j - sigma2_eps_j)
              select top-m dims, compute attention using only those dims

Methods tested:
  - fp16: No quantization baseline
  - pca_all: PCA + uniform quantize all d dims (existing baseline)
  - pca_topk_query: Query-conditional top-m selection (our method)
  - pca_topk_fixed: Static top-m by lambda_j only (ablation)
  - pca_topk_random: Random m dims (ablation)
  - pca_topk_oracle: Oracle selector using FP16 inner products (upper bound)
  - fp16_topk_query: FP16 keys + query-conditional selection (control: isolates
                     effect of selection from quantization interaction)

Metrics:
  - Attention score MSE vs FP16
  - Per-layer attention MSE breakdown
  - PPL (WikiText-2, 49K test)
  - Signed logit bias, entropy shift

Usage:
  CUDA_VISIBLE_DEVICES=0 python exp_query_conditional_pca.py \
    --model mistralai/Mistral-7B-v0.3 --bits 2 --mode microbench
  CUDA_VISIBLE_DEVICES=0 python exp_query_conditional_pca.py \
    --model mistralai/Mistral-7B-v0.3 --bits 2 --mode ppl
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


# ============================================================================
# Configuration
# ============================================================================

CALIB_TOKENS = 4096
EVAL_CHUNK = 2048
MAX_EVAL = 50000
M_SWEEP = [16, 32, 48, 64, 96, 128]
DTYPE = torch.bfloat16


# ============================================================================
# Uniform Quantization (per-dim, asymmetric)
# ============================================================================

def uniform_quantize_1d(col: torch.Tensor, bits: int) -> Tuple[torch.Tensor, float]:
    """Quantize a 1D tensor to `bits` with uniform step. Returns (quantized, quant_noise_var)."""
    n_levels = 2 ** bits
    c_min = col.min()
    c_max = col.max()
    rng = (c_max - c_min).item()
    if rng < 1e-10:
        return col.clone(), 0.0
    step = rng / (n_levels - 1)
    q = torch.round((col - c_min) / step) * step + c_min
    noise_var = float((col - q).pow(2).mean().item())
    return q, noise_var


# ============================================================================
# PCA Calibration
# ============================================================================

class PCACalibration:
    """Per-head PCA calibration data."""

    def __init__(self, V: torch.Tensor, evals: torch.Tensor,
                 K_mean: torch.Tensor, quant_noise_var: torch.Tensor):
        """
        Args:
            V: (d, d) eigenvector matrix, columns sorted by descending eigenvalue
            evals: (d,) eigenvalues, descending
            K_mean: (d,) mean of calibration keys
            quant_noise_var: (d,) per-dim quantization noise variance in PCA space
        """
        self.V = V
        self.evals = evals
        self.K_mean = K_mean
        self.quant_noise_var = quant_noise_var


def _find_attn_modules(model):
    """Find attention modules in model (supports Llama/Mistral/Qwen)."""
    attn_modules = []
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if 'Attention' in cls_name and hasattr(module, 'k_proj'):
            attn_modules.append((name, module))
    return attn_modules


def calibrate_pca(model, tokenizer, device, bits: int, calib_tokens: int = CALIB_TOKENS
                  ) -> Dict[Tuple[int, int], PCACalibration]:
    """Extract per-head PCA basis from POST-ROPE K and quantization noise.

    Critical: PCA must be computed on post-RoPE K to be consistent with
    the attention computation where dimension selection happens.

    Returns: dict of (layer_idx, head_idx) -> PCACalibration
    """
    from datasets import load_dataset
    calib_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_text = "\n\n".join([t for t in calib_ds["text"] if t.strip()])
    calib_ids = tokenizer.encode(calib_text, return_tensors="pt", truncation=False)
    calib_ids = calib_ids[:, :calib_tokens].to(device)

    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, 'num_key_value_heads', n_heads)
    n_layers = cfg.num_hidden_layers
    d_head = cfg.hidden_size // n_heads

    # Extract post-RoPE K by hooking into the attention module
    # We capture k_proj output and apply RoPE manually to get post-RoPE K
    k_pre_data = {}
    hooks = []
    attn_modules = _find_attn_modules(model)
    for li, (name, attn) in enumerate(attn_modules):
        def make_hook(li_=li):
            def fn(mod, inp, out):
                k_pre_data[li_] = out.detach().cpu().float()
            return fn
        hooks.append(attn.k_proj.register_forward_hook(make_hook()))

    with torch.no_grad():
        model(calib_ids, use_cache=False)
    for h in hooks:
        h.remove()

    # Apply RoPE to get post-RoPE K, then compute PCA
    calib = {}
    seq_len = calib_ids.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    for li, (name, attn) in enumerate(attn_modules):
        # Reshape: (1, T, n_kv*d) -> (1, n_kv, T, d)
        K_raw = k_pre_data[li].to(device)
        B, T, _ = K_raw.shape
        K_raw = K_raw.view(B, T, n_kv, d_head).transpose(1, 2)

        # Apply RoPE
        if hasattr(attn, 'rotary_emb'):
            # Need a dummy value_states for rotary_emb
            dummy_v = torch.zeros(B, n_kv, T, d_head, device=device, dtype=K_raw.dtype)
            cos, sin = attn.rotary_emb(dummy_v, position_ids)
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            # apply_rotary_pos_emb expects Q and K, we only need K
            # Use K as both args, discard first output
            _, K_rope = apply_rotary_pos_emb(K_raw, K_raw, cos, sin)
        else:
            K_rope = K_raw

        K_rope = K_rope.float()  # keep on GPU: (1, n_kv, T, d)

        for hk in range(n_kv):
            K = K_rope[0, hk]  # (T, d) — on GPU
            K_mean = K.mean(0)
            K_c = K - K_mean

            # PCA on GPU
            cov = (K_c.T @ K_c) / max(K.shape[0] - 1, 1)
            cov += torch.eye(d_head, device=K.device) * 1e-8
            evals, evecs = torch.linalg.eigh(cov)
            idx = torch.argsort(evals, descending=True)
            evals = evals[idx]
            evecs = evecs[:, idx]

            # Vectorized quantization noise measurement on GPU
            K_pca = K_c @ evecs  # (T, d)
            n_lev = 2 ** bits
            c_min = K_pca.amin(dim=0)  # (d,)
            c_max = K_pca.amax(dim=0)  # (d,)
            rng = (c_max - c_min).clamp(min=1e-10)
            step = rng / (n_lev - 1)
            K_pca_q = torch.round((K_pca - c_min) / step) * step + c_min
            quant_noise_var = (K_pca - K_pca_q).pow(2).mean(dim=0)  # (d,)

            calib[(li, hk)] = PCACalibration(
                V=evecs.cpu(), evals=evals.cpu(), K_mean=K_mean.cpu(),
                quant_noise_var=quant_noise_var.cpu(),
            )

    del k_pre_data
    gc.collect()
    torch.cuda.empty_cache()
    return calib


# ============================================================================
# Dimension Selection Strategies
# ============================================================================

def select_dims_query_conditional(q_pca: torch.Tensor, evals: torch.Tensor,
                                  quant_noise_var: torch.Tensor, m: int
                                  ) -> torch.Tensor:
    """Select top-m dims by query-conditional scoring.

    score_j = |q_pca_j|^2 * max(lambda_j - sigma2_eps_j, 0)

    Args:
        q_pca: (n_q, d) query vectors in PCA space
        evals: (d,) eigenvalues
        quant_noise_var: (d,) quantization noise variance per dim
        m: number of dims to keep
    Returns:
        mask: (n_q, d) boolean mask, True for selected dims
    """
    if m >= q_pca.shape[-1]:
        return torch.ones(q_pca.shape, dtype=torch.bool, device=q_pca.device)

    # SNR-adjusted eigenvalues: lambda_j - sigma2_eps_j
    snr_adjusted = torch.clamp(evals - quant_noise_var, min=0.0).to(q_pca.device)
    # Per-query scoring: |q_j|^2 * snr_adjusted_j
    scores = q_pca.pow(2) * snr_adjusted.unsqueeze(0)  # (n_q, d)
    # Top-m per query
    _, topk_idx = scores.topk(m, dim=-1)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, topk_idx, True)
    return mask


def select_dims_fixed(evals: torch.Tensor, quant_noise_var: torch.Tensor,
                      m: int, d: int) -> torch.Tensor:
    """Static selection: top-m by lambda_j (same for all queries).

    Returns: (d,) boolean mask
    """
    if m >= d:
        return torch.ones(d, dtype=torch.bool)
    # Since evals are already sorted descending, just take first m
    mask = torch.zeros(d, dtype=torch.bool)
    mask[:m] = True
    return mask


def select_dims_random(m: int, d: int, seed: int = 42) -> torch.Tensor:
    """Random m dims (ablation control).

    Returns: (d,) boolean mask
    """
    if m >= d:
        return torch.ones(d, dtype=torch.bool)
    rng = np.random.RandomState(seed)
    idx = rng.choice(d, m, replace=False)
    mask = torch.zeros(d, dtype=torch.bool)
    mask[torch.from_numpy(idx)] = True
    return mask


def select_dims_oracle(q_pca: torch.Tensor, K_pca_fp16: torch.Tensor,
                       m: int) -> torch.Tensor:
    """Oracle: select dims that maximize |q_j * k_j| using FP16 keys.

    Args:
        q_pca: (n_q, d)
        K_pca_fp16: (n_kv, d) FP16 PCA keys
        m: dims to keep
    Returns:
        mask: (n_q, d)
    """
    if m >= q_pca.shape[-1]:
        return torch.ones(q_pca.shape, dtype=torch.bool, device=q_pca.device)

    # For each query, compute expected contribution of each dim
    # |q_j|^2 * E_tokens[|k_j|^2] using actual FP16 K
    k_var = K_pca_fp16.pow(2).mean(0)  # (d,)
    scores = q_pca.pow(2) * k_var.unsqueeze(0).to(q_pca.device)  # (n_q, d)
    _, topk_idx = scores.topk(m, dim=-1)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, topk_idx, True)
    return mask


# ============================================================================
# Core: Compute Attention with Subspace Selection
# ============================================================================

def compute_attention_with_selection(
    Q: torch.Tensor,          # (batch, n_q_heads, seq_q, d)
    K_fp16: torch.Tensor,     # (batch, n_kv_heads, seq_kv, d)
    V: torch.Tensor,          # (batch, n_kv_heads, seq_kv, d)
    calib_dict: Dict[int, PCACalibration],  # head_idx -> PCACalibration
    method: str,
    bits: int,
    m: int,
    head_dim: int,
    n_kv_heads: int,
    n_q_heads: int,
    causal_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute attention with optional PCA quantization + subspace selection.

    Returns:
        attn_output: (batch, n_q_heads, seq_q, d)
        attn_weights: (batch, n_q_heads, seq_q, seq_kv) — for diagnostics
    """
    B = Q.shape[0]
    seq_q = Q.shape[2]
    seq_kv = K_fp16.shape[2]
    G = n_q_heads // n_kv_heads  # GQA group size

    if method == "fp16":
        # Standard FP16 attention
        K_exp = K_fp16.repeat_interleave(G, dim=1)  # (B, n_q, seq_kv, d)
        V_exp = V.repeat_interleave(G, dim=1)
        logits = torch.matmul(Q, K_exp.transpose(-1, -2)) / math.sqrt(head_dim)
        if causal_mask is not None:
            logits = logits + causal_mask
        weights = F.softmax(logits, dim=-1, dtype=torch.float32).to(Q.dtype)
        output = torch.matmul(weights, V_exp)
        return output, weights.float()

    # For PCA-based methods: per-chunk PCA (matching v3 fokvq protocol)
    all_outputs = []
    all_weights = []

    for hk in range(n_kv_heads):
        cal = calib_dict[hk]
        # Calibration stats used for dimension selection scoring only
        cal_evals = cal.evals.to(Q.device, dtype=torch.float32)
        cal_qnv = cal.quant_noise_var.to(Q.device, dtype=torch.float32)

        K_head = K_fp16[:, hk].float()  # (B, seq_kv, d)
        V_head = V[:, hk]  # (B, seq_kv, d)

        # Per-chunk PCA (like v3 fokvq_quantize_head): compute PCA on this chunk's K
        # This matches the existing baseline protocol where PPL=6.7
        K_flat = K_head.reshape(-1, head_dim)  # (B*seq_kv, d)
        K_mean_chunk = K_flat.mean(0)
        K_c = K_flat - K_mean_chunk
        cov = (K_c.T @ K_c) / max(K_c.shape[0] - 1, 1)
        cov += torch.eye(head_dim, device=Q.device) * 1e-8
        chunk_evals, chunk_evecs = torch.linalg.eigh(cov)
        idx = torch.argsort(chunk_evals, descending=True)
        chunk_evals = chunk_evals[idx]
        chunk_evecs = chunk_evecs[:, idx]

        # PCA transform with per-chunk basis
        K_centered = K_head - K_mean_chunk.unsqueeze(0).unsqueeze(0)
        K_pca = K_centered @ chunk_evecs  # (B, seq_kv, d)

        # Quantize in PCA space
        if "fp16_topk" in method:
            K_pca_q = K_pca
        else:
            n_lev = 2 ** bits
            c_min = K_pca.amin(dim=1, keepdim=True)
            c_max = K_pca.amax(dim=1, keepdim=True)
            rng = (c_max - c_min).clamp(min=1e-10)
            step = rng / (n_lev - 1)
            K_pca_q = torch.round((K_pca - c_min) / step) * step + c_min

        # Measure per-chunk quant noise for selection scoring
        chunk_qnv = (K_pca - K_pca_q).pow(2).mean(dim=(0, 1))  # (d,)

        # For pca_all: reconstruct K in original space, standard attention
        if method == "pca_all":
            K_recon = K_pca_q @ chunk_evecs.T + K_mean_chunk.unsqueeze(0).unsqueeze(0)
            for g in range(G):
                qh = hk * G + g
                logits = torch.bmm(Q[:, qh].float(), K_recon.transpose(1, 2)) / math.sqrt(head_dim)
                if causal_mask is not None:
                    logits = logits + causal_mask[:, qh, :seq_q, :seq_kv]
                weights = F.softmax(logits, dim=-1, dtype=torch.float32)
                out = torch.bmm(weights.to(V_head.dtype), V_head)
                all_outputs.append(out)
                all_weights.append(weights)
            continue

        # For topk methods: attention in PCA space with dimension selection
        for g in range(G):
            qh = hk * G + g
            Q_head_g = Q[:, qh].float()
            Q_pca = Q_head_g @ chunk_evecs  # (B, seq_q, d)

            if method == "pca_topk_fixed":
                fmask = select_dims_fixed(chunk_evals, chunk_qnv, m, head_dim).to(Q.device).float()
                Q_masked = Q_pca * fmask.unsqueeze(0).unsqueeze(0)
                logits = torch.bmm(Q_masked, K_pca_q.transpose(1, 2)) / math.sqrt(head_dim)
            elif method == "pca_topk_random":
                rmask = select_dims_random(m, head_dim, seed=42 + hk).to(Q.device).float()
                Q_masked = Q_pca * rmask.unsqueeze(0).unsqueeze(0)
                logits = torch.bmm(Q_masked, K_pca_q.transpose(1, 2)) / math.sqrt(head_dim)
            elif method in ("pca_topk_query", "fp16_topk_query"):
                logits_list = []
                for b in range(B):
                    mask = select_dims_query_conditional(Q_pca[b], chunk_evals, chunk_qnv, m)
                    Q_masked = Q_pca[b] * mask.float()
                    logits_b = (Q_masked @ K_pca_q[b].T) / math.sqrt(head_dim)
                    logits_list.append(logits_b)
                logits = torch.stack(logits_list)
            elif method == "pca_topk_oracle":
                logits_list = []
                for b in range(B):
                    mask = select_dims_oracle(Q_pca[b], K_pca[b], m)
                    Q_masked = Q_pca[b] * mask.float()
                    logits_b = (Q_masked @ K_pca_q[b].T) / math.sqrt(head_dim)
                    logits_list.append(logits_b)
                logits = torch.stack(logits_list)
            else:
                raise ValueError(f"Unknown method: {method}")

            if causal_mask is not None:
                logits = logits + causal_mask[:, qh, :seq_q, :seq_kv]

            weights = F.softmax(logits, dim=-1, dtype=torch.float32)
            out = torch.bmm(weights.to(V_head.dtype), V_head)
            all_outputs.append(out)
            all_weights.append(weights)

    # Stack: (B, n_q_heads, seq_q, d) and (B, n_q_heads, seq_q, seq_kv)
    output = torch.stack(all_outputs, dim=1).to(Q.dtype)
    weights = torch.stack(all_weights, dim=1)
    return output, weights


# ============================================================================
# Microbenchmark: Attention Score MSE
# ============================================================================

def run_microbenchmark(model, tokenizer, calib: Dict, device: str,
                       bits: int, max_tokens: int = 10000) -> Dict:
    """Measure attention score MSE for each method × m.

    For efficiency, processes 2 chunks from eval data, measuring
    per-layer attention weight MSE vs FP16 reference.
    """
    from datasets import load_dataset
    eval_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_text = "\n\n".join([t for t in eval_ds["text"] if t.strip()])
    eval_ids = tokenizer.encode(eval_text, return_tensors="pt", truncation=False)
    eval_ids = eval_ids[:, :max_tokens].to(device)

    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, 'num_key_value_heads', n_heads)
    n_layers = cfg.num_hidden_layers
    d_head = cfg.hidden_size // n_heads

    chunk_len = min(EVAL_CHUNK, eval_ids.shape[1])
    n_chunks = min(eval_ids.shape[1] // chunk_len, 5)  # up to 5 chunks for microbench

    methods = ["pca_all", "pca_topk_query", "pca_topk_fixed",
               "pca_topk_random", "fp16_topk_query"]

    # Ensure pca_all runs with m=128 (all dims) regardless of M_SWEEP
    m_values_with_baseline = sorted(set(M_SWEEP) | {128})

    results = {}

    for method in methods:
        for m in m_values_with_baseline:
            if method == "pca_all" and m != 128:
                continue  # pca_all always uses all 128 dims
            if method != "pca_all" and m == 128:
                continue  # topk methods with m=128 = pca_all (redundant)

            key = f"{method}_m{m}" if method != "pca_all" else "pca_all"
            if key in results:
                continue

            print(f"  [{key}] measuring attention MSE...")
            attn_mse_per_layer = [[] for _ in range(n_layers)]
            signed_bias_per_layer = [[] for _ in range(n_layers)]
            entropy_shift_per_layer = [[] for _ in range(n_layers)]

            for chunk_idx in range(n_chunks):
                chunk = eval_ids[:, chunk_idx*chunk_len:(chunk_idx+1)*chunk_len]

                # Hook into q_proj, k_proj, v_proj to get pre-RoPE states
                k_pre = {}
                q_pre = {}
                v_pre = {}
                hook_handles = []
                attn_modules = _find_attn_modules(model)

                for li, (name, attn) in enumerate(attn_modules):
                    def kh(li_=li):
                        def fn(mod, inp, out):
                            k_pre[li_] = out.detach()
                        return fn

                    def qh(li_=li):
                        def fn(mod, inp, out):
                            q_pre[li_] = out.detach()
                        return fn

                    def vh(li_=li):
                        def fn(mod, inp, out):
                            v_pre[li_] = out.detach()
                        return fn

                    hook_handles.append(attn.k_proj.register_forward_hook(kh()))
                    hook_handles.append(attn.q_proj.register_forward_hook(qh()))
                    hook_handles.append(attn.v_proj.register_forward_hook(vh()))

                with torch.no_grad():
                    model(chunk, use_cache=False)

                for h in hook_handles:
                    h.remove()

                # Step 2: For each layer, apply RoPE and compute attention
                for li, (name, attn) in enumerate(attn_modules):
                    B, S = chunk.shape[0], chunk.shape[1]

                    Q_raw = q_pre[li].view(B, S, n_heads, d_head).transpose(1, 2).float()
                    K_raw = k_pre[li].view(B, S, n_kv, d_head).transpose(1, 2).float()
                    V_raw = v_pre[li].view(B, S, n_kv, d_head).transpose(1, 2)

                    # Apply RoPE
                    position_ids = torch.arange(S, device=device).unsqueeze(0)
                    if hasattr(attn, 'rotary_emb'):
                        cos, sin = attn.rotary_emb(V_raw, position_ids)
                        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                        Q_rope, K_rope = apply_rotary_pos_emb(
                            Q_raw.to(cos.dtype), K_raw.to(cos.dtype), cos, sin)
                        Q_rope = Q_rope.float()
                        K_rope = K_rope.float()
                    else:
                        Q_rope = Q_raw
                        K_rope = K_raw

                    # Causal mask
                    causal = torch.triu(
                        torch.full((S, S), float('-inf'), device=device), diagonal=1
                    ).unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

                    # FP16 reference attention weights
                    K_rope_exp = K_rope.repeat_interleave(n_heads // n_kv, dim=1)
                    ref_logits = torch.matmul(Q_rope, K_rope_exp.transpose(-1, -2)) / math.sqrt(d_head)
                    ref_logits = ref_logits + causal
                    ref_weights = F.softmax(ref_logits, dim=-1, dtype=torch.float32)

                    # Layer-specific calib
                    layer_calib = {hk: calib[(li, hk)] for hk in range(n_kv)}

                    # Method attention weights
                    _, method_weights = compute_attention_with_selection(
                        Q_rope, K_rope, V_raw, layer_calib,
                        method=method, bits=bits, m=m,
                        head_dim=d_head, n_kv_heads=n_kv, n_q_heads=n_heads,
                        causal_mask=causal.expand(B, n_heads, S, S),
                    )

                    # Attention MSE (averaged over heads and positions)
                    mse = (ref_weights - method_weights).pow(2).mean().item()
                    attn_mse_per_layer[li].append(mse)

                    # Signed logit bias: mean(method_logit - ref_logit)
                    # (recompute method logits... simplified: use weight entropy)
                    ref_ent = -(ref_weights * (ref_weights + 1e-10).log()).sum(-1).mean().item()
                    meth_ent = -(method_weights * (method_weights + 1e-10).log()).sum(-1).mean().item()
                    entropy_shift_per_layer[li].append(meth_ent - ref_ent)

                    del Q_raw, K_raw, V_raw, Q_rope, K_rope
                    gc.collect()

                del k_pre, q_pre, v_pre
                gc.collect()
                torch.cuda.empty_cache()

            # Aggregate
            layer_mses = [np.mean(m_list) if m_list else 0.0 for m_list in attn_mse_per_layer]
            layer_ent = [np.mean(e) if e else 0.0 for e in entropy_shift_per_layer]

            results[key] = {
                "method": method,
                "m": m if method != "pca_all" else d_head,
                "attn_mse_mean": float(np.mean(layer_mses)),
                "attn_mse_per_layer": [round(x, 8) for x in layer_mses],
                "entropy_shift_mean": float(np.mean(layer_ent)),
                "entropy_shift_per_layer": [round(x, 6) for x in layer_ent],
            }
            print(f"    attn_MSE={results[key]['attn_mse_mean']:.6e}  "
                  f"entropy_shift={results[key]['entropy_shift_mean']:.4f}")

    return results


# ============================================================================
# Full PPL Evaluation with Query-Conditional Selection
# ============================================================================

class QueryConditionalAttentionPatcher:
    """Patches attention for query-conditional PCA subspace selection.

    Similar to AttentionKQuantPatcher but modifies the attention computation
    itself, not just the K tensor. This is necessary because the dimension
    selection depends on the runtime query.
    """

    def __init__(self, model, calib: Dict[Tuple[int, int], PCACalibration],
                 method: str, bits: int, m: int):
        self.model = model
        self.calib = calib
        self.method = method
        self.bits = bits
        self.m = m
        self.active = False
        self.original_forwards = {}
        self._patched = False

        cfg = model.config
        self.n_heads = cfg.num_attention_heads
        self.n_kv = getattr(cfg, 'num_key_value_heads', self.n_heads)
        self.n_layers = cfg.num_hidden_layers
        self.d_head = cfg.hidden_size // self.n_heads
        self.G = self.n_heads // self.n_kv

    def patch(self):
        if self._patched:
            return
        attn_modules = []
        for name, module in self.model.named_modules():
            cls_name = type(module).__name__
            if 'Attention' in cls_name and hasattr(module, 'k_proj'):
                attn_modules.append((name, module))

        for i, (name, attn_module) in enumerate(attn_modules):
            orig_forward = attn_module.forward
            self.original_forwards[name] = orig_forward
            layer_idx = i
            patched = self._make_patched_forward(attn_module, orig_forward, layer_idx)
            attn_module.forward = patched

        self._patched = True
        print(f"  Patched {len(attn_modules)} layers "
              f"(method={self.method}, bits={self.bits}, m={self.m})")

    def unpatch(self):
        if not self._patched:
            return
        for name, module in self.model.named_modules():
            if name in self.original_forwards:
                module.forward = self.original_forwards[name]
        self.original_forwards.clear()
        self._patched = False

    def _make_patched_forward(self, attn_module, orig_forward, layer_idx):
        patcher = self

        def patched_forward(hidden_states, attention_mask=None,
                            position_ids=None, past_key_value=None,
                            output_attentions=False, use_cache=False,
                            cache_position=None, position_embeddings=None,
                            **kwargs):
            if not patcher.active:
                return orig_forward(
                    hidden_states, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_value=past_key_value,
                    output_attentions=output_attentions, use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings, **kwargs)

            bsz, q_len, _ = hidden_states.size()

            query_states = attn_module.q_proj(hidden_states)
            key_states = attn_module.k_proj(hidden_states)
            value_states = attn_module.v_proj(hidden_states)

            num_heads = getattr(attn_module, 'num_heads',
                                attn_module.config.num_attention_heads)
            num_kv_heads = getattr(attn_module, 'num_key_value_heads',
                                   attn_module.config.num_key_value_heads)
            head_dim = attn_module.head_dim

            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

            # Apply RoPE
            if position_embeddings is not None:
                cos, sin = position_embeddings
            elif hasattr(attn_module, 'rotary_emb'):
                if position_ids is not None:
                    cos, sin = attn_module.rotary_emb(value_states, position_ids)
                else:
                    cos, sin = attn_module.rotary_emb(value_states, seq_len=q_len)
            else:
                cos, sin = None, None

            if cos is not None and sin is not None:
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin)

            # Build causal mask
            causal = torch.triu(
                torch.full((q_len, q_len), float('-inf'),
                           device=hidden_states.device, dtype=torch.float32),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)

            layer_calib = {hk: patcher.calib[(layer_idx, hk)] for hk in range(num_kv_heads)}

            attn_output, attn_weights = compute_attention_with_selection(
                query_states.float(), key_states.float(), value_states,
                layer_calib,
                method=patcher.method, bits=patcher.bits, m=patcher.m,
                head_dim=head_dim, n_kv_heads=num_kv_heads, n_q_heads=num_heads,
                causal_mask=causal.expand(bsz, num_heads, q_len, q_len),
            )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)
            attn_output = attn_module.o_proj(attn_output.to(hidden_states.dtype))

            return attn_output, attn_weights if output_attentions else None

        return patched_forward


def run_ppl_eval(model, tokenizer, calib: Dict, device: str,
                 method: str, bits: int, m: int) -> Dict:
    """Run full PPL evaluation with query-conditional selection."""
    from datasets import load_dataset
    eval_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_text = "\n\n".join([t for t in eval_ds["text"] if t.strip()])
    eval_ids = tokenizer.encode(eval_text, return_tensors="pt", truncation=False)

    total_len = min(eval_ids.shape[1], MAX_EVAL)
    n_chunks = total_len // EVAL_CHUNK
    eval_ids = eval_ids[:, :n_chunks * EVAL_CHUNK]

    if method == "fp16":
        # Baseline: no patching
        model.eval()
        total_nll = 0.0
        total_count = 0
        t0 = time.time()
        for i in range(n_chunks):
            chunk = eval_ids[:, i*EVAL_CHUNK:(i+1)*EVAL_CHUNK].to(device)
            with torch.no_grad():
                out = model(chunk, use_cache=False)
            logits = out.logits.float()
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), reduction='sum')
            total_nll += loss.item()
            total_count += shift_labels.numel()
            if (i + 1) % 5 == 0:
                ppl_running = math.exp(min(total_nll / max(total_count, 1), 100))
                print(f"    chunk {i+1}/{n_chunks} ppl={ppl_running:.2f}")
        ppl = math.exp(total_nll / max(total_count, 1))
        return {"ppl": round(ppl, 4), "runtime_s": round(time.time() - t0, 1)}

    # Patched evaluation
    patcher = QueryConditionalAttentionPatcher(model, calib, method, bits, m)
    patcher.patch()
    patcher.active = True

    model.eval()
    total_nll = 0.0
    total_count = 0
    t0 = time.time()

    for i in range(n_chunks):
        chunk = eval_ids[:, i*EVAL_CHUNK:(i+1)*EVAL_CHUNK].to(device)
        with torch.no_grad():
            out = model(chunk, use_cache=False)
        logits = out.logits.float()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), reduction='sum')
        total_nll += loss.item()
        total_count += shift_labels.numel()
        if (i + 1) % 5 == 0:
            ppl_running = math.exp(min(total_nll / max(total_count, 1), 100))
            elapsed = time.time() - t0
            print(f"    chunk {i+1}/{n_chunks} ppl={ppl_running:.2f} ({elapsed:.0f}s)")

    patcher.active = False
    patcher.unpatch()

    ppl = math.exp(total_nll / max(total_count, 1))
    return {"ppl": round(ppl, 4), "runtime_s": round(time.time() - t0, 1)}


# ============================================================================
# Main
# ============================================================================

def main():
    global MAX_EVAL, M_SWEEP

    parser = argparse.ArgumentParser(description="Query-Conditional PCA Subspace Selection")
    parser.add_argument("--model", required=True, help="HF model name")
    parser.add_argument("--bits", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--mode", default="microbench", choices=["microbench", "ppl", "both"])
    parser.add_argument("--m-values", nargs="+", type=int, default=None,
                        help="Override M_SWEEP values")
    parser.add_argument("--output-dir", default="results/query_conditional")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--max-eval-tokens", type=int, default=MAX_EVAL)
    args = parser.parse_args()

    MAX_EVAL = args.max_eval_tokens
    if args.m_values:
        M_SWEEP = args.m_values

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    short = args.model.split("/")[-1].replace(".", "_")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"{'='*60}")
    print(f"QUERY-CONDITIONAL PCA: {args.model}, {args.bits}-bit")
    print(f"Mode: {args.mode}, m_values: {M_SWEEP}")
    print(f"{'='*60}")

    tok_kw = {"trust_remote_code": True}
    mdl_kw = {"torch_dtype": DTYPE, "trust_remote_code": True,
              "attn_implementation": "eager"}
    if args.hf_token:
        tok_kw["token"] = args.hf_token
        mdl_kw["token"] = args.hf_token

    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kw)
    model = AutoModelForCausalLM.from_pretrained(args.model, **mdl_kw).to(device).eval()

    print("\n[Calibrating PCA...]")
    t0 = time.time()
    calib = calibrate_pca(model, tokenizer, device, args.bits)
    print(f"  Calibrated {len(calib)} heads in {time.time()-t0:.1f}s")

    # Print PCA stats
    sample_key = list(calib.keys())[0]
    sample = calib[sample_key]
    snr = sample.evals / (sample.quant_noise_var + 1e-10)
    n_snr_gt1 = int((snr > 1).sum())
    print(f"  Sample head ({sample_key}): "
          f"top5 eigenvalues={sample.evals[:5].numpy().round(3)}, "
          f"dims with SNR>1: {n_snr_gt1}/{sample.evals.shape[0]}")

    results = {
        "model": args.model,
        "bits": args.bits,
        "mode": args.mode,
        "m_values": M_SWEEP,
        "n_heads": len(calib),
    }

    if args.mode in ("microbench", "both"):
        print(f"\n{'='*60}")
        print(f"MICROBENCHMARK: Attention Score MSE")
        print(f"{'='*60}")
        mb_results = run_microbenchmark(model, tokenizer, calib, device, args.bits)
        results["microbench"] = mb_results

        # Summary table
        print(f"\n{'Method':<25s} {'m':>4s} {'Attn MSE':>12s} {'Ent Shift':>10s}")
        for key in sorted(mb_results.keys()):
            r = mb_results[key]
            print(f"{key:<25s} {r['m']:>4d} {r['attn_mse_mean']:>12.6e} "
                  f"{r['entropy_shift_mean']:>10.4f}")

    if args.mode in ("ppl", "both"):
        print(f"\n{'='*60}")
        print(f"PPL EVALUATION")
        print(f"{'='*60}")

        ppl_results = {}

        # FP16 baseline
        print("\n[FP16 baseline]")
        ppl_results["fp16"] = run_ppl_eval(model, tokenizer, calib, device,
                                            "fp16", args.bits, 128)
        print(f"  PPL = {ppl_results['fp16']['ppl']}")

        # PCA all dims
        print(f"\n[pca_all {args.bits}-bit]")
        ppl_results["pca_all"] = run_ppl_eval(model, tokenizer, calib, device,
                                               "pca_all", args.bits, 128)
        print(f"  PPL = {ppl_results['pca_all']['ppl']}")

        # Query-conditional at best m from microbench (or sweep)
        for m in M_SWEEP:
            if m == 128:
                continue  # already covered by pca_all
            for method in ["pca_topk_query", "pca_topk_fixed"]:
                key = f"{method}_m{m}"
                print(f"\n[{key} {args.bits}-bit]")
                ppl_results[key] = run_ppl_eval(
                    model, tokenizer, calib, device, method, args.bits, m)
                print(f"  PPL = {ppl_results[key]['ppl']}")

        results["ppl"] = ppl_results

        # Summary table
        print(f"\n{'Method':<30s} {'PPL':>8s} {'Time':>8s}")
        for key in sorted(ppl_results.keys()):
            r = ppl_results[key]
            print(f"{key:<30s} {r['ppl']:>8.4f} {r['runtime_s']:>8.1f}s")

    # Save
    out_path = out_dir / f"{short}_{args.bits}b_query_conditional.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
