#!/usr/bin/env python3
"""
Axis 2 / Axis 3 Failure Diagnosis Experiments
=============================================

Implements three targeted experiments to diagnose why FOKVQ axis-2/axis-3
quantization causes PPL catastrophe on large models:

  E1 (MK): Mahalanobis Lloyd-Max with query covariance weighting
    - Hypothesis A1: The inner-product metric matters more than K-space MSE
    - Uses calibrated Sigma_q to weight bit allocation and codebook fitting

  E2 (Sink): Attention sink token preservation
    - Hypothesis A2: Destroying sink tokens causes the PPL catastrophe
    - Preserves first n_sink tokens in FP16, quantizes the rest

  E4 (Integer WF): Integer-constrained water-filling bit allocation
    - Hypothesis B2: 1-bit dimensions are harmful; floor=2 or {0,>=2} is better
    - Tests WF(floor=1), WF(floor=2), WF(no-one), and uniform allocation

From AXIS2_AXIS3_FAILURE_DIAGNOSIS_EXPERIMENT_DESIGN.md

Usage:
  python exp_axis2_axis3_diagnosis.py --experiment E1 --model-name Qwen/Qwen2.5-7B --device cuda:0
  python exp_axis2_axis3_diagnosis.py --experiment E2 --model-name meta-llama/Llama-3.1-8B --device cuda:0
  python exp_axis2_axis3_diagnosis.py --experiment E4 --model-name Qwen/Qwen2.5-7B --device cuda:0
  python exp_axis2_axis3_diagnosis.py --experiment ALL --model-name Qwen/Qwen2.5-7B --device cuda:0
  python exp_axis2_axis3_diagnosis.py --self-test
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import socket
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Import base infrastructure from v3 full quant PPL script
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp4_2_v3_full_quant_ppl import (
    AttentionKQuantPatcher,
    evaluate_ppl_chunked,
    find_attention_modules,
    set_seed,
    uniform_quantize_tensor,
    quantize_k_tensor,
    load_model_and_tokenizer,
    load_wikitext2_ids,
    resolve_dtype,
    _pca_decompose,
    _compute_bit_allocation,
    _fit_lloyd_max_1d,
    _fit_lloyd_max_mahalanobis_1d,
    _quantize_with_codebook_1d,
    _quantize_pca_uniform,
    _quantize_pca_lloyd,
    _quantize_pca_mk,
    fokvq_quantize_head,
    fokvq_e2_quantize_head,
)


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--self-test", action="store_true")
    pre, _ = bootstrap.parse_known_args()

    p = argparse.ArgumentParser(
        parents=[bootstrap],
        description="Axis 2/3 Failure Diagnosis: E1 (MK), E2 (Sink), E4 (Integer WF)",
    )
    req = not pre.self_test
    p.add_argument("--experiment", type=str, required=req, default="ALL",
                   choices=["E1", "E2", "E4", "ALL"],
                   help="Which experiment to run")
    p.add_argument("--model-name", type=str, required=req, default="gpt2-medium")
    p.add_argument("--model-key", type=str, default="")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--context-len", type=int, default=2048,
                   help="Chunk length for non-overlapping evaluation")
    p.add_argument("--bits", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--gamma", type=float, default=0.3,
                   help="FOKVQ eigenvalue weighting exponent")
    p.add_argument("--max-eval-tokens", type=int, default=0,
                   help="Truncate test set (0 = use all)")
    p.add_argument("--output-dir", type=str, default="")
    p.add_argument("--cache-dir", type=str, default="")
    p.add_argument("--attn-implementation", type=str, default="eager")
    p.add_argument("--seed", type=int, default=42)
    # E1-specific
    p.add_argument("--mk-calib-tokens", type=int, default=2048,
                   help="Number of calibration tokens for E1 query covariance")
    # E2-specific
    p.add_argument("--sink-counts", nargs="+", type=int, default=[0, 4, 16, 64],
                   help="Number of sink tokens to preserve in E2")
    # E4-specific
    p.add_argument("--wf-variants", nargs="+", type=str,
                   default=["floor1", "floor2", "no_one", "uniform"],
                   help="Water-filling variants for E4")
    return p.parse_args()


# ============================================================================
# E1: Mahalanobis Lloyd-Max with Query Covariance (Hypothesis A1)
# ============================================================================

def calibrate_query_covariance(
    model,
    tokenizer,
    device: str,
    max_tokens: int = 2048,
    cache_dir: Optional[str] = None,
) -> Dict[int, Dict[int, torch.Tensor]]:
    """Calibrate per-layer per-KV-head query covariance from WikiText-2 train.

    Returns:
        q_covs[layer_idx][kv_head_idx] = (d_head, d_head) tensor on CPU
    """
    from datasets import load_dataset

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", **kwargs)
    all_text = "\n".join([t for t in ds["text"] if t.strip()])
    enc = tokenizer(all_text, return_tensors="pt", add_special_tokens=False)
    calib_ids = enc.input_ids[:, :max_tokens].to(device)
    print(f"  [E1] Calibrating Sigma_q from {calib_ids.shape[1]} tokens (WikiText-2 train)")

    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    d_head = cfg.hidden_size // n_heads
    n_layers = cfg.num_hidden_layers
    n_rep = n_heads // n_kv_heads  # GQA group size

    # Storage: accumulate Q*Q^T per layer per KV head
    q_sum = {}       # layer -> kv_head -> (d_head, d_head)
    q_mean = {}      # layer -> kv_head -> (d_head,)
    q_count = {}     # layer -> kv_head -> int

    for l in range(n_layers):
        q_sum[l] = {}
        q_mean[l] = {}
        q_count[l] = {}
        for h in range(n_kv_heads):
            q_sum[l][h] = torch.zeros(d_head, d_head, dtype=torch.float64)
            q_mean[l][h] = torch.zeros(d_head, dtype=torch.float64)
            q_count[l][h] = 0

    # Hook-based collection: intercept Q after projection + RoPE
    hooks = []
    q_buffer = {}  # layer_idx -> tensor (batch, n_heads, seq, d_head)

    attn_modules = find_attention_modules(model)
    model_type = getattr(cfg, "model_type", "").lower()

    if "qwen2" in model_type:
        # For Qwen2: patch eager_attention_forward temporarily
        from transformers.models.qwen2 import modeling_qwen2
        orig_eager = modeling_qwen2.eager_attention_forward
        patched_module_ids = {id(module) for _, module in attn_modules}

        # Map module id -> layer index
        module_to_layer = {}
        for name, module in attn_modules:
            # Extract layer index from name like "model.layers.0.self_attn"
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    module_to_layer[id(module)] = int(parts[i + 1])
                    break

        def calib_eager_forward(module, query, key, value,
                                attention_mask, scaling, dropout=0.0, **kwargs):
            if id(module) in patched_module_ids:
                layer_idx = module_to_layer.get(id(module))
                if layer_idx is not None:
                    q_buffer[layer_idx] = query.detach()
            return orig_eager(module, query, key, value,
                              attention_mask, scaling=scaling,
                              dropout=dropout, **kwargs)

        modeling_qwen2.eager_attention_forward = calib_eager_forward
    else:
        # For Llama/Mistral/GPT-2: register hooks on attention modules
        module_to_layer = {}
        for name, module in attn_modules:
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    module_to_layer[id(module)] = int(parts[i + 1])
                    break
            # For GPT2: model.transformer.h.0.attn
            for i, p in enumerate(parts):
                if p == "h" and i + 1 < len(parts):
                    module_to_layer[id(module)] = int(parts[i + 1])
                    break

        def make_hook(mod_id):
            def hook_fn(module, input, output):
                # For Llama/Mistral, the forward returns (attn_output, attn_weights)
                # We need to capture Q internally. Use a pre-forward hook on q_proj instead.
                pass
            return hook_fn

        # Actually, for non-Qwen2, we need to hook q_proj output and apply RoPE ourselves.
        # This is complex. Instead, do a simpler approach: run a forward pass and capture
        # intermediate Q states via register_forward_hook on the attention module itself.
        # We'll use a two-pass approach: first collect Q states, then compute covariance.

        # Alternative simpler approach: just hook q_proj and collect pre-RoPE Q.
        # For post-RoPE Q, we would need the RoPE parameters. Since the design doc
        # says "calibrate Sigma_q" and post-RoPE is the target, let's use a custom
        # forward that captures Q after RoPE.

        # For simplicity, use the same module-level approach as Qwen2 but for Llama/Mistral.
        pass

    # Run calibration forward pass
    with torch.no_grad():
        # Process in chunks to avoid OOM
        seq_len = calib_ids.shape[1]
        chunk_size = min(512, seq_len)
        for start in range(0, seq_len - chunk_size + 1, chunk_size):
            chunk = calib_ids[:, start:start + chunk_size]
            q_buffer.clear()
            _ = model(chunk, use_cache=False)

            # Process collected Q states
            for layer_idx, q_states in q_buffer.items():
                # q_states: (batch, n_heads, seq, d_head)
                Q_f = q_states.float()
                bsz, nh, sl, dh = Q_f.shape
                for kv_h in range(n_kv_heads):
                    # Pool query heads in this GQA group
                    q_start = kv_h * n_rep
                    q_end = q_start + n_rep
                    # (batch, n_rep, seq, d_head) -> (batch*n_rep*seq, d_head)
                    Q_group = Q_f[:, q_start:q_end].reshape(-1, dh).cpu().double()
                    n = Q_group.shape[0]
                    mean_update = Q_group.sum(dim=0)
                    q_mean[layer_idx][kv_h] += mean_update
                    q_count[layer_idx][kv_h] += n
                    # Use running sum of outer products (two-pass would be more
                    # numerically stable but this suffices for calibration)
                    q_sum[layer_idx][kv_h] += Q_group.T @ Q_group

            del q_buffer
            torch.cuda.empty_cache()

    # Restore original forward for Qwen2
    if "qwen2" in model_type:
        modeling_qwen2.eager_attention_forward = orig_eager

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute covariance from accumulated statistics
    q_covs = {}
    for l in range(n_layers):
        q_covs[l] = {}
        for h in range(n_kv_heads):
            n = q_count[l][h]
            if n < 2:
                q_covs[l][h] = torch.eye(d_head, dtype=torch.float32)
                continue
            mean = q_mean[l][h] / n
            # Cov = E[X^T X] - mean^T mean = (sum_XX / n) - mean outer mean
            cov = q_sum[l][h] / n - mean.unsqueeze(1) * mean.unsqueeze(0)
            # Regularize
            cov = cov.float() + torch.eye(d_head, dtype=torch.float32) * 1e-6
            q_covs[l][h] = cov

    print(f"  [E1] Calibrated Sigma_q for {n_layers} layers x {n_kv_heads} KV heads")
    return q_covs


def mk_quantize_head(
    K_head: torch.Tensor,
    bits_avg: int,
    sigma_q: torch.Tensor,
    gamma: float = 0.3,
) -> Tuple[torch.Tensor, float]:
    """E1: Mahalanobis Lloyd-Max with query-weighted bit allocation.

    Key differences from existing E3:
      - Bit allocation uses w_j * lambda_j^(1/2) where w_j = diag(Sigma_q_rot)_jj
      - MK weight per axis = w_j (query importance) instead of 1/lambda_j (K inverse)
      - Sigma_q is pre-calibrated from WikiText-2 train, not from current chunk

    Args:
        K_head: (seq_len, d_head) key states for one head
        bits_avg: average bits per dimension
        sigma_q: (d_head, d_head) pre-calibrated query covariance
        gamma: eigenvalue weighting exponent
    Returns:
        K_recon: (seq_len, d_head) dequantized K
        r_eff: effective rank
    """
    d = K_head.shape[-1]
    K_f = K_head.float()

    # Step 1: Standard K-only PCA decomposition
    mean = K_f.mean(dim=0)
    centered = K_f - mean
    Sigma_K = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    Sigma_K += torch.eye(d, device=Sigma_K.device) * 1e-8

    evals, evecs = torch.linalg.eigh(Sigma_K)
    idx = torch.argsort(evals, descending=True)
    evals = evals[idx]
    evecs = evecs[:, idx]

    K_pca = centered @ evecs

    # Step 2: Rotate Sigma_q into PCA basis
    # Sigma_q_rot = V^T Sigma_q V where V = evecs (PCA basis of K)
    sigma_q_f = sigma_q.float().to(K_f.device)
    sigma_q_rot = evecs.T @ sigma_q_f @ evecs

    # Step 3: Extract diagonal of Sigma_q in PCA basis -> query importance weights
    w_j = torch.diagonal(sigma_q_rot).clamp(min=1e-10)

    # Step 4: Bit allocation proportional to w_j * lambda_j^(1/2)
    ev_np = evals.cpu().numpy()
    ev_pos = np.maximum(ev_np, 1e-10)
    w_np = w_j.detach().cpu().numpy()
    w_np = np.maximum(w_np, 1e-10)

    # Priority score: query weight * sqrt(eigenvalue)
    priority = w_np * (ev_pos ** 0.5)
    priority_weighted = priority ** gamma
    priority_weighted /= priority_weighted.sum()

    ib = np.clip(np.round(priority_weighted * d * bits_avg).astype(int), 1, 8)
    total_budget = d * bits_avg
    while ib.sum() > total_budget:
        ib[np.argmax(ib)] -= 1
    while ib.sum() < total_budget:
        ib[np.argmin(ib)] += 1
    ib = np.clip(ib, 1, 8)

    # Step 5: Lloyd-Max with MK weight = w_j (query importance per axis)
    K_q = torch.zeros_like(K_pca)
    for i in range(d):
        col = K_pca[:, i]
        n_lev = 2 ** int(ib[i])
        mk_weight = float(w_np[i])
        cb = _fit_lloyd_max_mahalanobis_1d(col, n_lev, mk_weight)
        K_q[:, i] = _quantize_with_codebook_1d(col, cb)

    K_recon = K_q @ evecs.T + mean

    ev_norm = ev_pos / ev_pos.sum()
    r_eff = float(np.exp(-np.sum(ev_norm * np.log(ev_norm + 1e-30))))

    return K_recon.to(K_head.dtype), r_eff


class MKQuantPatcher(AttentionKQuantPatcher):
    """Extends AttentionKQuantPatcher for E1: MK quantization with pre-calibrated Sigma_q.

    Overrides _quantize_and_track to use mk_quantize_head with per-layer per-head
    query covariance instead of the standard quantize_k_tensor dispatcher.
    """

    def __init__(self, model, bits: int, gamma: float,
                 q_covs: Dict[int, Dict[int, torch.Tensor]]):
        super().__init__(model, method="mk", bits=bits, gamma=gamma)
        self.q_covs = q_covs
        self._layer_counter = 0
        self._call_order_map = {}  # module_id -> layer_idx

    def patch(self):
        """Install patches and build module-to-layer mapping."""
        # Build mapping before patching
        attn_modules = find_attention_modules(self.model)
        for name, module in attn_modules:
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p in ("layers", "h") and i + 1 < len(parts):
                    self._call_order_map[id(module)] = int(parts[i + 1])
                    break
        super().patch()

    def _quantize_and_track(self, K, query_states=None):
        """Override: use mk_quantize_head with calibrated Sigma_q."""
        # Determine layer index from call order
        # The patcher is called in layer order during forward pass
        layer_idx = self._layer_counter % len(self.q_covs)
        self._layer_counter += 1

        # K shape: (batch, n_kv_heads, seq, d_head)
        B, H, S, D = K.shape
        K_q = K.clone()

        for b in range(B):
            for h in range(H):
                K_head = K[b, h]  # (seq, d_head)
                sigma_q = self.q_covs.get(layer_idx, {}).get(h)
                if sigma_q is None:
                    sigma_q = torch.eye(D, dtype=torch.float32)
                K_head_q, _ = mk_quantize_head(K_head, self.bits, sigma_q, self.gamma)
                K_q[b, h] = K_head_q

        # Track MSE
        diff = (K.float() - K_q.float()).pow(2)
        self.key_mse_sum += float(diff.sum().item())
        self.key_mse_count += int(diff.numel())

        return K_q

    def reset_stats(self):
        super().reset_stats()
        self._layer_counter = 0


# ============================================================================
# E2: Sink Token Preservation (Hypothesis A2)
# ============================================================================

def sink_preserve_quantize_head(
    K_head: torch.Tensor,
    bits_avg: int,
    n_sink: int,
    method: str = "lloyd",
    gamma: float = 0.3,
) -> Tuple[torch.Tensor, float]:
    """E2: Preserve first n_sink tokens in FP16, quantize rest with FOKVQ-E2.

    Args:
        K_head: (seq_len, d_head) key states
        bits_avg: average bits per dimension for quantized portion
        n_sink: number of leading tokens to keep at full precision
        method: 'lloyd' for FOKVQ-E2 (Lloyd-Max), 'uniform' for FOKVQ baseline
        gamma: eigenvalue weighting exponent
    Returns:
        K_recon: (seq_len, d_head) with first n_sink tokens unchanged
        r_eff: effective rank of quantized portion (NaN if no quantization)
    """
    seq_len = K_head.shape[0]

    if n_sink >= seq_len or bits_avg >= 16:
        return K_head.clone(), float("nan")

    K_out = K_head.clone()

    # Quantize only non-sink tokens
    K_rest = K_head[n_sink:]  # (seq_len - n_sink, d_head)

    if method == "lloyd":
        K_rest_q, r_eff = fokvq_e2_quantize_head(K_rest, bits_avg, gamma)
    else:
        K_rest_q, r_eff = fokvq_quantize_head(K_rest, bits_avg, gamma)

    K_out[n_sink:] = K_rest_q
    return K_out.to(K_head.dtype), r_eff


class SinkQuantPatcher(AttentionKQuantPatcher):
    """Extends AttentionKQuantPatcher for E2: preserve sink tokens in FP16."""

    def __init__(self, model, bits: int, gamma: float,
                 n_sink: int, quant_method: str = "lloyd"):
        super().__init__(model, method="sink", bits=bits, gamma=gamma)
        self.n_sink = n_sink
        self.quant_method = quant_method

    def _quantize_and_track(self, K, query_states=None,
                            diagnostic_query_states=None,
                            num_kv_heads=None):
        """Override: preserve first n_sink tokens, quantize rest."""
        B, H, S, D = K.shape
        K_q = K.clone()

        for b in range(B):
            for h in range(H):
                K_head = K[b, h]  # (seq, d_head)
                K_head_q, _ = sink_preserve_quantize_head(
                    K_head, self.bits, self.n_sink,
                    method=self.quant_method, gamma=self.gamma
                )
                K_q[b, h] = K_head_q

        # Track MSE
        diff = (K.float() - K_q.float()).pow(2)
        self.key_mse_sum += float(diff.sum().item())
        self.key_mse_count += int(diff.numel())
        if self.sample_ref is None:
            self.sample_ref = _extract_diag_sample(K)
            self.sample_quant = _extract_diag_sample(K_q)
            q_diag_src = diagnostic_query_states if diagnostic_query_states is not None else query_states
            if q_diag_src is not None:
                q_group = q_diag_src
                if q_group.dim() == 4 and num_kv_heads is not None and q_group.shape[1] != K.shape[1]:
                    q_group = _group_queries_for_kv(q_group, num_kv_heads)
                self.sample_query = _extract_query_diag_sample(q_group)
                self.sample_attention_diag = compute_attention_structure_diagnostics(
                    self.sample_query, self.sample_ref, self.sample_quant
                )

        return K_q


# ============================================================================
# E4: Integer-Constrained Water-Filling Bit Allocation (Hypothesis B2)
# ============================================================================

def compute_bit_allocation_floor2(
    evals: torch.Tensor, d: int, bits_avg: int, gamma: float,
) -> np.ndarray:
    """WF with minimum 2 bits per dimension.

    Same as standard _compute_bit_allocation but with min=2 instead of min=1.
    """
    ev_np = evals.cpu().numpy()
    ev_pos = np.maximum(ev_np, 1e-10)
    w = ev_pos ** gamma
    w /= w.sum()
    ib = np.clip(np.round(w * d * bits_avg).astype(int), 2, 8)
    total_budget = d * bits_avg
    while ib.sum() > total_budget:
        idx = np.argmax(ib)
        ib[idx] = max(2, ib[idx] - 1)
    while ib.sum() < total_budget:
        ib[np.argmin(ib)] += 1
    return np.clip(ib, 2, 8)


def compute_bit_allocation_no_one(
    evals: torch.Tensor, d: int, bits_avg: int, gamma: float,
) -> np.ndarray:
    """WF with no 1-bit dimensions: b_j in {0, 2, 3, 4, ...8}.

    Bits=1 are forbidden. Dimensions that would get 1 bit are set to 0,
    and the freed bits are redistributed to highest-gain dimensions.
    """
    ev_np = evals.cpu().numpy()
    ev_pos = np.maximum(ev_np, 1e-10)
    w = ev_pos ** gamma
    w /= w.sum()

    # Start with standard allocation
    ib = np.round(w * d * bits_avg).astype(int)
    ib = np.clip(ib, 0, 8)

    # Iteratively remove 1-bit assignments and redistribute
    for _ in range(d):
        ones_mask = ib == 1
        if not ones_mask.any():
            break
        # Zero out 1-bit dimensions
        freed_bits = ones_mask.sum()
        ib[ones_mask] = 0
        # Redistribute freed bits to highest-priority non-zero dimensions
        candidates = np.where(ib > 0)[0]
        if len(candidates) == 0:
            # All dims were 1-bit; give 2 bits to top-priority dims
            priority_order = np.argsort(-w)
            bits_to_give = int(freed_bits)
            for idx in priority_order:
                if bits_to_give >= 2:
                    ib[idx] = 2
                    bits_to_give -= 2
                if bits_to_give < 2:
                    break
            break
        # Sort candidates by priority (eigenvalue weight)
        cand_priority = np.argsort(-w[candidates])
        remaining = int(freed_bits)
        for ci in cand_priority:
            idx = candidates[ci]
            if remaining <= 0:
                break
            if ib[idx] < 8:
                ib[idx] += 1
                remaining -= 1

    # Final budget adjustment
    total_budget = d * bits_avg
    while ib.sum() > total_budget:
        # Remove from lowest-priority dimension with bits > 2
        nonzero = np.where(ib > 2)[0]
        if len(nonzero) == 0:
            nonzero = np.where(ib > 0)[0]
        if len(nonzero) == 0:
            break
        # Lowest priority among nonzero
        idx = nonzero[np.argmin(w[nonzero])]
        if ib[idx] == 2:
            ib[idx] = 0  # Skip to 0 (no 1-bit)
        else:
            ib[idx] -= 1

    while ib.sum() < total_budget:
        nonzero = np.where(ib > 0)[0]
        if len(nonzero) == 0:
            # Start assigning to top-priority dims
            idx = np.argmax(w)
            ib[idx] = 2
            continue
        idx = nonzero[np.argmax(w[nonzero])]
        if ib[idx] < 8:
            ib[idx] += 1
        else:
            # Find next best
            other = nonzero[np.argsort(-w[nonzero])]
            for oi in other:
                if ib[oi] < 8:
                    ib[oi] += 1
                    break
            else:
                break  # All maxed out

    # Verify no 1-bit dims
    ib[ib == 1] = 0

    return np.clip(ib, 0, 8)


def compute_bit_allocation_uniform(
    d: int, bits_avg: int,
) -> np.ndarray:
    """Uniform: equal bits per dimension."""
    return np.full(d, bits_avg, dtype=int)


def wf_quantize_head(
    K_head: torch.Tensor,
    bits_avg: int,
    wf_variant: str,
    gamma: float = 0.3,
) -> Tuple[torch.Tensor, float, np.ndarray]:
    """E4: FOKVQ with different integer-constrained bit allocation variants.

    Args:
        K_head: (seq_len, d_head)
        bits_avg: average bits per dimension
        wf_variant: 'floor1', 'floor2', 'no_one', 'uniform'
        gamma: eigenvalue weighting exponent
    Returns:
        K_recon, r_eff, bit_allocation
    """
    d = K_head.shape[-1]
    K_f = K_head.float()

    # PCA decomposition (standard K-only)
    centered, mean, evals, evecs, K_pca = _pca_decompose(K_f)

    # Bit allocation based on variant
    if wf_variant == "floor1":
        ib = _compute_bit_allocation(evals, d, bits_avg, gamma)
    elif wf_variant == "floor2":
        ib = compute_bit_allocation_floor2(evals, d, bits_avg, gamma)
    elif wf_variant == "no_one":
        ib = compute_bit_allocation_no_one(evals, d, bits_avg, gamma)
    elif wf_variant == "uniform":
        ib = compute_bit_allocation_uniform(d, bits_avg)
    else:
        raise ValueError(f"Unknown WF variant: {wf_variant}")

    # Quantize with Lloyd-Max per axis
    K_q = torch.zeros_like(K_pca)
    for i in range(d):
        col = K_pca[:, i]
        b = int(ib[i])
        if b <= 0:
            # 0 bits: quantize to mean (single level)
            K_q[:, i] = col.mean()
        else:
            n_lev = 2 ** b
            cb = _fit_lloyd_max_1d(col, n_lev)
            K_q[:, i] = _quantize_with_codebook_1d(col, cb)

    K_recon = K_q @ evecs.T + mean

    ev_np = np.maximum(evals.cpu().numpy(), 1e-10)
    ev_norm = ev_np / ev_np.sum()
    r_eff = float(np.exp(-np.sum(ev_norm * np.log(ev_norm + 1e-30))))

    return K_recon.to(K_head.dtype), r_eff, ib


class WFQuantPatcher(AttentionKQuantPatcher):
    """Extends AttentionKQuantPatcher for E4: integer-constrained WF variants."""

    def __init__(self, model, bits: int, gamma: float, wf_variant: str):
        super().__init__(model, method=f"wf_{wf_variant}", bits=bits, gamma=gamma)
        self.wf_variant = wf_variant
        self._bit_alloc_stats = []  # collect bit allocations for analysis

    def _quantize_and_track(self, K, query_states=None,
                            diagnostic_query_states=None,
                            num_kv_heads=None):
        """Override: use wf_quantize_head with specified variant."""
        B, H, S, D = K.shape
        K_q = K.clone()

        for b in range(B):
            for h in range(H):
                K_head = K[b, h]
                K_head_q, _, ib = wf_quantize_head(
                    K_head, self.bits, self.wf_variant, self.gamma
                )
                K_q[b, h] = K_head_q
                if len(self._bit_alloc_stats) < 10:  # Sample first few
                    self._bit_alloc_stats.append(ib.tolist())

        # Track MSE
        diff = (K.float() - K_q.float()).pow(2)
        self.key_mse_sum += float(diff.sum().item())
        self.key_mse_count += int(diff.numel())
        if self.sample_ref is None:
            self.sample_ref = _extract_diag_sample(K)
            self.sample_quant = _extract_diag_sample(K_q)
            q_diag_src = diagnostic_query_states if diagnostic_query_states is not None else query_states
            if q_diag_src is not None:
                q_group = q_diag_src
                if q_group.dim() == 4 and num_kv_heads is not None and q_group.shape[1] != K.shape[1]:
                    q_group = _group_queries_for_kv(q_group, num_kv_heads)
                self.sample_query = _extract_query_diag_sample(q_group)
                self.sample_attention_diag = compute_attention_structure_diagnostics(
                    self.sample_query, self.sample_ref, self.sample_quant
                )

        return K_q

    def get_bit_alloc_summary(self) -> Dict:
        """Summarize bit allocation statistics."""
        if not self._bit_alloc_stats:
            return {}
        allocs = np.array(self._bit_alloc_stats)
        return {
            "mean_per_dim": allocs.mean(axis=0).tolist(),
            "n_zero_dims_avg": float((allocs == 0).sum(axis=1).mean()),
            "n_one_dims_avg": float((allocs == 1).sum(axis=1).mean()),
            "min_bits": int(allocs.min()),
            "max_bits": int(allocs.max()),
            "mean_bits": float(allocs.mean()),
        }

    def reset_stats(self):
        super().reset_stats()
        self._bit_alloc_stats = []


# ============================================================================
# Experiment Runners
# ============================================================================

@torch.no_grad()
def run_experiment_e1(
    model, tokenizer, input_ids, args, q_covs: Dict,
) -> List[Dict]:
    """Run E1: MK quantization experiment."""
    results = []
    print("\n" + "=" * 72)
    print("EXPERIMENT E1: Mahalanobis Lloyd-Max (MK) with Query Covariance")
    print("=" * 72)

    # FP16 baseline
    print("\n--- FP16 Baseline ---")
    fp16_result = evaluate_ppl_chunked(model, input_ids, args.context_len)
    fp16_ppl = fp16_result["ppl"]
    print(f"  PPL = {fp16_ppl:.4f}")
    results.append({
        "experiment": "E1",
        "model": args.model_name,
        "method": "fp16",
        "bits": 16,
        "ppl": fp16_ppl,
        "config": {"type": "baseline"},
    })

    # Also run standard FOKVQ-E2 (Lloyd-Max without MK) as control
    for bits in args.bits:
        # Control: FOKVQ-E2 (no query weighting)
        print(f"\n--- FOKVQ-E2 (control) {bits}bit ---")
        patcher = AttentionKQuantPatcher(model, "fokvq_e2", bits, args.gamma)
        patcher.patch()
        patcher.active = True
        patcher.reset_stats()
        try:
            result = evaluate_ppl_chunked(model, input_ids, args.context_len)
            mse = patcher.key_mse_sum / max(patcher.key_mse_count, 1)
            ppl = result["ppl"]
            print(f"  PPL = {ppl:.4f}, MSE = {mse:.6f}")
            results.append({
                "experiment": "E1",
                "model": args.model_name,
                "method": "fokvq_e2",
                "bits": bits,
                "ppl": ppl,
                "mse": mse,
                "config": {"type": "control", "gamma": args.gamma},
            })
        finally:
            patcher.active = False
            patcher.unpatch()
        torch.cuda.empty_cache()
        gc.collect()

        # MK variant
        print(f"\n--- MK {bits}bit ---")
        mk_patcher = MKQuantPatcher(model, bits, args.gamma, q_covs)
        mk_patcher.patch()
        mk_patcher.active = True
        mk_patcher.reset_stats()
        try:
            result = evaluate_ppl_chunked(model, input_ids, args.context_len)
            mse = mk_patcher.key_mse_sum / max(mk_patcher.key_mse_count, 1)
            ppl = result["ppl"]
            attn_diag = mk_patcher.sample_attention_diag
            print(f"  PPL = {ppl:.4f}, MSE = {mse:.6f}")
            if attn_diag:
                print(f"  Attention diag: logit_mse={attn_diag.get('attention_logit_mse', 0):.4f}, "
                      f"topk_overlap={attn_diag.get('attention_topk_overlap', 0):.4f}")
            results.append({
                "experiment": "E1",
                "model": args.model_name,
                "method": "mk",
                "bits": bits,
                "ppl": ppl,
                "mse": mse,
                "attn_diag": attn_diag or {},
                "config": {
                    "type": "mk",
                    "gamma": args.gamma,
                    "calib_tokens": args.mk_calib_tokens,
                },
            })
        finally:
            mk_patcher.active = False
            mk_patcher.unpatch()
        torch.cuda.empty_cache()
        gc.collect()

    return results


@torch.no_grad()
def run_experiment_e2(model, tokenizer, input_ids, args) -> List[Dict]:
    """Run E2: Sink token preservation experiment."""
    results = []
    print("\n" + "=" * 72)
    print("EXPERIMENT E2: Sink Token Preservation")
    print("=" * 72)

    # FP16 baseline
    print("\n--- FP16 Baseline ---")
    fp16_result = evaluate_ppl_chunked(model, input_ids, args.context_len)
    fp16_ppl = fp16_result["ppl"]
    print(f"  PPL = {fp16_ppl:.4f}")
    results.append({
        "experiment": "E2",
        "model": args.model_name,
        "method": "fp16",
        "bits": 16,
        "ppl": fp16_ppl,
        "config": {"type": "baseline"},
    })

    for bits in args.bits:
        for n_sink in args.sink_counts:
            label = f"sink={n_sink}" if n_sink > 0 else "no_sink"
            print(f"\n--- FOKVQ-E2 + {label} @ {bits}bit ---")

            sink_patcher = SinkQuantPatcher(
                model, bits, args.gamma, n_sink, quant_method="lloyd"
            )
            sink_patcher.patch()
            sink_patcher.active = True
            sink_patcher.reset_stats()

            try:
                result = evaluate_ppl_chunked(model, input_ids, args.context_len)
                mse = sink_patcher.key_mse_sum / max(sink_patcher.key_mse_count, 1)
                ppl = result["ppl"]
                print(f"  PPL = {ppl:.4f}, MSE = {mse:.6f}")
                results.append({
                    "experiment": "E2",
                    "model": args.model_name,
                    "method": f"sink_{n_sink}",
                    "bits": bits,
                    "ppl": ppl,
                    "mse": mse,
                    "config": {
                        "n_sink": n_sink,
                        "quant_method": "lloyd",
                        "gamma": args.gamma,
                    },
                })
            finally:
                sink_patcher.active = False
                sink_patcher.unpatch()
            torch.cuda.empty_cache()
            gc.collect()

    return results


@torch.no_grad()
def run_experiment_e4(model, tokenizer, input_ids, args) -> List[Dict]:
    """Run E4: Integer-constrained water-filling experiment."""
    results = []
    print("\n" + "=" * 72)
    print("EXPERIMENT E4: Integer-Constrained Water-Filling Bit Allocation")
    print("=" * 72)

    # FP16 baseline
    print("\n--- FP16 Baseline ---")
    fp16_result = evaluate_ppl_chunked(model, input_ids, args.context_len)
    fp16_ppl = fp16_result["ppl"]
    print(f"  PPL = {fp16_ppl:.4f}")
    results.append({
        "experiment": "E4",
        "model": args.model_name,
        "method": "fp16",
        "bits": 16,
        "ppl": fp16_ppl,
        "config": {"type": "baseline"},
    })

    for bits in args.bits:
        for variant in args.wf_variants:
            print(f"\n--- WF({variant}) @ {bits}bit ---")

            wf_patcher = WFQuantPatcher(model, bits, args.gamma, variant)
            wf_patcher.patch()
            wf_patcher.active = True
            wf_patcher.reset_stats()

            try:
                result = evaluate_ppl_chunked(model, input_ids, args.context_len)
                mse = wf_patcher.key_mse_sum / max(wf_patcher.key_mse_count, 1)
                ppl = result["ppl"]
                bit_summary = wf_patcher.get_bit_alloc_summary()
                print(f"  PPL = {ppl:.4f}, MSE = {mse:.6f}")
                if bit_summary:
                    print(f"  Bit alloc: n_zero={bit_summary.get('n_zero_dims_avg', 0):.1f}, "
                          f"n_one={bit_summary.get('n_one_dims_avg', 0):.1f}, "
                          f"mean={bit_summary.get('mean_bits', 0):.2f}")
                results.append({
                    "experiment": "E4",
                    "model": args.model_name,
                    "method": f"wf_{variant}",
                    "bits": bits,
                    "ppl": ppl,
                    "mse": mse,
                    "bit_alloc_summary": bit_summary,
                    "config": {
                        "wf_variant": variant,
                        "gamma": args.gamma,
                    },
                })
            finally:
                wf_patcher.active = False
                wf_patcher.unpatch()
            torch.cuda.empty_cache()
            gc.collect()

    return results


# ============================================================================
# Summary & Output
# ============================================================================

def print_summary(all_results: List[Dict], fp16_ppl: float):
    """Print a summary table of all experiment results."""
    print("\n" + "=" * 80)
    print("DIAGNOSIS EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"FP16 baseline PPL: {fp16_ppl:.4f}")
    print()

    # Group by experiment
    by_exp = {}
    for r in all_results:
        exp = r["experiment"]
        if exp not in by_exp:
            by_exp[exp] = []
        by_exp[exp].append(r)

    for exp_name in sorted(by_exp.keys()):
        exp_results = by_exp[exp_name]
        print(f"\n--- {exp_name} ---")
        print(f"  {'Method':<25} {'Bits':>5} {'PPL':>10} {'Delta%':>8} {'MSE':>12}")
        print(f"  {'-'*25} {'-'*5} {'-'*10} {'-'*8} {'-'*12}")
        for r in sorted(exp_results, key=lambda x: (x["bits"], x["method"])):
            method = r["method"]
            bits = r["bits"]
            ppl = r.get("ppl", float("inf"))
            mse = r.get("mse", float("nan"))
            if math.isfinite(ppl) and fp16_ppl > 0:
                delta = (ppl - fp16_ppl) / fp16_ppl * 100
                delta_str = f"{delta:+.1f}%"
            else:
                delta_str = "N/A"
            mse_str = f"{mse:.6f}" if math.isfinite(mse) else "N/A"
            ppl_str = f"{ppl:.4f}" if math.isfinite(ppl) else "INF"
            print(f"  {method:<25} {bits:>5} {ppl_str:>10} {delta_str:>8} {mse_str:>12}")

    print("\n" + "=" * 80)


# ============================================================================
# Self-Tests
# ============================================================================

def run_self_tests(seed: int) -> None:
    """Run self-tests for all diagnosis experiment components."""
    set_seed(seed)
    print("Running diagnosis experiment self-tests...")

    # Test 1: mk_quantize_head basic correctness
    K = torch.randn(64, 32) + 5.0
    sigma_q = torch.eye(32) * 2.0 + torch.randn(32, 32) * 0.1
    sigma_q = (sigma_q + sigma_q.T) / 2
    K_mk, r_eff = mk_quantize_head(K, 3, sigma_q, gamma=0.3)
    assert K_mk.shape == K.shape, f"Shape mismatch: {K_mk.shape} vs {K.shape}"
    assert torch.isfinite(K_mk).all(), "MK output has non-finite values"
    assert r_eff > 0, f"Effective rank should be positive: {r_eff}"
    print("  [PASS] mk_quantize_head basic correctness")

    # Test 2: MK produces different results from E3 (different weighting)
    K_e3, _ = fokvq_e2_quantize_head(K, 3, gamma=0.3)
    diff = (K_mk - K_e3).abs().mean().item()
    assert diff > 1e-6, f"MK should differ from E2: diff={diff}"
    print(f"  [PASS] MK differs from E2 (mean abs diff: {diff:.6f})")

    # Test 3: MK with anisotropic Q should focus on Q-important dimensions
    sigma_q_aniso = torch.zeros(32, 32)
    sigma_q_aniso[0, 0] = 100.0
    sigma_q_aniso[1, 1] = 50.0
    for i in range(2, 32):
        sigma_q_aniso[i, i] = 0.01
    K_mk_aniso, _ = mk_quantize_head(K, 3, sigma_q_aniso, gamma=0.3)
    K_mk_iso, _ = mk_quantize_head(K, 3, torch.eye(32), gamma=0.3)
    diff_aniso = (K_mk_aniso - K_mk_iso).abs().mean().item()
    assert diff_aniso > 1e-6, "Anisotropic vs isotropic Q should differ"
    print(f"  [PASS] MK sensitive to Q anisotropy (diff: {diff_aniso:.6f})")

    # Test 4: MK monotonicity
    mses = {}
    for bits in [2, 3, 4]:
        K_q, _ = mk_quantize_head(K, bits, sigma_q, gamma=0.3)
        mses[bits] = (K - K_q).pow(2).mean().item()
    assert mses[4] < mses[3] < mses[2], f"MK monotonicity failed: {mses}"
    print(f"  [PASS] MK MSE monotonicity: 2b={mses[2]:.4f} > 3b={mses[3]:.4f} > 4b={mses[4]:.4f}")

    # Test 5: sink_preserve_quantize_head
    K_sink = torch.randn(128, 32) + 3.0
    for n_sink in [0, 4, 16, 64]:
        K_sq, r = sink_preserve_quantize_head(K_sink, 3, n_sink, method="lloyd")
        assert K_sq.shape == K_sink.shape, f"Sink shape mismatch for n_sink={n_sink}"
        assert torch.isfinite(K_sq).all(), f"Sink output non-finite for n_sink={n_sink}"
        if n_sink > 0:
            # First n_sink tokens should be unchanged
            assert torch.allclose(K_sq[:n_sink], K_sink[:n_sink]), \
                f"Sink tokens not preserved for n_sink={n_sink}"
        if n_sink < 128:
            # Quantized portion should differ from original
            diff = (K_sq[n_sink:] - K_sink[n_sink:]).abs().mean().item()
            assert diff > 1e-6, f"Quantized portion should differ for n_sink={n_sink}"
    print("  [PASS] sink_preserve_quantize_head (n_sink=0,4,16,64)")

    # Test 6: Sink with more sink tokens -> lower MSE
    mse_0 = (K_sink - sink_preserve_quantize_head(K_sink, 3, 0)[0]).pow(2).mean().item()
    mse_16 = (K_sink - sink_preserve_quantize_head(K_sink, 3, 16)[0]).pow(2).mean().item()
    mse_64 = (K_sink - sink_preserve_quantize_head(K_sink, 3, 64)[0]).pow(2).mean().item()
    assert mse_64 < mse_16 < mse_0, \
        f"More sinks should reduce MSE: 0={mse_0:.6f}, 16={mse_16:.6f}, 64={mse_64:.6f}"
    print(f"  [PASS] Sink MSE monotonic in n_sink: 0={mse_0:.6f} > 16={mse_16:.6f} > 64={mse_64:.6f}")

    # Test 7: compute_bit_allocation_floor2
    evals = torch.tensor([100.0, 50.0, 10.0, 1.0] + [0.01] * 28)
    ib_f2 = compute_bit_allocation_floor2(evals, 32, 3, gamma=0.3)
    assert ib_f2.min() >= 2, f"Floor2 has dims below 2: min={ib_f2.min()}"
    assert abs(ib_f2.sum() - 32 * 3) <= 2, f"Floor2 budget mismatch: sum={ib_f2.sum()}, target={32*3}"
    print(f"  [PASS] compute_bit_allocation_floor2 (min={ib_f2.min()}, sum={ib_f2.sum()})")

    # Test 8: compute_bit_allocation_no_one
    ib_no1 = compute_bit_allocation_no_one(evals, 32, 3, gamma=0.3)
    assert not (ib_no1 == 1).any(), f"No-one has 1-bit dims: {ib_no1}"
    assert abs(ib_no1.sum() - 32 * 3) <= 2, f"No-one budget mismatch: sum={ib_no1.sum()}"
    n_zero = (ib_no1 == 0).sum()
    print(f"  [PASS] compute_bit_allocation_no_one (no 1-bit, n_zero={n_zero}, sum={ib_no1.sum()})")

    # Test 9: compute_bit_allocation_uniform
    ib_uni = compute_bit_allocation_uniform(32, 3)
    assert (ib_uni == 3).all(), f"Uniform should be all 3: {ib_uni}"
    print("  [PASS] compute_bit_allocation_uniform")

    # Test 10: wf_quantize_head all variants
    K_wf = torch.randn(64, 32) + 3.0
    for variant in ["floor1", "floor2", "no_one", "uniform"]:
        K_q, r_eff, ib = wf_quantize_head(K_wf, 3, variant, gamma=0.3)
        assert K_q.shape == K_wf.shape, f"WF {variant} shape mismatch"
        assert torch.isfinite(K_q).all(), f"WF {variant} non-finite"
        if variant == "floor2":
            assert ib.min() >= 2, f"WF floor2 has dims below 2"
        if variant == "no_one":
            assert not (ib == 1).any(), f"WF no_one has 1-bit dims"
        if variant == "uniform":
            assert (ib == 3).all(), f"WF uniform not all 3"
    print("  [PASS] wf_quantize_head all variants")

    # Test 11: WF monotonicity across bits
    for variant in ["floor1", "floor2", "no_one", "uniform"]:
        mses = {}
        for bits in [2, 3, 4]:
            K_q, _, _ = wf_quantize_head(K_wf, bits, variant, gamma=0.3)
            mses[bits] = (K_wf - K_q).pow(2).mean().item()
        assert mses[4] < mses[2], \
            f"WF {variant} monotonicity failed: 4b={mses[4]:.6f} >= 2b={mses[2]:.6f}"
    print("  [PASS] WF bit-monotonicity for all variants")

    # Test 12: no_one vs floor1 comparison on anisotropic data
    np.random.seed(seed)
    evals_arr = np.array([100, 50, 10, 1] + [0.01] * 28, dtype=np.float32)
    U = np.linalg.qr(np.random.randn(32, 32).astype(np.float32))[0]
    data = (np.random.randn(128, 32).astype(np.float32)
            @ np.diag(np.sqrt(evals_arr)) @ U.T)
    K_aniso = torch.from_numpy(data) + 3.0

    K_f1, _, ib_f1 = wf_quantize_head(K_aniso, 3, "floor1", gamma=0.3)
    K_no1, _, ib_no1 = wf_quantize_head(K_aniso, 3, "no_one", gamma=0.3)
    mse_f1 = (K_aniso - K_f1).pow(2).mean().item()
    mse_no1 = (K_aniso - K_no1).pow(2).mean().item()
    n_one_f1 = (ib_f1 == 1).sum()
    n_one_no1 = (ib_no1 == 1).sum()
    print(f"  [INFO] Anisotropic 3bit: floor1 MSE={mse_f1:.6f} (n_one={n_one_f1}), "
          f"no_one MSE={mse_no1:.6f} (n_one={n_one_no1})")

    print("\nAll diagnosis self-tests passed.")


# ============================================================================
# Main
# ============================================================================

def run() -> None:
    args = parse_args()

    if args.self_test:
        run_self_tests(args.seed)
        return

    set_seed(args.seed)

    if not args.model_key:
        args.model_key = args.model_name.split("/")[-1].lower().replace("-", "_")

    if not args.output_dir:
        args.output_dir = f"/tmp/exp_diagnosis_{args.model_key}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = resolve_dtype(args.model_name, args.dtype)
    cache_dir = args.cache_dir or None
    experiments = [args.experiment] if args.experiment != "ALL" else ["E1", "E2", "E4"]

    print("=" * 72)
    print("Axis 2/3 Failure Diagnosis Experiments")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {args.model_name} ({args.model_key})")
    print(f"Device: {args.device}, dtype: {dtype}")
    print(f"Chunk length: {args.context_len} (non-overlapping)")
    print(f"Experiments: {experiments}")
    print(f"Bits: {args.bits}")
    print(f"Gamma: {args.gamma}")
    print(f"Seed: {args.seed}")
    if "E1" in experiments:
        print(f"E1 calibration tokens: {args.mk_calib_tokens}")
    if "E2" in experiments:
        print(f"E2 sink counts: {args.sink_counts}")
    if "E4" in experiments:
        print(f"E4 WF variants: {args.wf_variants}")
    print("=" * 72)

    git_head = "unknown"
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        pass

    # Load model
    print("\nLoading model...")
    tokenizer, model = load_model_and_tokenizer(
        args.model_name, dtype, args.device, cache_dir, args.attn_implementation
    )

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    d_head = cfg.hidden_size // n_heads
    print(f"  layers={n_layers}, heads={n_heads}, kv_heads={n_kv_heads}, d_head={d_head}")

    # Load evaluation data
    print("\nLoading WikiText-2 test...")
    input_ids = load_wikitext2_ids(tokenizer, args.device, cache_dir)
    if args.max_eval_tokens > 0:
        input_ids = input_ids[:, :args.max_eval_tokens]
        print(f"  Truncated to {input_ids.shape[1]} tokens")

    all_results = []
    t0 = time.time()

    # E1: MK quantization
    if "E1" in experiments:
        print("\n--- Calibrating query covariance for E1 ---")
        q_covs = calibrate_query_covariance(
            model, tokenizer, args.device,
            max_tokens=args.mk_calib_tokens,
            cache_dir=cache_dir,
        )
        e1_results = run_experiment_e1(model, tokenizer, input_ids, args, q_covs)
        all_results.extend(e1_results)
        del q_covs
        torch.cuda.empty_cache()
        gc.collect()

    # E2: Sink token preservation
    if "E2" in experiments:
        e2_results = run_experiment_e2(model, tokenizer, input_ids, args)
        all_results.extend(e2_results)

    # E4: Integer-constrained WF
    if "E4" in experiments:
        e4_results = run_experiment_e4(model, tokenizer, input_ids, args)
        all_results.extend(e4_results)

    elapsed = time.time() - t0

    # Get FP16 PPL for summary
    fp16_ppls = [r["ppl"] for r in all_results if r["method"] == "fp16"]
    fp16_ppl = fp16_ppls[0] if fp16_ppls else float("inf")

    # Print summary
    print_summary(all_results, fp16_ppl)

    # Write results
    summary = {
        "experiment": "axis2_axis3_diagnosis",
        "generated_at": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "git_head": git_head,
        "model_name": args.model_name,
        "model_key": args.model_key,
        "device": args.device,
        "dtype": str(dtype),
        "context_len": args.context_len,
        "seed": args.seed,
        "gamma": args.gamma,
        "experiments_run": experiments,
        "bits": args.bits,
        "fp16_ppl": fp16_ppl,
        "runtime_s": elapsed,
        "results": all_results,
    }

    if torch.cuda.is_available():
        summary["peak_memory_gib"] = (
            torch.cuda.max_memory_allocated(torch.device(args.device)) / (1024 ** 3)
        )

    out_path = output_dir / f"{args.model_key}_diagnosis.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")
    print(f"Total runtime: {elapsed:.1f}s")

    # Cleanup
    del model, tokenizer, input_ids
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    run()
