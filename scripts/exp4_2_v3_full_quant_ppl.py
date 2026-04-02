"""
FOKVQ Experiment 4-2 v3: Full KV Cache Quantization PPL Benchmark
=================================================================

PROBLEM with v2:
  Sliding-window protocol quantizes only prefix K (50% of window).
  Eval tokens' K remain FP16, diluting the quantization effect.
  Result: GPT-2 3bit shows only +0.4% vs FP16 -- unmeasurably small.

SOLUTION (this version):
  Inject quantization into the model's attention layers so that
  100% of K tensors are quantized during the forward pass.

  Two protocols implemented:
    Protocol A: "k_proj hook" -- quantize k_proj output (pre-RoPE)
      - register_forward_hook on k_proj Linear
      - Output is quantized before RoPE application
      - Matches KVQuant's default protocol (pre-RoPE per-channel)
      - Simple, model-agnostic

    Protocol B: "attention wrapper" -- quantize K after RoPE (post-RoPE)
      - Monkey-patch attention forward to intercept post-RoPE K
      - Matches actual deployment (KV cache stores post-RoPE K)
      - Model-specific but more accurate

  Both use non-overlapping chunk evaluation (no sliding window).

References:
  - KVQuant (Hooper et al., NeurIPS 2024): module replacement, non-overlapping chunks
  - KIVI (Liu et al., ICML 2024): custom attention with inline quantization
  - SKVQ (Duanmu et al., 2024): sliding-window with full quantization

Methods: FP16, Uniform, KIVI, FOKVQ, FOKVQ-QW
Models:  GPT-2 Medium, Qwen2.5-7B, Llama-3-8B (via --model-name)

FOKVQ-QW (E1: Q-Weighted PCA):
  Instead of K's own covariance eigenvectors, uses eigenvectors of
  Σ_Q^{1/2} · Σ_K · Σ_Q^{1/2} — these maximize K variance in the
  directions that matter for Q·K inner product accuracy.
  Q covariance is computed on-the-fly from the current chunk's Q states.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--self-test", action="store_true")
    pre, _ = bootstrap.parse_known_args()

    p = argparse.ArgumentParser(parents=[bootstrap],
        description="FOKVQ v3: Full KV Cache Quantization PPL Benchmark")
    req = not pre.self_test
    p.add_argument("--model-name", type=str, required=req, default="gpt2-medium")
    p.add_argument("--model-key", type=str, required=req, default="gpt2-medium")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--context-len", type=int, required=req, default=1024,
                   help="Chunk length for non-overlapping evaluation")
    p.add_argument("--methods", nargs="+",
                   default=["fp16", "uniform", "kivi", "fokvq"])
    p.add_argument("--bits", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--gamma", type=float, default=0.3,
                   help="FOKVQ eigenvalue weighting exponent")
    p.add_argument("--protocol", type=str, default="post_rope",
                   choices=["pre_rope", "post_rope"],
                   help="pre_rope: hook on k_proj output; post_rope: wrap attention forward")
    p.add_argument("--max-eval-tokens", type=int, default=0,
                   help="Truncate test set (0 = use all)")
    p.add_argument("--output-dir", type=str, required=req,
                   default="/tmp/exp4_2_v3")
    p.add_argument("--cache-dir", type=str, default="")
    p.add_argument("--attn-implementation", type=str, default="eager",
                   help="Must be 'eager' for post_rope protocol (need manual attention)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", type=str, default="ppl",
                   choices=["ppl", "niah", "both"],
                   help="ppl: WikiText-2 PPL, niah: Needle-in-a-Haystack, both: run both")
    p.add_argument("--niah-context-lens", nargs="+", type=int,
                   default=[4096, 8192, 16384],
                   help="Context lengths for NIAH evaluation")
    p.add_argument("--niah-depths", nargs="+", type=float,
                   default=[0.1, 0.3, 0.5, 0.7, 0.9],
                   help="Needle depth positions (fraction of context)")
    p.add_argument("--niah-repeats", type=int, default=3,
                   help="Number of repeats per NIAH condition")
    return p.parse_args()


def resolve_dtype(name: str, dtype_arg: str) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float32":
        return torch.float32
    lowered = name.lower()
    if "qwen" in lowered or "llama" in lowered:
        return torch.bfloat16
    return torch.float16


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Model / Data Loading
# ============================================================================

def load_model_and_tokenizer(model_name: str, dtype: torch.dtype, device: str,
                             cache_dir: Optional[str], attn_impl: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,
                                              **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation=attn_impl,
        trust_remote_code=True,
        **kwargs,
    )
    model.eval()
    return tokenizer, model


def load_wikitext2_ids(tokenizer, device: str, cache_dir: Optional[str]
                       ) -> torch.Tensor:
    from datasets import load_dataset

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", **kwargs)
    all_text = "\n".join([t for t in ds["text"] if t.strip()])
    enc = tokenizer(all_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(device)  # (1, total_tokens)
    print(f"  WikiText-2 test: {ds.num_rows} lines -> {input_ids.shape[1]} tokens")
    return input_ids


# ============================================================================
# Quantization Methods (identical to v2, verified correct)
# ============================================================================

def uniform_quantize_tensor(K: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-token asymmetric uniform quantization."""
    if bits >= 16:
        return K.clone()
    K_f = K.float()
    n = 2 ** bits
    x_min = K_f.min(dim=-1, keepdim=True).values
    x_max = K_f.max(dim=-1, keepdim=True).values
    rng = torch.clamp(x_max - x_min, min=1e-8)
    step = rng / (n - 1)
    K_q = torch.round((K_f - x_min) / step) * step + x_min
    return K_q.to(K.dtype)


def kivi_quantize_tensor(K: torch.Tensor, bits: int) -> torch.Tensor:
    """KIVI-style asymmetric quantization along channel (sequence) dimension."""
    if bits >= 16:
        return K.clone()
    K_f = K.float()
    n = 2 ** bits
    x_min = K_f.min(dim=-2, keepdim=True).values
    x_max = K_f.max(dim=-2, keepdim=True).values
    rng = torch.clamp(x_max - x_min, min=1e-8)
    step = rng / (n - 1)
    K_q = torch.round((K_f - x_min) / step) * step + x_min
    return K_q.to(K.dtype)


# ============================================================================
# SOTA Methods (QuIP#, KVQuant, GEAR, ZipCache, TurboQuant)
# Adapted from Phase 7 numpy implementations to torch for v3 PPL benchmark
# ============================================================================

def _hadamard_matrix(d: int, device: torch.device) -> torch.Tensor:
    """Deterministic Hadamard-like orthogonal matrix."""
    if d & (d - 1) != 0:
        # Not power of 2: use deterministic QR
        rng = torch.Generator(device='cpu').manual_seed(42)
        M = torch.randn(d, d, generator=rng, dtype=torch.float32)
        Q, _ = torch.linalg.qr(M)
        return Q.to(device)
    H = torch.tensor([[1.0]], device=device)
    while H.shape[0] < d:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0) / (2 ** 0.5)
    return H


def _per_dim_uniform(K: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-dimension uniform quantization (for rotated space)."""
    K_f = K.float()
    n = 2 ** bits
    x_min = K_f.min(dim=-2, keepdim=True).values
    x_max = K_f.max(dim=-2, keepdim=True).values
    rng = torch.clamp(x_max - x_min, min=1e-8)
    step = rng / (n - 1)
    K_q = torch.round((K_f - x_min) / step) * step + x_min
    return K_q.to(K.dtype)


def quip_quantize_head(K_head: torch.Tensor, bits: int) -> torch.Tensor:
    """QuIP#: Hadamard incoherence rotation + uniform quantization.
    Chee et al. (2024). Rotates K to spread outlier energy, then uniform quantize."""
    d = K_head.shape[-1]
    H = _hadamard_matrix(d, K_head.device)
    K_rot = K_head.float() @ H
    K_q = _per_dim_uniform(K_rot, bits)
    return (K_q @ H.T).to(K_head.dtype)


def kvquant_quantize_head(K_head: torch.Tensor, bits: int) -> torch.Tensor:
    """KVQuant: per-channel Lloyd-Max + outlier separation at FP16.
    Hooper et al. (NeurIPS 2024)."""
    d = K_head.shape[-1]
    K_f = K_head.float()
    K_q = torch.zeros_like(K_f)
    for i in range(d):
        col = K_f[:, i]
        mu, sigma = col.mean(), col.std()
        outlier_mask = (col - mu).abs() > 2.5 * sigma
        # Lloyd-Max on this dimension
        cb = _fit_lloyd_max_1d(col, 2 ** bits)
        K_q[:, i] = _quantize_with_codebook_1d(col, cb)
        # Restore outliers at full precision
        K_q[outlier_mask, i] = col[outlier_mask]
    return K_q.to(K_head.dtype)


def gear_quantize_head(K_head: torch.Tensor, bits: int,
                       rank: int = 4) -> torch.Tensor:
    """GEAR: uniform quantize + low-rank residual + sparse outlier.
    Kang et al. (NeurIPS 2024)."""
    K_f = K_head.float()
    # Stage 1: Uniform quantize
    K_q1 = _per_dim_uniform(K_f, bits)
    residual = K_f - K_q1

    # Stage 2: Low-rank SVD of residual
    U, S, Vt = torch.linalg.svd(residual, full_matrices=False)
    K_lr = U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]

    # Stage 3: Sparse outlier (top-1% of remaining residual)
    remain = residual - K_lr
    threshold = torch.quantile(remain.abs().reshape(-1).float(), 0.99)
    sparse = torch.where(remain.abs() > threshold, remain, torch.zeros_like(remain))

    K_recon = K_q1 + K_lr + sparse
    return K_recon.to(K_head.dtype)


def zipcache_quantize_head(K_head: torch.Tensor, bits: int) -> torch.Tensor:
    """ZipCache: token importance split (recent=4bit, old=2bit).
    NeurIPS 2024. Approximation: recent 30% tokens get bits+2, rest get bits."""
    n = K_head.shape[0]
    n_recent = max(1, int(n * 0.3))
    K_f = K_head.float()
    K_q = torch.zeros_like(K_f)
    # Old tokens: low bits
    low_bits = max(1, bits)
    high_bits = min(8, bits + 2)
    if n - n_recent > 0:
        K_old = K_f[:n - n_recent]
        K_q[:n - n_recent] = _per_dim_uniform(K_old, low_bits)
    # Recent tokens: high bits
    K_recent = K_f[n - n_recent:]
    K_q[n - n_recent:] = _per_dim_uniform(K_recent, high_bits)
    return K_q.to(K_head.dtype)


def turbo_quantize_head(K_head: torch.Tensor, bits: int) -> torch.Tensor:
    """TurboQuant: Hadamard + Lloyd-Max + QJL 1-bit residual correction.
    Zandieh et al. (ICLR 2026)."""
    d = K_head.shape[-1]
    H = _hadamard_matrix(d, K_head.device)
    K_f = K_head.float()
    K_rot = K_f @ H

    # Lloyd-Max per dimension
    K_q = torch.zeros_like(K_rot)
    for i in range(d):
        cb = _fit_lloyd_max_1d(K_rot[:, i], 2 ** bits)
        K_q[:, i] = _quantize_with_codebook_1d(K_rot[:, i], cb)

    # QJL 1-bit residual correction
    residual = K_rot - K_q
    rng = torch.Generator(device='cpu').manual_seed(42)
    R_jl = torch.randn(d, d, generator=rng, dtype=torch.float32, device=K_head.device)
    R_jl = R_jl / R_jl.norm(dim=1, keepdim=True)
    proj = residual @ R_jl.T
    signs = torch.sign(proj)
    mags = proj.abs().mean(dim=0, keepdim=True)
    correction = (signs * mags) @ R_jl
    scale = residual.norm() / correction.norm().clamp(min=1e-8)
    K_corrected = K_q + correction * scale * 0.5

    K_recon = K_corrected @ H.T
    return K_recon.to(K_head.dtype)


def fokvq_quantize_head(K_head: torch.Tensor, bits_avg: int,
                        gamma: float = 0.3) -> Tuple[torch.Tensor, float]:
    """FOKVQ per-head: PCA rotate -> continuous bit alloc -> asymmetric quant."""
    d = K_head.shape[-1]
    K_f = K_head.float()

    # Centering
    mean = K_f.mean(dim=0)
    centered = K_f - mean

    # PCA
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    cov += torch.eye(d, device=cov.device) * 1e-8
    evals, evecs = torch.linalg.eigh(cov)
    idx = torch.argsort(evals, descending=True)
    evals = evals[idx]
    evecs = evecs[:, idx]

    K_pca = centered @ evecs

    # Continuous bit allocation
    ev_np = evals.cpu().numpy()
    ev_pos = np.maximum(ev_np, 1e-10)
    w = ev_pos ** gamma
    w /= w.sum()
    ib = np.clip(np.round(w * d * bits_avg).astype(int), 1, 8)
    while ib.sum() > d * bits_avg:
        ib[np.argmax(ib)] -= 1
    while ib.sum() < d * bits_avg:
        ib[np.argmin(ib)] += 1
    ib = np.clip(ib, 1, 8)

    # Per-dimension asymmetric quantization
    K_q = torch.zeros_like(K_pca)
    for i in range(d):
        col = K_pca[:, i]
        c_min = col.min()
        c_max = col.max()
        n_lev = 2 ** int(ib[i])
        step = max((c_max - c_min).item(), 1e-8) / (n_lev - 1)
        K_q[:, i] = torch.round((col - c_min) / step) * step + c_min

    K_recon = K_q @ evecs.T + mean

    ev_norm = ev_pos / ev_pos.sum()
    r_eff = float(np.exp(-np.sum(ev_norm * np.log(ev_norm + 1e-30))))

    return K_recon.to(K_head.dtype), r_eff


def fokvq_qw_quantize_head(K_head: torch.Tensor, Q_cov: torch.Tensor,
                            bits_avg: int, gamma: float = 0.3
                            ) -> Tuple[torch.Tensor, float]:
    """FOKVQ-QW: Q-weighted PCA axes for K quantization.

    Instead of K's own covariance eigenvectors, uses eigenvectors of
    Σ_Q^{1/2} · Σ_K · Σ_Q^{1/2} — these maximize K variance in the
    directions that matter for Q·K inner product accuracy.

    The key insight: KIVI's per-channel quantization implicitly respects
    the Q distribution. FOKVQ-QW explicitly rotates K into the axes
    that minimize Q·K inner product error, then applies non-uniform
    bit allocation.

    Args:
        K_head: (seq_len, d_head) — K states for one head
        Q_cov:  (d_head, d_head) — pre-computed Q covariance for this head
        bits_avg: average bits per dimension
        gamma: eigenvalue weighting exponent for bit allocation
    Returns:
        K_recon: (seq_len, d_head) — dequantized K
        r_eff: effective rank of Q-weighted K covariance
    """
    d = K_head.shape[-1]
    K_f = K_head.float()

    # Centering
    mean = K_f.mean(dim=0)
    centered = K_f - mean

    # K covariance
    Sigma_K = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    Sigma_K += torch.eye(d, device=Sigma_K.device) * 1e-8

    # Q covariance: regularize and compute matrix square root
    Q_cov_f = Q_cov.float().to(K_f.device)
    Q_cov_f += torch.eye(d, device=Q_cov_f.device) * 1e-6
    evals_q, evecs_q = torch.linalg.eigh(Q_cov_f)
    evals_q = torch.clamp(evals_q, min=1e-8)

    sqrt_Q = evecs_q @ torch.diag(evals_q.sqrt()) @ evecs_q.T

    # Q-weighted K covariance: Σ_{K|Q} = Σ_Q^{1/2} · Σ_K · Σ_Q^{1/2}
    Sigma_KQ = sqrt_Q @ Sigma_K @ sqrt_Q
    Sigma_KQ = (Sigma_KQ + Sigma_KQ.T) / 2  # enforce symmetry
    Sigma_KQ += torch.eye(d, device=Sigma_KQ.device) * 1e-8

    # Eigenvectors of Q-weighted covariance
    evals, evecs_kq = torch.linalg.eigh(Sigma_KQ)
    idx = torch.argsort(evals, descending=True)
    evals = evals[idx]
    evecs_kq = evecs_kq[:, idx]

    # Transform back to original K space: Σ_Q^{-1/2} · evecs_kq
    inv_sqrt_Q = evecs_q @ torch.diag(evals_q.rsqrt()) @ evecs_q.T
    evecs = inv_sqrt_Q @ evecs_kq
    # Re-orthonormalize for numerical stability
    evecs, _ = torch.linalg.qr(evecs)

    # Project K onto Q-weighted axes
    K_pca = centered @ evecs

    # Continuous bit allocation (same as regular FOKVQ)
    ev_np = evals.cpu().numpy()
    ev_pos = np.maximum(ev_np, 1e-10)
    w = ev_pos ** gamma
    w /= w.sum()
    ib = np.clip(np.round(w * d * bits_avg).astype(int), 1, 8)
    while ib.sum() > d * bits_avg:
        ib[np.argmax(ib)] -= 1
    while ib.sum() < d * bits_avg:
        ib[np.argmin(ib)] += 1
    ib = np.clip(ib, 1, 8)

    # Per-dimension asymmetric quantization
    K_q = torch.zeros_like(K_pca)
    for i in range(d):
        col = K_pca[:, i]
        c_min = col.min()
        c_max = col.max()
        n_lev = 2 ** int(ib[i])
        step = max((c_max - c_min).item(), 1e-8) / (n_lev - 1)
        K_q[:, i] = torch.round((col - c_min) / step) * step + c_min

    K_recon = K_q @ evecs.T + mean

    ev_norm = ev_pos / ev_pos.sum()
    r_eff = float(np.exp(-np.sum(ev_norm * np.log(ev_norm + 1e-30))))

    return K_recon.to(K_head.dtype), r_eff


# ============================================================================
# E2: Per-Axis Adaptive Quantizer (Lloyd-Max codebook per PCA axis)
# ============================================================================

def _fit_lloyd_max_1d(data: torch.Tensor, n_levels: int,
                      n_iters: int = 20) -> torch.Tensor:
    """Fit 1D Lloyd-Max codebook to data via iterative quantile refinement.

    Returns: codebook of shape (n_levels,), sorted ascending.
    """
    vals = data.reshape(-1).float()
    if vals.numel() == 0 or n_levels <= 1:
        return vals.mean().unsqueeze(0)
    # Initialize with quantiles
    quantiles = torch.linspace(0.0, 1.0, n_levels + 2, dtype=torch.float32)[1:-1]
    cb = torch.quantile(vals, quantiles.to(vals.device)).unique(sorted=True)
    if cb.numel() < n_levels:
        cb = torch.linspace(vals.min(), vals.max(), n_levels,
                            device=vals.device, dtype=torch.float32)
    for _ in range(n_iters):
        thresholds = 0.5 * (cb[:-1] + cb[1:])
        bucket = torch.bucketize(vals, thresholds)
        new_cb = cb.clone()
        for idx in range(n_levels):
            mask = bucket == idx
            if mask.any():
                new_cb[idx] = vals[mask].mean()
        if torch.allclose(new_cb, cb, atol=1e-7):
            break
        cb = new_cb
    return cb.sort().values


def _quantize_with_codebook_1d(data: torch.Tensor,
                               codebook: torch.Tensor) -> torch.Tensor:
    """Quantize 1D tensor to nearest codebook entry."""
    flat = data.reshape(-1, 1).float()
    levels = codebook.to(device=data.device, dtype=torch.float32).reshape(1, -1)
    nearest = (flat - levels).abs().argmin(dim=1)
    return levels.squeeze(0)[nearest].reshape_as(data).to(data.dtype)


def _fit_lloyd_max_mahalanobis_1d(data: torch.Tensor, n_levels: int,
                                   weight: float,
                                   n_iters: int = 20) -> torch.Tensor:
    """Fit 1D Lloyd-Max codebook minimizing Mahalanobis-weighted distortion.

    Mahalanobis weight = 1/variance for this axis.
    Higher weight → this axis matters more → finer codebook placement.
    In practice, the optimal codebook for weighted MSE on 1D data is the same
    as standard Lloyd-Max applied to scaled data: data * sqrt(weight).

    We scale data, fit standard Lloyd-Max, then unscale codebook.
    """
    scale = max(weight, 1e-10) ** 0.5
    scaled = data * scale
    cb_scaled = _fit_lloyd_max_1d(scaled, n_levels, n_iters)
    return cb_scaled / scale


def _pca_decompose(K_head: torch.Tensor, Q_cov: Optional[torch.Tensor] = None):
    """Shared PCA decomposition for E1/E2/E3 variants.

    Returns: (centered, mean, evals, evecs, K_pca)
      - If Q_cov is None: standard K-only PCA
      - If Q_cov is provided: Q-weighted PCA (E1)
    """
    d = K_head.shape[-1]
    K_f = K_head.float()
    mean = K_f.mean(dim=0)
    centered = K_f - mean

    Sigma_K = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    Sigma_K += torch.eye(d, device=Sigma_K.device) * 1e-8

    if Q_cov is not None:
        # E1: Q-weighted PCA
        Q_cov_f = Q_cov.float().to(K_f.device)
        Q_cov_f += torch.eye(d, device=Q_cov_f.device) * 1e-6
        ev_q, U_q = torch.linalg.eigh(Q_cov_f)
        ev_q = torch.clamp(ev_q, min=1e-8)
        sqrt_Q = U_q @ torch.diag(ev_q.sqrt()) @ U_q.T
        inv_sqrt_Q = U_q @ torch.diag(ev_q.rsqrt()) @ U_q.T

        Sigma_KQ = sqrt_Q @ Sigma_K @ sqrt_Q
        Sigma_KQ = (Sigma_KQ + Sigma_KQ.T) / 2
        Sigma_KQ += torch.eye(d, device=Sigma_KQ.device) * 1e-8

        evals, evecs_kq = torch.linalg.eigh(Sigma_KQ)
        idx = torch.argsort(evals, descending=True)
        evals = evals[idx]
        evecs_kq = evecs_kq[:, idx]
        evecs = inv_sqrt_Q @ evecs_kq
        evecs, _ = torch.linalg.qr(evecs)
    else:
        # Standard K-only PCA
        evals, evecs = torch.linalg.eigh(Sigma_K)
        idx = torch.argsort(evals, descending=True)
        evals = evals[idx]
        evecs = evecs[:, idx]

    K_pca = centered @ evecs
    return centered, mean, evals, evecs, K_pca


def _compute_bit_allocation(evals: torch.Tensor, d: int, bits_avg: int,
                            gamma: float) -> np.ndarray:
    """Shared eigenvalue-weighted bit allocation."""
    ev_np = evals.cpu().numpy()
    ev_pos = np.maximum(ev_np, 1e-10)
    w = ev_pos ** gamma
    w /= w.sum()
    ib = np.clip(np.round(w * d * bits_avg).astype(int), 1, 8)
    while ib.sum() > d * bits_avg:
        ib[np.argmax(ib)] -= 1
    while ib.sum() < d * bits_avg:
        ib[np.argmin(ib)] += 1
    return np.clip(ib, 1, 8)


def _quantize_pca_uniform(K_pca: torch.Tensor, ib: np.ndarray) -> torch.Tensor:
    """Standard per-axis uniform asymmetric quantization (baseline FOKVQ)."""
    d = K_pca.shape[-1]
    K_q = torch.zeros_like(K_pca)
    for i in range(d):
        col = K_pca[:, i]
        c_min, c_max = col.min(), col.max()
        n_lev = 2 ** int(ib[i])
        step = max((c_max - c_min).item(), 1e-8) / (n_lev - 1)
        K_q[:, i] = torch.round((col - c_min) / step) * step + c_min
    return K_q


def _quantize_pca_lloyd(K_pca: torch.Tensor, ib: np.ndarray) -> torch.Tensor:
    """E2: Per-axis Lloyd-Max adaptive codebook quantization."""
    d = K_pca.shape[-1]
    K_q = torch.zeros_like(K_pca)
    for i in range(d):
        col = K_pca[:, i]
        n_lev = 2 ** int(ib[i])
        cb = _fit_lloyd_max_1d(col, n_lev)
        K_q[:, i] = _quantize_with_codebook_1d(col, cb)
    return K_q


def _quantize_pca_mk(K_pca: torch.Tensor, ib: np.ndarray,
                      evals: torch.Tensor) -> torch.Tensor:
    """E3: Per-axis Mahalanobis-weighted Lloyd-Max codebook.

    Axes with smaller eigenvalue (lower K variance) get higher MK weight
    because errors there are amplified by Σ_K^{-1}.
    MK weight for axis i = 1 / eigenvalue_i.
    """
    d = K_pca.shape[-1]
    ev_np = evals.cpu().numpy()
    ev_pos = np.maximum(ev_np, 1e-10)
    K_q = torch.zeros_like(K_pca)
    for i in range(d):
        col = K_pca[:, i]
        n_lev = 2 ** int(ib[i])
        mk_weight = 1.0 / ev_pos[i]
        cb = _fit_lloyd_max_mahalanobis_1d(col, n_lev, mk_weight)
        K_q[:, i] = _quantize_with_codebook_1d(col, cb)
    return K_q


def fokvq_e2_quantize_head(K_head: torch.Tensor, bits_avg: int,
                           gamma: float = 0.3,
                           Q_cov: Optional[torch.Tensor] = None
                           ) -> Tuple[torch.Tensor, float]:
    """E2: FOKVQ + per-axis Lloyd-Max adaptive codebook.

    Same PCA axes as FOKVQ (or Q-weighted if Q_cov given),
    but replaces uniform scalar quantizer with per-axis Lloyd-Max codebook.
    """
    d = K_head.shape[-1]
    centered, mean, evals, evecs, K_pca = _pca_decompose(K_head, Q_cov)
    ib = _compute_bit_allocation(evals, d, bits_avg, gamma)
    K_q = _quantize_pca_lloyd(K_pca, ib)
    K_recon = K_q @ evecs.T + mean
    ev_np = np.maximum(evals.cpu().numpy(), 1e-10)
    ev_norm = ev_np / ev_np.sum()
    r_eff = float(np.exp(-np.sum(ev_norm * np.log(ev_norm + 1e-30))))
    return K_recon.to(K_head.dtype), r_eff


def fokvq_e3_quantize_head(K_head: torch.Tensor, bits_avg: int,
                           gamma: float = 0.3,
                           Q_cov: Optional[torch.Tensor] = None
                           ) -> Tuple[torch.Tensor, float]:
    """E3: FOKVQ + Mahalanobis-weighted Lloyd-Max codebook.

    Same PCA axes, but codebook optimized for MK distortion (1/eigenvalue weight).
    Low-variance axes get finer codebook placement because Σ_K^{-1} amplifies
    errors there.
    """
    d = K_head.shape[-1]
    centered, mean, evals, evecs, K_pca = _pca_decompose(K_head, Q_cov)
    ib = _compute_bit_allocation(evals, d, bits_avg, gamma)
    K_q = _quantize_pca_mk(K_pca, ib, evals)
    K_recon = K_q @ evecs.T + mean
    ev_np = np.maximum(evals.cpu().numpy(), 1e-10)
    ev_norm = ev_np / ev_np.sum()
    r_eff = float(np.exp(-np.sum(ev_norm * np.log(ev_norm + 1e-30))))
    return K_recon.to(K_head.dtype), r_eff


def fokvq_full_quantize_head(K_head: torch.Tensor, bits_avg: int,
                             gamma: float = 0.3,
                             Q_cov: Optional[torch.Tensor] = None
                             ) -> Tuple[torch.Tensor, float]:
    """E1+E2+E3 combined: Q-weighted PCA + MK-weighted Lloyd-Max codebook.

    The strongest variant:
      - E1: Q-weighted PCA axes (if Q_cov provided)
      - E2: Per-axis Lloyd-Max adaptive codebook
      - E3: Mahalanobis (1/eigenvalue) weighted codebook optimization
    """
    d = K_head.shape[-1]
    centered, mean, evals, evecs, K_pca = _pca_decompose(K_head, Q_cov)
    ib = _compute_bit_allocation(evals, d, bits_avg, gamma)
    K_q = _quantize_pca_mk(K_pca, ib, evals)
    K_recon = K_q @ evecs.T + mean
    ev_np = np.maximum(evals.cpu().numpy(), 1e-10)
    ev_norm = ev_np / ev_np.sum()
    r_eff = float(np.exp(-np.sum(ev_norm * np.log(ev_norm + 1e-30))))
    return K_recon.to(K_head.dtype), r_eff


def quantize_k_tensor(K: torch.Tensor, method: str, bits: int,
                      gamma: float = 0.3,
                      q_covs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Quantize a K tensor of shape (..., seq, d_head).

    Handles both 2D (seq, d_head) and 4D (batch, heads, seq, d_head).

    Args:
        K: key tensor
        method: "fp16", "uniform", "kivi", "fokvq", "fokvq_qw",
                "fokvq_e2", "fokvq_e3", "fokvq_full"
        bits: quantization bits
        gamma: FOKVQ eigenvalue weighting exponent
        q_covs: For fokvq_qw only. Q covariance matrices per head.
                 Shape depends on K dim:
                   2D K: (d_head, d_head) — single head
                   3D K: (heads, d_head, d_head)
                   4D K: (batch, heads, d_head, d_head) or (heads, d_head, d_head)
    """
    if method == "fp16" or bits >= 16:
        return K

    def _quantize_single_head(K_head, method, bits, gamma, q_cov=None):
        """Quantize a single (seq, d_head) head."""
        if method == "uniform":
            return uniform_quantize_tensor(K_head, bits)
        elif method == "kivi":
            return kivi_quantize_tensor(K_head, bits)
        elif method == "fokvq":
            K_q, _ = fokvq_quantize_head(K_head, bits, gamma)
            return K_q
        elif method == "fokvq_qw":
            if q_cov is not None:
                K_q, _ = fokvq_qw_quantize_head(K_head, q_cov, bits, gamma)
            else:
                K_q, _ = fokvq_quantize_head(K_head, bits, gamma)
            return K_q
        elif method == "fokvq_e2":
            # E2: K-only PCA + Lloyd-Max codebook (no Q_cov needed)
            K_q, _ = fokvq_e2_quantize_head(K_head, bits, gamma)
            return K_q
        elif method == "fokvq_e3":
            # E3: K-only PCA + MK-weighted Lloyd-Max (no Q_cov needed)
            K_q, _ = fokvq_e3_quantize_head(K_head, bits, gamma)
            return K_q
        elif method == "fokvq_full":
            # E1+E2+E3: Q-weighted PCA + MK-weighted Lloyd-Max
            K_q, _ = fokvq_full_quantize_head(K_head, bits, gamma, Q_cov=q_cov)
            return K_q
        # --- SOTA methods ---
        elif method == "quip":
            return quip_quantize_head(K_head, bits)
        elif method == "kvquant":
            return kvquant_quantize_head(K_head, bits)
        elif method == "gear":
            return gear_quantize_head(K_head, bits)
        elif method == "zipcache":
            return zipcache_quantize_head(K_head, bits)
        elif method == "turboquant":
            return turbo_quantize_head(K_head, bits)
        else:
            raise ValueError(f"Unknown method: {method}")

    if K.dim() == 2:
        # (seq, d_head) -- single head
        q_cov = q_covs if q_covs is not None else None
        return _quantize_single_head(K, method, bits, gamma, q_cov)
    elif K.dim() == 4:
        # (batch, heads, seq, d_head)
        K_out = K.clone()
        for b in range(K.shape[0]):
            for h in range(K.shape[1]):
                q_cov = None
                if q_covs is not None:
                    if q_covs.dim() == 4:
                        q_cov = q_covs[b, h]
                    elif q_covs.dim() == 3:
                        q_cov = q_covs[h]
                K_out[b, h] = _quantize_single_head(
                    K[b, h], method, bits, gamma, q_cov)
        return K_out
    elif K.dim() == 3:
        # (batch_or_heads, seq, d_head) -- e.g., from k_proj output reshaped
        K_out = K.clone()
        for i in range(K.shape[0]):
            q_cov = q_covs[i] if q_covs is not None else None
            K_out[i] = _quantize_single_head(
                K[i], method, bits, gamma, q_cov)
        return K_out
    else:
        raise ValueError(f"Unexpected K dim: {K.dim()}")


# ============================================================================
# Protocol A: Pre-RoPE K-Proj Hook
# ============================================================================

class KProjQuantHook:
    """Hook to quantize k_proj output (pre-RoPE).

    After k_proj produces K = W_k @ hidden_states, this hook applies
    quantization to the output before RoPE is applied.

    Shape: k_proj output is (batch, seq, num_kv_heads * d_head)
    We reshape to (batch * num_kv_heads, seq, d_head) for per-head quantization.
    """

    def __init__(self, num_kv_heads: int, d_head: int,
                 method: str, bits: int, gamma: float = 0.3):
        self.num_kv_heads = num_kv_heads
        self.d_head = d_head
        self.method = method
        self.bits = bits
        self.gamma = gamma
        self.active = False
        self.handle = None
        self.key_mse_sum = 0.0
        self.key_mse_count = 0

    def __call__(self, module, input, output):
        if not self.active:
            return output

        orig_shape = output.shape  # (batch, seq, num_kv_heads * d_head)
        B, S, D = orig_shape

        # Reshape: (B, S, n_kv_heads * d) -> (B, n_kv_heads, S, d)
        K = output.view(B, S, self.num_kv_heads, self.d_head)
        K = K.transpose(1, 2)  # (B, n_kv_heads, S, d)

        K_q = quantize_k_tensor(K, self.method, self.bits, self.gamma)

        # MSE tracking
        diff = (K.float() - K_q.float()).pow(2)
        self.key_mse_sum += float(diff.sum().item())
        self.key_mse_count += int(diff.numel())

        # Reshape back: (B, n_kv_heads, S, d) -> (B, S, n_kv_heads * d)
        K_q = K_q.transpose(1, 2).contiguous().view(orig_shape)
        return K_q

    def reset_stats(self):
        self.key_mse_sum = 0.0
        self.key_mse_count = 0


# ============================================================================
# Protocol B: Post-RoPE Attention Wrapper
# ============================================================================

class PostRoPEQuantWrapper:
    """Wraps attention module to quantize K after RoPE application.

    Strategy: For each attention layer, we replace the forward method with
    a wrapper that:
      1. Calls the original forward
      2. BUT intercepts the K tensor after RoPE and before attention computation

    Implementation: We use a two-forward approach per chunk:
      Pass 1: Normal forward with use_cache=True -> capture post-RoPE K from cache
      Pass 2: We don't need pass 2! Instead, we quantize the K in the cache
              and compute attention manually.

    Actually, the cleanest approach for post-RoPE:
      - Single forward with use_cache=True
      - Extract K from KV cache (already post-RoPE)
      - Quantize K
      - Recompute attention: attn = softmax(Q @ K_q^T / sqrt(d)) @ V
      - Replace the attention output in the hidden states

    This is complex. Instead, we use a SIMPLER approach:
      - Register hooks on the attention module that capture Q, K, V after RoPE
      - Quantize K
      - Recompute attention output
      - Return the modified output

    For HuggingFace models with eager attention, the attention computation
    happens inside the attention module's forward. We can intercept by
    monkey-patching the _attn method or the scaled_dot_product_attention call.
    """
    pass  # See implementation below


def find_k_proj_modules(model) -> list:
    """Find all k_proj Linear layers in the model."""
    k_proj_modules = []
    for name, module in model.named_modules():
        if name.endswith('.k_proj') and isinstance(module, nn.Linear):
            k_proj_modules.append((name, module))
    return k_proj_modules


def find_attention_modules(model) -> list:
    """Find all attention modules in the model."""
    attn_modules = []
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if 'Attention' in cls_name and (
            (hasattr(module, 'k_proj') and hasattr(module, 'q_proj')) or  # Llama/Qwen2/Mistral
            hasattr(module, 'c_attn')                                     # GPT-2
        ):
            attn_modules.append((name, module))
    return attn_modules


@contextmanager
def k_quantization_active(hooks: list):
    """Context manager to activate/deactivate K quantization hooks."""
    for hook in hooks:
        hook.active = True
        hook.reset_stats()
    try:
        yield hooks
    finally:
        for hook in hooks:
            hook.active = False


def install_pre_rope_hooks(model, method: str, bits: int, gamma: float = 0.3
                           ) -> list:
    """Install pre-RoPE quantization hooks on all k_proj layers.

    Returns list of KProjQuantHook objects (inactive by default).
    """
    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv_heads = getattr(cfg, 'num_key_value_heads', n_heads)
    d_head = cfg.hidden_size // n_heads

    k_proj_modules = find_k_proj_modules(model)
    if not k_proj_modules:
        raise RuntimeError("No k_proj modules found in model")

    hooks = []
    for name, module in k_proj_modules:
        hook = KProjQuantHook(n_kv_heads, d_head, method, bits, gamma)
        hook.handle = module.register_forward_hook(hook)
        hooks.append(hook)

    print(f"  Installed {len(hooks)} pre-RoPE K quantization hooks "
          f"(method={method}, bits={bits})")
    return hooks


def remove_hooks(hooks: list):
    """Remove all installed hooks."""
    for hook in hooks:
        if hook.handle is not None:
            hook.handle.remove()
            hook.handle = None


# ============================================================================
# Protocol B: Post-RoPE via Attention Module Patching
# ============================================================================

class AttentionKQuantPatcher:
    """Patches attention modules for post-RoPE K quantization.

    For each attention module, replaces the forward method to intercept
    the key_states tensor after RoPE application and quantize it.

    Supports:
      - GPT2Attention (GPT-2 family)
      - LlamaAttention (Llama, Qwen2 family)
      - MistralAttention

    The patching approach:
      We store a reference to the original forward and create a new one
      that modifies key_states in-flight.

    For models using eager attention (required!), the attention forward
    computes Q, K, V, applies RoPE to Q and K, then does:
      attn_weights = torch.matmul(Q, K.transpose(-1, -2)) / sqrt(d)
      attn_weights = softmax(attn_weights + causal_mask)
      attn_output = torch.matmul(attn_weights, V)

    We intercept by wrapping the forward and modifying key_states after RoPE.
    """

    def __init__(self, model, method: str, bits: int, gamma: float = 0.3):
        self.model = model
        self.method = method
        self.bits = bits
        self.gamma = gamma
        self.active = False
        self.original_forwards = {}
        self.key_mse_sum = 0.0
        self.key_mse_count = 0
        self._patched = False

    def patch(self):
        """Install patches on all attention modules."""
        if self._patched:
            return

        model_type = self._detect_model_type()
        attn_modules = find_attention_modules(self.model)

        if not attn_modules:
            raise RuntimeError("No attention modules found")

        for name, attn_module in attn_modules:
            orig_forward = attn_module.forward
            self.original_forwards[name] = orig_forward

            if model_type == "gpt2":
                patched = self._make_gpt2_patched_forward(attn_module, orig_forward)
            elif model_type in ("llama", "qwen2", "mistral"):
                patched = self._make_llama_patched_forward(attn_module, orig_forward)
            else:
                print(f"  WARNING: Unknown model type '{model_type}' for {name}, "
                      f"falling back to pre-RoPE hook")
                continue

            attn_module.forward = patched

        self._patched = True
        print(f"  Patched {len(self.original_forwards)} attention modules "
              f"(type={model_type}, method={self.method}, bits={self.bits})")

    def unpatch(self):
        """Restore original forwards."""
        if not self._patched:
            return
        for name, module in find_attention_modules(self.model):
            if name in self.original_forwards:
                module.forward = self.original_forwards[name]
        self.original_forwards.clear()
        self._patched = False

    def reset_stats(self):
        self.key_mse_sum = 0.0
        self.key_mse_count = 0

    def _detect_model_type(self) -> str:
        """Detect model architecture type."""
        model_cls = type(self.model).__name__.lower()
        config_type = getattr(self.model.config, 'model_type', '').lower()

        if 'gpt2' in model_cls or 'gpt2' in config_type:
            return 'gpt2'
        elif 'llama' in model_cls or 'llama' in config_type:
            return 'llama'
        elif 'qwen2' in model_cls or 'qwen2' in config_type:
            return 'qwen2'
        elif 'mistral' in model_cls or 'mistral' in config_type:
            return 'mistral'
        else:
            return config_type or 'unknown'

    def _quantize_and_track(self, K: torch.Tensor,
                            query_states: Optional[torch.Tensor] = None
                            ) -> torch.Tensor:
        """Quantize K tensor and track MSE.

        Args:
            K: key states, shape (batch, heads, seq, d_head)
            query_states: if method is fokvq_qw, Q states for on-the-fly
                         Q covariance computation. Shape (batch, heads, seq, d_head).
                         For GQA models, Q heads are already grouped to match KV heads.
        """
        q_covs = None
        if self.method in ("fokvq_qw", "fokvq_full") and query_states is not None:
            # Compute per-head Q covariance on-the-fly from current chunk
            # query_states: (batch, n_heads_or_kv_heads, seq, d_head)
            # We compute covariance per head across the sequence dimension
            Q_f = query_states.float()
            # Average over batch dimension, compute per-head covariance
            # Result: (n_heads, d_head, d_head)
            n_heads = Q_f.shape[1]
            d_head = Q_f.shape[-1]
            q_covs = torch.zeros(n_heads, d_head, d_head,
                                 device=Q_f.device, dtype=torch.float32)
            for h in range(n_heads):
                # Pool across batch: (batch * seq, d_head)
                Q_h = Q_f[:, h].reshape(-1, d_head)
                q_covs[h] = (Q_h.T @ Q_h) / max(Q_h.shape[0] - 1, 1)

        K_q = quantize_k_tensor(K, self.method, self.bits, self.gamma,
                                q_covs=q_covs)
        diff = (K.float() - K_q.float()).pow(2)
        self.key_mse_sum += float(diff.sum().item())
        self.key_mse_count += int(diff.numel())
        return K_q

    def _make_gpt2_patched_forward(self, attn_module, orig_forward):
        """Patch for GPT-2 style attention.

        GPT-2 uses c_attn (combined QKV projection) and split_heads.
        K is not separately accessible before attention computation.

        Strategy: We hook into the combined projection output, split it,
        quantize K, and recompute attention.
        """
        patcher = self

        def patched_forward(hidden_states, layer_past=None, attention_mask=None,
                          head_mask=None, encoder_hidden_states=None,
                          encoder_attention_mask=None, use_cache=False,
                          output_attentions=False, **kwargs):
            if not patcher.active:
                return orig_forward(
                    hidden_states, layer_past=layer_past,
                    attention_mask=attention_mask, head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache, output_attentions=output_attentions,
                    **kwargs)

            # GPT-2 combined QKV projection
            qkv = attn_module.c_attn(hidden_states)
            query, key, value = qkv.split(attn_module.split_size, dim=2)

            # split_heads: (batch, seq, n_embd) -> (batch, heads, seq, head_dim)
            num_heads = attn_module.num_heads
            head_dim = attn_module.head_dim
            bsz, seq_len = hidden_states.shape[:2]

            query = query.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
            key = key.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
            value = value.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)

            # GPT-2 has no RoPE, so key here is already the final K
            # Handle layer_past (KV cache from previous steps)
            if layer_past is not None:
                past_key, past_value = layer_past
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)

            present = (key, value) if use_cache else None

            # >>> QUANTIZE K HERE (post any concatenation) <<<
            key = patcher._quantize_and_track(key, query_states=query)

            # Manual attention computation
            attn_weights = torch.matmul(query, key.transpose(-1, -2))
            attn_weights = attn_weights / math.sqrt(head_dim)

            # Causal mask
            if not attn_module.is_cross_attention:
                query_length = query.size(-2)
                key_length = key.size(-2)
                # Build causal mask manually (transformers 5.x removed attn_module.bias)
                causal_mask = torch.tril(
                    torch.ones(key_length, key_length,
                               device=attn_weights.device, dtype=torch.bool)
                )
                causal_mask = causal_mask[key_length - query_length : key_length, :key_length]
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, q, k)
                mask_value = torch.finfo(attn_weights.dtype).min
                attn_weights = attn_weights.masked_fill(~causal_mask, mask_value)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = attn_weights.to(value.dtype)

            if head_mask is not None:
                attn_weights = attn_weights * head_mask

            attn_output = torch.matmul(attn_weights, value)
            # merge_heads: (batch, heads, seq, head_dim) -> (batch, seq, n_embd)
            attn_output = attn_output.transpose(1, 2).contiguous().view(
                bsz, seq_len, num_heads * head_dim)
            attn_output = attn_module.c_proj(attn_output)
            attn_output = attn_module.resid_dropout(attn_output)

            outputs = (attn_output, present)
            if output_attentions:
                outputs += (attn_weights,)

            return outputs

        return patched_forward

    def _make_llama_patched_forward(self, attn_module, orig_forward):
        """Patch for Llama/Qwen2/Mistral style attention.

        These models have separate q_proj, k_proj, v_proj and apply RoPE
        to Q and K. We intercept K after RoPE application.
        """
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
                    position_embeddings=position_embeddings,
                    **kwargs)

            bsz, q_len, _ = hidden_states.size()

            query_states = attn_module.q_proj(hidden_states)
            key_states = attn_module.k_proj(hidden_states)
            value_states = attn_module.v_proj(hidden_states)

            # Get head dimensions (transformers 5.x: may be on config, not module)
            num_heads = getattr(attn_module, 'num_heads',
                                attn_module.config.num_attention_heads)
            num_kv_heads = getattr(attn_module, 'num_key_value_heads',
                                   attn_module.config.num_key_value_heads)
            head_dim = attn_module.head_dim

            query_states = query_states.view(
                bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(
                bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

            # Apply RoPE
            if position_embeddings is not None:
                cos, sin = position_embeddings
            elif hasattr(attn_module, 'rotary_emb'):
                if position_ids is not None:
                    cos, sin = attn_module.rotary_emb(
                        value_states, position_ids)
                else:
                    cos, sin = attn_module.rotary_emb(
                        value_states, seq_len=q_len)
            else:
                cos, sin = None, None

            if cos is not None and sin is not None:
                # Apply rotary embeddings
                from transformers.models.llama.modeling_llama import (
                    apply_rotary_pos_emb)
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin)

            # >>> QUANTIZE K HERE (post-RoPE) <<<
            # For fokvq_qw: compute Q covariance per KV head.
            # With GQA, multiple Q heads map to one KV head.
            # We pool Q heads within each group for the covariance.
            q_for_qw = None
            if patcher.method in ("fokvq_qw", "fokvq_full"):
                if num_kv_heads != num_heads:
                    # GQA: pool Q heads per KV group -> (bsz, num_kv_heads, seq, d)
                    n_rep = num_heads // num_kv_heads
                    q_grouped = query_states.view(
                        bsz, num_kv_heads, n_rep, q_len, head_dim)
                    # Average across the group for covariance estimation
                    q_for_qw = q_grouped.mean(dim=2)
                else:
                    q_for_qw = query_states
            key_states = patcher._quantize_and_track(
                key_states, query_states=q_for_qw)

            # Handle KV cache
            if past_key_value is not None:
                # For transformers 4.x DynamicCache
                if hasattr(past_key_value, 'update'):
                    key_states, value_states = past_key_value.update(
                        key_states, value_states,
                        attn_module.layer_idx if hasattr(attn_module, 'layer_idx') else 0,
                        {"cache_position": cache_position} if cache_position is not None else None)
                else:
                    # Legacy tuple cache
                    past_key, past_value = past_key_value
                    key_states = torch.cat([past_key, key_states], dim=2)
                    value_states = torch.cat([past_value, value_states], dim=2)

            # GQA: repeat K,V for grouped query attention
            if num_kv_heads != num_heads:
                n_rep = num_heads // num_kv_heads
                key_states = key_states[:, :, None, :, :].expand(
                    bsz, num_kv_heads, n_rep, -1, head_dim
                ).reshape(bsz, num_heads, -1, head_dim)
                value_states = value_states[:, :, None, :, :].expand(
                    bsz, num_kv_heads, n_rep, -1, head_dim
                ).reshape(bsz, num_heads, -1, head_dim)

            # Manual attention computation (eager mode)
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

            if attention_mask is not None:
                causal_mask = attention_mask
                if causal_mask.dim() == 2:
                    # (batch, seq) -> (batch, 1, 1, seq)
                    causal_mask = causal_mask[:, None, None, :]
                elif causal_mask.dim() == 3:
                    causal_mask = causal_mask[:, None, :, :]
                # 4D mask: (batch, 1, q_len, kv_len)
                attn_weights = attn_weights + causal_mask

            attn_weights = F.softmax(
                attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)
            attn_output = attn_module.o_proj(attn_output)

            # Qwen2/Llama decoder expects: (attn_output, attn_weights, past_kv)
            # attn_weights=None when output_attentions=False
            # past_kv=None when use_cache=False
            outputs = (attn_output,
                       attn_weights if output_attentions else None,
                       past_key_value if use_cache else None)

            return outputs

        return patched_forward


# ============================================================================
# Non-Overlapping Chunk Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_ppl_chunked(model, input_ids: torch.Tensor,
                         chunk_len: int) -> Dict[str, float]:
    """Non-overlapping chunk PPL evaluation.

    Same protocol as KVQuant/GPTQ: split sequence into chunks of chunk_len,
    forward each independently, compute NLL on all tokens.

    No sliding window, no overlap. Each chunk is independent.
    """
    total_tokens = input_ids.shape[1]
    n_chunks = total_tokens // chunk_len

    if n_chunks == 0:
        raise ValueError(f"Sequence length {total_tokens} < chunk_len {chunk_len}")

    # Truncate to exact multiple of chunk_len
    input_ids = input_ids[:, :n_chunks * chunk_len]

    total_nll = 0.0
    total_count = 0
    t0 = time.time()

    for i in range(n_chunks):
        start = i * chunk_len
        end = start + chunk_len
        chunk = input_ids[:, start:end]

        outputs = model(chunk, use_cache=False)
        logits = outputs.logits.float()

        # Shift: logits[:, :-1] predicts tokens[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='sum')

        total_nll += loss.item()
        total_count += shift_labels.numel()

        del outputs, logits
        torch.cuda.empty_cache()

        if (i + 1) % 10 == 0 or i == 0:
            running_ppl = math.exp(min(total_nll / max(total_count, 1), 100.0))
            elapsed = time.time() - t0
            print(f"    chunk {i+1}/{n_chunks} | tokens={total_count} | "
                  f"ppl={running_ppl:.2f} | {elapsed:.1f}s", flush=True)

    ppl = math.exp(total_nll / max(total_count, 1))
    elapsed = time.time() - t0

    return {
        "ppl": ppl,
        "total_nll": total_nll,
        "total_tokens": total_count,
        "n_chunks": n_chunks,
        "runtime_s": elapsed,
    }


# ============================================================================
# Needle-in-a-Haystack (NIAH) Evaluation
# ============================================================================

NIAH_NEEDLES = [
    ("The secret verification code is 7392.", "What is the secret verification code?", "7392"),
    ("The hidden password for the vault is ALPHA-BRAVO-9.", "What is the hidden password for the vault?", "ALPHA-BRAVO-9"),
    ("The special launch sequence is 4-7-2-9-1.", "What is the special launch sequence?", "4-7-2-9-1"),
]

NIAH_FILLER_TOPICS = [
    "The history of maritime exploration spans thousands of years, from ancient Polynesian wayfinding to modern GPS navigation.",
    "Advances in renewable energy have transformed how nations approach electricity generation and distribution.",
    "The development of programming languages from assembly to modern high-level languages reflects changing computational needs.",
    "Agricultural practices have evolved from subsistence farming to precision agriculture using satellite imagery.",
    "Medical research continues to push boundaries in understanding genetic disorders and developing targeted therapies.",
    "Urban planning challenges include managing population density, transportation infrastructure, and green spaces.",
    "The evolution of financial markets has been shaped by technological innovation and regulatory frameworks.",
    "Climate science integrates atmospheric physics, ocean chemistry, and ecological modeling to project future conditions.",
    "Telecommunications networks have progressed from telegraph wires to fiber optic cables and satellite constellations.",
    "Archaeological discoveries continue to reshape our understanding of ancient civilizations and human migration.",
    "Materials science enables breakthroughs from semiconductor fabrication to biocompatible implant design.",
    "The philosophy of science examines how empirical methods generate reliable knowledge about natural phenomena.",
    "Space exploration missions have revealed details about planetary geology, atmospheric composition, and potential habitability.",
    "Cognitive psychology studies how attention, memory, and perception interact during complex decision making.",
    "International trade agreements balance economic growth objectives with environmental and labor protections.",
]


def _build_haystack(tokenizer, context_len: int, needle_text: str,
                    depth: float, seed: int = 42) -> str:
    """Build a haystack of approximately context_len tokens with needle at depth."""
    rng = np.random.RandomState(seed)

    # Build filler text by repeating and shuffling topics
    filler_paragraphs = list(NIAH_FILLER_TOPICS)
    filler_text = ""
    while True:
        rng.shuffle(filler_paragraphs)
        candidate = filler_text + "\n\n".join(filler_paragraphs) + "\n\n"
        tokens = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        if len(tokens) >= context_len * 2:
            break
        filler_text = candidate

    # Tokenize filler, truncate to make room for needle
    needle_tokens = tokenizer(needle_text, add_special_tokens=False)["input_ids"]
    target_filler_tokens = context_len - len(needle_tokens) - 10  # margin

    filler_tokens = tokenizer(filler_text, add_special_tokens=False)["input_ids"]
    filler_tokens = filler_tokens[:target_filler_tokens]

    # Insert needle at depth position
    insert_pos = max(1, int(len(filler_tokens) * depth))
    combined = filler_tokens[:insert_pos] + needle_tokens + filler_tokens[insert_pos:]
    combined = combined[:context_len]  # exact truncation

    return tokenizer.decode(combined)


@torch.no_grad()
def evaluate_niah_single(model, tokenizer, context_text: str,
                         question: str, answer: str,
                         max_new_tokens: int = 30) -> float:
    """Evaluate a single NIAH instance. Returns 1.0 if answer found, 0.0 otherwise."""
    prompt = context_text + f"\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=model.config.max_position_embeddings
                       if hasattr(model.config, 'max_position_embeddings') else 32768)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
    )
    # Decode only new tokens
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Check if answer is in response
    return 1.0 if answer.lower() in response.lower() else 0.0


@torch.no_grad()
def evaluate_niah_method(model, tokenizer, method: str, bits: int,
                         gamma: float, protocol: str,
                         context_lens: List[int], depths: List[float],
                         n_repeats: int = 3) -> Dict:
    """Run full NIAH evaluation for a single method/bits combination."""
    results = {}

    for ctx_len in context_lens:
        results[str(ctx_len)] = {}
        for depth in depths:
            scores = []
            for rep in range(n_repeats):
                # Pick needle (cycle through available needles)
                needle_idx = rep % len(NIAH_NEEDLES)
                needle_text, question, answer = NIAH_NEEDLES[needle_idx]

                # Build haystack
                haystack = _build_haystack(
                    tokenizer, ctx_len, needle_text, depth,
                    seed=42 + rep * 1000 + int(depth * 100))

                score = evaluate_niah_single(
                    model, tokenizer, haystack, question, answer)
                scores.append(score)

            avg_score = float(np.mean(scores))
            results[str(ctx_len)][f"{depth:.1f}"] = avg_score

    # Compute averages
    all_scores = [results[cl][d] for cl in results for d in results[cl]]
    results["avg"] = float(np.mean(all_scores)) if all_scores else 0.0

    return results


def run_niah(args, model, tokenizer) -> Dict:
    """Run NIAH evaluation for all methods/bits combinations."""
    print("\n" + "=" * 72)
    print("NIAH (Needle-in-a-Haystack) Evaluation")
    print(f"Context lengths: {args.niah_context_lens}")
    print(f"Depths: {args.niah_depths}")
    print(f"Repeats: {args.niah_repeats}")
    print("=" * 72)

    niah_results = {}
    t0 = time.time()

    # FP16 baseline
    if "fp16" in args.methods:
        print("\n--- NIAH: FP16 ---")
        result = evaluate_niah_method(
            model, tokenizer, "fp16", 16, args.gamma, args.protocol,
            args.niah_context_lens, args.niah_depths, args.niah_repeats)
        niah_results["fp16"] = result
        print(f"  Avg score: {result['avg']:.3f}")

    # Quantized methods
    for method in [m for m in args.methods if m != "fp16"]:
        niah_results[method] = {}
        for bits in args.bits:
            key = f"{method}_{bits}bit"
            print(f"\n--- NIAH: {key} ({args.protocol}) ---")

            # Install quantization
            if args.protocol == "post_rope":
                patcher = AttentionKQuantPatcher(model, method, bits, args.gamma)
                patcher.patch()
                patcher.active = True
                patcher.reset_stats()
            elif args.protocol == "pre_rope":
                hooks = install_pre_rope_hooks(model, method, bits, args.gamma)
                for h in hooks:
                    h.active = True
                    h.reset_stats()

            try:
                result = evaluate_niah_method(
                    model, tokenizer, method, bits, args.gamma, args.protocol,
                    args.niah_context_lens, args.niah_depths, args.niah_repeats)
                niah_results[method][str(bits)] = result
                print(f"  Avg score: {result['avg']:.3f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                niah_results[method][str(bits)] = {"avg": 0.0, "error": str(e)}
            finally:
                if args.protocol == "post_rope":
                    patcher.active = False
                    patcher.unpatch()
                elif args.protocol == "pre_rope":
                    for h in hooks:
                        h.active = False
                    remove_hooks(hooks)

            torch.cuda.empty_cache()
            gc.collect()

    elapsed = time.time() - t0

    # Print summary table
    print("\n" + "=" * 72)
    print("NIAH SUMMARY")
    print("=" * 72)
    fp16_avg = niah_results.get("fp16", {}).get("avg", 0.0)
    print(f"  FP16 baseline: {fp16_avg:.3f}")
    print(f"  {'Method':<15} {'2bit':>8} {'3bit':>8} {'4bit':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8}")
    for method in [m for m in args.methods if m != "fp16"]:
        row = f"  {method:<15}"
        for bits in args.bits:
            r = niah_results.get(method, {}).get(str(bits), {})
            avg = r.get("avg", 0.0) if isinstance(r, dict) else 0.0
            row += f" {avg:>8.3f}"
        print(row)
    print(f"  Total NIAH time: {elapsed:.1f}s")

    return niah_results


# ============================================================================
# Main Evaluation Dispatch
# ============================================================================

@torch.no_grad()
def evaluate_method(model, input_ids: torch.Tensor, chunk_len: int,
                    method: str, bits: int, gamma: float,
                    protocol: str) -> Dict[str, float]:
    """Evaluate a single method/bits combination."""

    if method == "fp16":
        return evaluate_ppl_chunked(model, input_ids, chunk_len)

    if protocol == "pre_rope":
        # Install pre-RoPE hooks
        hooks = install_pre_rope_hooks(model, method, bits, gamma)
        try:
            with k_quantization_active(hooks):
                result = evaluate_ppl_chunked(model, input_ids, chunk_len)

            # Collect MSE stats
            total_mse_sum = sum(h.key_mse_sum for h in hooks)
            total_mse_count = sum(h.key_mse_count for h in hooks)
            result["avg_key_mse"] = (total_mse_sum / total_mse_count
                                      if total_mse_count > 0 else 0.0)
            result["quantization_point"] = "pre_rope"
        finally:
            remove_hooks(hooks)
        return result

    elif protocol == "post_rope":
        # Install post-RoPE attention patches
        patcher = AttentionKQuantPatcher(model, method, bits, gamma)
        patcher.patch()
        patcher.active = True
        patcher.reset_stats()

        try:
            result = evaluate_ppl_chunked(model, input_ids, chunk_len)
            result["avg_key_mse"] = (patcher.key_mse_sum / patcher.key_mse_count
                                      if patcher.key_mse_count > 0 else 0.0)
            result["quantization_point"] = "post_rope"
        finally:
            patcher.active = False
            patcher.unpatch()

        return result

    else:
        raise ValueError(f"Unknown protocol: {protocol}")


# ============================================================================
# Self-Test
# ============================================================================

def run_self_tests(seed: int) -> None:
    set_seed(seed)
    print("Running self-tests...")

    # Test 1: Quantization functions
    t = torch.randn(8, 16)
    for bits in [2, 3, 4]:
        q = uniform_quantize_tensor(t, bits)
        assert q.shape == t.shape and torch.isfinite(q).all()
    print("  [PASS] uniform_quantize_tensor")

    for bits in [2, 3, 4]:
        q = kivi_quantize_tensor(t, bits)
        assert q.shape == t.shape and torch.isfinite(q).all()
    print("  [PASS] kivi_quantize_tensor")

    K = torch.randn(64, 32) + 5.0
    K_q, r_eff = fokvq_quantize_head(K, 4, 0.3)
    assert K_q.shape == K.shape and torch.isfinite(K_q).all()
    print("  [PASS] fokvq_quantize_head")

    # Test 1b: fokvq_qw_quantize_head basic
    Q_cov_test = torch.eye(32) * 2.0 + torch.randn(32, 32) * 0.1
    Q_cov_test = (Q_cov_test + Q_cov_test.T) / 2  # symmetric
    K_qw, r_eff_qw = fokvq_qw_quantize_head(K, Q_cov_test, 4, 0.3)
    assert K_qw.shape == K.shape and torch.isfinite(K_qw).all()
    print("  [PASS] fokvq_qw_quantize_head (basic)")

    # Test 1c: fokvq_qw produces different axes than fokvq
    # When Q has strong directional preference, QW should differ
    Q_cov_aniso = torch.zeros(32, 32)
    Q_cov_aniso[0, 0] = 100.0  # Q variance concentrated in dim 0
    Q_cov_aniso[1, 1] = 50.0
    for i in range(2, 32):
        Q_cov_aniso[i, i] = 0.01
    K_qw_aniso, _ = fokvq_qw_quantize_head(K, Q_cov_aniso, 4, 0.3)
    K_fokvq_plain, _ = fokvq_quantize_head(K, 4, 0.3)
    # They should differ (different rotation axes)
    diff_axes = (K_qw_aniso - K_fokvq_plain).abs().mean().item()
    print(f"  [INFO] FOKVQ vs FOKVQ-QW mean abs diff: {diff_axes:.6f}")
    assert diff_axes > 1e-6, "QW and plain FOKVQ should differ with anisotropic Q"
    print("  [PASS] fokvq_qw produces different results from fokvq (anisotropic Q)")

    # Test 2: quantize_k_tensor with different dims
    K_2d = torch.randn(32, 16)
    K_3d = torch.randn(4, 32, 16)
    K_4d = torch.randn(1, 4, 32, 16)
    for method in ["uniform", "kivi", "fokvq"]:
        q2 = quantize_k_tensor(K_2d, method, 3)
        q3 = quantize_k_tensor(K_3d, method, 3)
        q4 = quantize_k_tensor(K_4d, method, 3)
        assert q2.shape == K_2d.shape
        assert q3.shape == K_3d.shape
        assert q4.shape == K_4d.shape
    # Test fokvq_qw with q_covs
    q_cov_2d = torch.eye(16)
    q_cov_3d = torch.eye(16).unsqueeze(0).expand(4, -1, -1)
    q_cov_4d = torch.eye(16).unsqueeze(0).unsqueeze(0).expand(1, 4, -1, -1)
    q2 = quantize_k_tensor(K_2d, "fokvq_qw", 3, q_covs=q_cov_2d)
    q3 = quantize_k_tensor(K_3d, "fokvq_qw", 3, q_covs=q_cov_3d)
    q4 = quantize_k_tensor(K_4d, "fokvq_qw", 3, q_covs=q_cov_4d)
    assert q2.shape == K_2d.shape
    assert q3.shape == K_3d.shape
    assert q4.shape == K_4d.shape
    # Test fokvq_qw fallback (no q_covs -> regular fokvq)
    q2_fb = quantize_k_tensor(K_2d, "fokvq_qw", 3)
    assert q2_fb.shape == K_2d.shape
    print("  [PASS] quantize_k_tensor (2D/3D/4D, incl. fokvq_qw)")

    # Test 3: KProjQuantHook shape handling
    hook = KProjQuantHook(num_kv_heads=4, d_head=16, method="fokvq",
                          bits=3, gamma=0.3)
    hook.active = True
    fake_output = torch.randn(1, 32, 64)  # (batch, seq, 4*16)
    result = hook(None, None, fake_output)
    assert result.shape == fake_output.shape
    assert torch.isfinite(result).all()
    assert hook.key_mse_count > 0
    print(f"  [PASS] KProjQuantHook (MSE={hook.key_mse_sum/hook.key_mse_count:.6f})")

    # Test 4: FOKVQ beats uniform on anisotropic data
    np.random.seed(seed)
    evals_arr = np.array([100, 50, 10, 1] + [0.01] * 28, dtype=np.float32)
    U = np.linalg.qr(np.random.randn(32, 32).astype(np.float32))[0]
    data = (np.random.randn(128, 32).astype(np.float32)
            @ np.diag(np.sqrt(evals_arr)) @ U.T)
    K_aniso = torch.from_numpy(data) + 3.0
    K_q_fokvq, _ = fokvq_quantize_head(K_aniso, 3, 0.3)
    K_q_unif = uniform_quantize_tensor(K_aniso, 3)
    mse_f = (K_aniso - K_q_fokvq).pow(2).mean().item()
    mse_u = (K_aniso - K_q_unif).pow(2).mean().item()
    print(f"  [INFO] Anisotropic: FOKVQ 3bit MSE={mse_f:.6f}, "
          f"Uniform 3bit MSE={mse_u:.6f}")
    if mse_f < mse_u:
        print("  [PASS] FOKVQ beats uniform on anisotropic data")
    else:
        print("  [WARN] FOKVQ did not beat uniform")

    # Test 4b: FOKVQ-QW inner product error vs FOKVQ on structured data
    # When Q has a strong directional preference, QW should give lower
    # Q·K inner product error even if K-space MSE is not necessarily lower.
    np.random.seed(seed + 1)
    d_test = 32
    # Q concentrated in first 4 dims, K spread across all dims
    Q_evals = np.array([100, 50, 25, 10] + [0.01] * 28, dtype=np.float32)
    K_evals = np.array([10, 8, 6, 4] + [2.0] * 28, dtype=np.float32)
    U_q = np.linalg.qr(np.random.randn(d_test, d_test).astype(np.float32))[0]
    U_k = np.linalg.qr(np.random.randn(d_test, d_test).astype(np.float32))[0]
    Q_data = (np.random.randn(64, d_test).astype(np.float32)
              @ np.diag(np.sqrt(Q_evals)) @ U_q.T)
    K_data = (np.random.randn(64, d_test).astype(np.float32)
              @ np.diag(np.sqrt(K_evals)) @ U_k.T)
    Q_t = torch.from_numpy(Q_data)
    K_t = torch.from_numpy(K_data)
    Q_cov_emp = (Q_t.T @ Q_t) / (Q_t.shape[0] - 1)
    # Ground truth inner product: Q @ K^T
    ip_true = Q_t @ K_t.T
    # FOKVQ reconstruction
    K_fokvq_r, _ = fokvq_quantize_head(K_t, 3, 0.3)
    ip_fokvq = Q_t @ K_fokvq_r.T
    err_fokvq = (ip_true - ip_fokvq).pow(2).mean().item()
    # FOKVQ-QW reconstruction
    K_qw_r, _ = fokvq_qw_quantize_head(K_t, Q_cov_emp, 3, 0.3)
    ip_qw = Q_t @ K_qw_r.T
    err_qw = (ip_true - ip_qw).pow(2).mean().item()
    print(f"  [INFO] Q·K inner product MSE: FOKVQ={err_fokvq:.4f}, "
          f"FOKVQ-QW={err_qw:.4f}, ratio={err_qw/max(err_fokvq,1e-10):.3f}")
    if err_qw < err_fokvq:
        print("  [PASS] FOKVQ-QW beats FOKVQ on Q·K inner product error")
    else:
        print("  [WARN] FOKVQ-QW did not beat FOKVQ on Q·K inner product error "
              "(may happen with certain random seeds)")

    # Test 5: MSE monotonicity
    K = torch.randn(64, 32) + 2.0
    mses = {}
    for bits in [2, 3, 4]:
        K_q, _ = fokvq_quantize_head(K, bits, 0.3)
        mses[bits] = (K - K_q).pow(2).mean().item()
    assert mses[4] < mses[3] < mses[2], f"Monotonicity failed: {mses}"
    print(f"  [PASS] MSE monotonicity (fokvq): 2b={mses[2]:.4f} > 3b={mses[3]:.4f} > 4b={mses[4]:.4f}")

    # Test 5b: MSE monotonicity for fokvq_qw
    Q_cov_mono = torch.eye(32) + torch.randn(32, 32) * 0.05
    Q_cov_mono = (Q_cov_mono + Q_cov_mono.T) / 2
    mses_qw = {}
    for bits in [2, 3, 4]:
        K_q, _ = fokvq_qw_quantize_head(K, Q_cov_mono, bits, 0.3)
        mses_qw[bits] = (K - K_q).pow(2).mean().item()
    assert mses_qw[4] < mses_qw[3] < mses_qw[2], f"QW Monotonicity failed: {mses_qw}"
    print(f"  [PASS] MSE monotonicity (fokvq_qw): 2b={mses_qw[2]:.4f} > 3b={mses_qw[3]:.4f} > 4b={mses_qw[4]:.4f}")

    # Test 6: E2 (Lloyd-Max), E3 (MK), full (E1+E2+E3) basic correctness
    K_test = torch.randn(64, 32) + 3.0
    Q_cov_test = torch.eye(32) + torch.randn(32, 32) * 0.1
    Q_cov_test = (Q_cov_test + Q_cov_test.T) / 2

    K_e2, r_e2 = fokvq_e2_quantize_head(K_test, 3, 0.3)
    K_e3, r_e3 = fokvq_e3_quantize_head(K_test, 3, 0.3)
    K_full, r_full = fokvq_full_quantize_head(K_test, 3, 0.3, Q_cov=Q_cov_test)
    assert K_e2.shape == K_test.shape and torch.isfinite(K_e2).all()
    assert K_e3.shape == K_test.shape and torch.isfinite(K_e3).all()
    assert K_full.shape == K_test.shape and torch.isfinite(K_full).all()
    print("  [PASS] E2/E3/full basic shape and finiteness")

    # E2/E3 should produce different results from base FOKVQ
    K_base, _ = fokvq_quantize_head(K_test, 3, 0.3)
    diff_e2 = (K_e2 - K_base).abs().mean().item()
    diff_e3 = (K_e3 - K_base).abs().mean().item()
    diff_full = (K_full - K_base).abs().mean().item()
    print(f"  [INFO] Mean abs diff vs base: E2={diff_e2:.6f}, E3={diff_e3:.6f}, full={diff_full:.6f}")

    # Test 7: E2/E3/full MSE comparison on anisotropic data
    np.random.seed(seed)
    evals_arr = np.array([100, 50, 10, 1] + [0.01] * 28, dtype=np.float32)
    U = np.linalg.qr(np.random.randn(32, 32).astype(np.float32))[0]
    data = (np.random.randn(128, 32).astype(np.float32)
            @ np.diag(np.sqrt(evals_arr)) @ U.T)
    K_aniso = torch.from_numpy(data) + 3.0

    mse_base = (K_aniso - fokvq_quantize_head(K_aniso, 3, 0.3)[0]).pow(2).mean().item()
    mse_e2 = (K_aniso - fokvq_e2_quantize_head(K_aniso, 3, 0.3)[0]).pow(2).mean().item()
    mse_e3 = (K_aniso - fokvq_e3_quantize_head(K_aniso, 3, 0.3)[0]).pow(2).mean().item()
    mse_unif = (K_aniso - uniform_quantize_tensor(K_aniso, 3)).pow(2).mean().item()
    print(f"  [INFO] Anisotropic 3bit MSE: uniform={mse_unif:.6f}, "
          f"fokvq={mse_base:.6f}, E2={mse_e2:.6f}, E3={mse_e3:.6f}")

    # Test 8: quantize_k_tensor dispatches E2/E3/full correctly
    K_4d = torch.randn(1, 4, 32, 16)
    for m in ["fokvq_e2", "fokvq_e3", "fokvq_full"]:
        q = quantize_k_tensor(K_4d, m, 3)
        assert q.shape == K_4d.shape and torch.isfinite(q).all()
    print("  [PASS] quantize_k_tensor dispatches E2/E3/full correctly")

    # Test 9: MSE monotonicity for E2 and full
    K_mono = torch.randn(64, 32) + 2.0
    for label, fn in [("E2", fokvq_e2_quantize_head), ("E3", fokvq_e3_quantize_head)]:
        mses_x = {}
        for bits in [2, 3, 4]:
            K_q, _ = fn(K_mono, bits, 0.3)
            mses_x[bits] = (K_mono - K_q).pow(2).mean().item()
        assert mses_x[4] < mses_x[3] < mses_x[2], f"{label} monotonicity failed: {mses_x}"
        print(f"  [PASS] MSE monotonicity ({label}): 2b={mses_x[2]:.4f} > 3b={mses_x[3]:.4f} > 4b={mses_x[4]:.4f}")

    print("\nAll self-tests passed.")


# ============================================================================
# Main
# ============================================================================

def run() -> None:
    args = parse_args()

    if args.self_test:
        run_self_tests(args.seed)
        return

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = resolve_dtype(args.model_name, args.dtype)
    cache_dir = args.cache_dir or None

    print("=" * 72)
    print("FOKVQ Exp 4-2 v3: Full KV Cache Quantization PPL Benchmark")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {args.model_name} ({args.model_key})")
    print(f"Device: {args.device}, dtype: {dtype}")
    print(f"Chunk length: {args.context_len} (non-overlapping)")
    print(f"Protocol: {args.protocol}")
    print(f"Methods: {args.methods}, Bits: {args.bits}")
    print(f"FOKVQ gamma: {args.gamma}")
    print("=" * 72)
    print()
    print("KEY DIFFERENCE vs v2:")
    print("  v2: sliding window, only prefix K quantized (50% of window)")
    print("  v3: non-overlapping chunks, ALL K quantized via hooks (100%)")
    print()

    # Load model
    print("Loading model...")
    tokenizer, model = load_model_and_tokenizer(
        args.model_name, dtype, args.device, cache_dir, args.attn_implementation)

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    n_kv_heads = getattr(cfg, 'num_key_value_heads', n_heads)
    d_head = cfg.hidden_size // n_heads
    print(f"  layers={n_layers}, heads={n_heads}, kv_heads={n_kv_heads}, "
          f"d_head={d_head}")

    # Validate protocol choice
    if args.protocol == "post_rope":
        model_type = getattr(cfg, 'model_type', '').lower()
        if 'gpt2' not in model_type and not hasattr(model, 'model'):
            print("  WARNING: post_rope protocol may not work with this model type")

    # ================================================================
    # MODE: PPL
    # ================================================================
    summary = {
        "experiment": "exp4_2_v3",
        "mode": args.mode,
        "model_key": args.model_key,
        "model_name": args.model_name,
        "device": args.device,
        "dtype": str(dtype),
        "protocol": args.protocol,
        "fokvq_gamma": args.gamma,
        "methods": args.methods,
        "bits": args.bits,
    }

    t0 = time.time()

    if args.mode in ("ppl", "both"):
        print(f"\n{'='*72}")
        print("MODE: WikiText-2 PPL (v3: 100% K quantized)")
        print(f"{'='*72}")

        # Load data
        print("\nLoading WikiText-2...")
        input_ids = load_wikitext2_ids(tokenizer, args.device, cache_dir)
        if args.max_eval_tokens > 0:
            input_ids = input_ids[:, :args.max_eval_tokens]
            print(f"  Truncated to {input_ids.shape[1]} tokens")

        summary["chunk_len"] = args.context_len
        summary["total_tokens_in_corpus"] = int(input_ids.shape[1])
        summary["ppl_results"] = {}

        # --- FP16 Baseline ---
        if "fp16" in args.methods:
            print("\n--- FP16 Baseline ---")
            result = evaluate_ppl_chunked(model, input_ids, args.context_len)
            summary["ppl_results"]["fp16"] = result
            print(f"  PPL = {result['ppl']:.4f} "
                  f"({result['total_tokens']} tokens, {result['runtime_s']:.1f}s)")

        # --- Quantized Methods ---
        for method in [m for m in args.methods if m != "fp16"]:
            summary["ppl_results"][method] = {}
            for bits in args.bits:
                key = f"{method}_{bits}bit"
                print(f"\n--- {key} ({args.protocol}) ---")
                try:
                    result = evaluate_method(
                        model, input_ids, args.context_len,
                        method, bits, args.gamma, args.protocol)
                    summary["ppl_results"][method][str(bits)] = result
                    mse_str = (f", key_MSE={result.get('avg_key_mse', 0):.6f}"
                              if 'avg_key_mse' in result else "")
                    print(f"  PPL = {result['ppl']:.4f} "
                          f"({result['total_tokens']} tokens, "
                          f"{result['runtime_s']:.1f}s{mse_str})")
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    summary["ppl_results"][method][str(bits)] = {
                        "ppl": float('inf'), "error": str(e),
                    }
                torch.cuda.empty_cache()
                gc.collect()

        # PPL Summary Table
        print("\n" + "=" * 72)
        print("PPL SUMMARY (v3: 100% K quantized)")
        print("=" * 72)
        fp16_ppl = summary["ppl_results"].get("fp16", {}).get("ppl", float('inf'))
        print(f"  FP16 baseline: {fp16_ppl:.2f}")
        print(f"  {'Method':<15} {'2bit':>12} {'3bit':>12} {'4bit':>12}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
        for method in [m for m in args.methods if m != "fp16"]:
            row = f"  {method:<15}"
            for bits in args.bits:
                ppl = summary["ppl_results"].get(method, {}).get(
                    str(bits), {}).get("ppl", float('inf'))
                if math.isfinite(ppl):
                    if math.isfinite(fp16_ppl) and fp16_ppl > 0:
                        delta = (ppl - fp16_ppl) / fp16_ppl * 100
                        row += f" {ppl:>7.2f}({delta:+.0f}%)"
                    else:
                        row += f" {ppl:>12.2f}"
                else:
                    row += f" {'N/A':>12}"
            print(row)
        print("=" * 72)

        del input_ids
        torch.cuda.empty_cache()

    # ================================================================
    # MODE: NIAH
    # ================================================================
    if args.mode in ("niah", "both"):
        niah_results = run_niah(args, model, tokenizer)
        summary["niah_results"] = niah_results

    # ================================================================
    # Write results
    # ================================================================
    summary["runtime_s"] = time.time() - t0
    if torch.cuda.is_available():
        summary["peak_memory_gib"] = (
            torch.cuda.max_memory_allocated(torch.device(args.device)) / (1024 ** 3))

    out_path = output_dir / f"{args.model_key}_{args.mode}_v3.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")
    print(f"Total runtime: {summary['runtime_s']:.1f}s")

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    run()
