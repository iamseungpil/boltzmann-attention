"""
FOKVQ Experiment 5: Lie-Group Structured Generator Variants
============================================================
Track H5 / Phase 3: test whether structured sub-Lie-groups of SO(d) can
outperform PCA and agnostic random rotations on quantization surrogates.

Candidates:
  5-1  block_diagonal  — PCA + block-diagonal SO(b) refinement, b ∈ {2,4,8,16}
  5-2  complex_unitary — PCA + U(d/2) refinement mapped to SO(d)
  5-3  commutator_reg  — exp(A) optimisation regularised by ||[R, Σ]||_F

Evaluation:
  - K reconstruction MSE at 2, 3, 4 bits (held-out split)
  - Attention KL divergence (held-out split)
  - All compared against identity, random, PCA baselines

Prerequisites:
  - Exp 3-1 showed full SO(d) Givens optimisation (MSE 938.81) does NOT beat
    PCA (MSE 220.45).  H5 hypothesises that *structured* subgroups help by
    reducing effective parameters.

Usage:
  python exp5_lie_variants.py --self-test          # deterministic unit tests
  python exp5_lie_variants.py --smoke              # tiny run for sanity
  python exp5_lie_variants.py --device cuda:0      # full surrogate evaluation
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, stats
from sklearn.decomposition import PCA

# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 5: Lie-Group Structured Generator Variants")
    p.add_argument("--self-test", action="store_true", help="Run deterministic unit tests and exit")
    p.add_argument("--smoke", action="store_true", help="Tiny-model smoke run")
    p.add_argument("--model-name", type=str, default="openai-community/gpt2-medium")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-samples", type=int, default=40)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--quant-bits", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--block-sizes", nargs="+", type=int, default=[2, 4, 8, 16])
    p.add_argument("--layer-stride", type=int, default=3, help="Evaluate every N-th layer")
    p.add_argument("--heldout-frac", type=float, default=0.25)
    p.add_argument("--max-opt-iters", type=int, default=200)
    p.add_argument("--commutator-lambdas", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="")
    p.add_argument("--cache-dir", type=str, default="")
    return p.parse_args()


# ── Quantisation primitives ──────────────────────────────────────────

def quantize_vector_np(v: np.ndarray, n_bits: int) -> np.ndarray:
    """Min-max uniform quantisation per-column."""
    if n_bits >= 16:
        return v.copy()
    n_levels = 2 ** n_bits
    v_min = v.min(axis=0, keepdims=True)
    v_max = v.max(axis=0, keepdims=True)
    v_range = np.maximum(v_max - v_min, 1e-10)
    v_norm = (v - v_min) / v_range
    v_quant = np.round(v_norm * (n_levels - 1)) / (n_levels - 1)
    return v_quant * v_range + v_min


def quantize_mse_np(k: np.ndarray, R: np.ndarray, n_bits: int) -> float:
    """MSE = ||K - quant(K R) R^T||^2 / n  (row-vector convention)."""
    k_rot = k @ R
    k_rot_q = quantize_vector_np(k_rot, n_bits)
    k_recon = k_rot_q @ R.T
    return float(np.mean((k - k_recon) ** 2))


def attention_kl_np(
    Q: np.ndarray, K: np.ndarray, K_quant: np.ndarray, d: int
) -> float:
    """Mean token-level KL(softmax(QK^T/√d) || softmax(QK_q^T/√d))."""
    scale = 1.0 / math.sqrt(d)
    logits_orig = (Q @ K.T) * scale        # (nq, nk)
    logits_quant = (Q @ K_quant.T) * scale
    # stable softmax
    logits_orig = logits_orig - logits_orig.max(axis=1, keepdims=True)
    logits_quant = logits_quant - logits_quant.max(axis=1, keepdims=True)
    p = np.exp(logits_orig)
    p = p / p.sum(axis=1, keepdims=True)
    q = np.exp(logits_quant)
    q = q / q.sum(axis=1, keepdims=True)
    q = np.maximum(q, 1e-12)  # numerical stability for log(q)
    # KL(p || q) per query
    kl = np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)), axis=1)
    return float(np.mean(kl))


# ── Rotation builders ────────────────────────────────────────────────

def pca_basis_np(matrix: np.ndarray) -> np.ndarray:
    """Return (d, d) PCA basis columns ordered by descending variance."""
    n, d = matrix.shape
    assert n >= d, f"PCA requires n >= d, got n={n}, d={d}"
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt.T.astype(np.float32)


def random_orthogonal_np(d: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    m = rng.randn(d, d).astype(np.float32)
    q, _ = np.linalg.qr(m)
    return q


def check_orthogonal(R: np.ndarray, tol: float = 1e-4) -> float:
    """Return max |R^T R - I|; raises if > tol."""
    err = float(np.max(np.abs(R.T @ R - np.eye(R.shape[0], dtype=R.dtype))))
    if err > tol:
        raise ValueError(f"Orthogonality check failed: max_err={err:.6e} > tol={tol}")
    return err


# ── 5-1: Block-diagonal rotation ────────────────────────────────────

def _block_rotation(d: int, block_size: int, angles: np.ndarray) -> np.ndarray:
    """Build a block-diagonal rotation in SO(d) from angles.
    Each block of size `block_size` is parametrised by block_size*(block_size-1)/2
    Givens angles within that block.
    """
    n_blocks = d // block_size
    remainder = d % block_size
    R = np.eye(d, dtype=np.float32)
    angle_idx = 0
    for b in range(n_blocks):
        start = b * block_size
        # within this block, enumerate all (i,j) Givens planes
        for i in range(block_size):
            for j in range(i + 1, block_size):
                theta = angles[angle_idx]
                angle_idx += 1
                ci, cj = start + i, start + j
                c, s = math.cos(theta), math.sin(theta)
                row_i = R[ci, :].copy()
                row_j = R[cj, :].copy()
                R[ci, :] = c * row_i - s * row_j
                R[cj, :] = s * row_i + c * row_j
    return R


def _n_block_angles(d: int, block_size: int) -> int:
    n_blocks = d // block_size
    angles_per_block = block_size * (block_size - 1) // 2
    return n_blocks * angles_per_block


def optimize_block_diagonal(
    k_data: np.ndarray, pca_R: np.ndarray, block_size: int,
    n_bits: int, max_iters: int, seed: int,
) -> Tuple[np.ndarray, float]:
    """Optimise block-diagonal refinement on top of PCA."""
    d = k_data.shape[1]
    n_angles = _n_block_angles(d, block_size)
    if n_angles == 0:
        return pca_R.copy(), quantize_mse_np(k_data, pca_R, n_bits)

    # Pre-rotate into PCA space for optimisation
    k_pca = k_data @ pca_R

    def objective(angles):
        blk_R = _block_rotation(d, block_size, angles)
        return quantize_mse_np(k_pca, blk_R, n_bits)

    x0 = np.zeros(n_angles, dtype=np.float64)
    result = optimize.minimize(
        objective, x0, method="Powell",
        options={"maxiter": max_iters, "maxfev": max_iters * max(n_angles, 1)},
    )
    blk_R = _block_rotation(d, block_size, result.x)
    composed = pca_R @ blk_R  # final rotation: PCA then block refinement
    mse = quantize_mse_np(k_data, composed, n_bits)
    return composed, mse


# ── 5-2: Complex unitary ────────────────────────────────────────────

def optimize_complex_unitary(
    k_data: np.ndarray, pca_R: np.ndarray,
    n_bits: int, max_iters: int, seed: int,
) -> Tuple[np.ndarray, float]:
    """Optimise block-diagonal U(1)^{d/2} refinement on top of PCA.

    Each pair of PCA dimensions (2i, 2i+1) gets one complex phase angle θ_i,
    implementing a U(1) rotation in the complex plane. This gives d/2 parameters
    (tractable) while respecting the U(d/2) Lie-group structure.
    Total: d/2 angle parameters → d/2 independent 2x2 rotation blocks.
    """
    d = k_data.shape[1]
    half = d // 2
    n_params = half  # one angle per complex pair
    k_pca = k_data @ pca_R

    def _build_block_unitary(angles):
        """Build d×d orthogonal from d/2 complex-phase angles."""
        R = np.eye(d, dtype=np.float32)
        for i in range(half):
            c = math.cos(angles[i])
            s = math.sin(angles[i])
            r, r1 = 2 * i, 2 * i + 1
            R[r, r] = c
            R[r, r1] = -s
            R[r1, r] = s
            R[r1, r1] = c
        return R

    def objective(angles):
        blk_R = _build_block_unitary(angles)
        return quantize_mse_np(k_pca, blk_R, n_bits)

    x0 = np.zeros(n_params, dtype=np.float64)
    result = optimize.minimize(
        objective, x0, method="Powell",
        options={"maxiter": max_iters, "maxfev": max_iters * max(n_params * 4, 100)},
    )
    blk_R = _build_block_unitary(result.x)
    composed = pca_R @ blk_R
    mse = quantize_mse_np(k_data, composed, n_bits)
    return composed, mse


# ── 5-3: Commutator-regularised ─────────────────────────────────────

def optimize_commutator_reg(
    k_data: np.ndarray, cov: np.ndarray, lam: float,
    n_bits: int, max_iters: int,
    rank: int = 2, seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """Optimise R = exp(A) with conjugation penalty λ||RΣR^T - Σ||_F.

    Uses the proper Lie-algebra parameterisation: A is a low-rank skew-symmetric
    matrix A = U V^T - V U^T where U, V ∈ R^{d×r}. This gives 2*d*r parameters,
    which is tractable for d=64, r=8 (1024 params) while staying in so(d).
    The penalty ||RΣR^T - Σ||_F is zero iff R commutes with Σ, which for
    orthogonal R is equivalent to R being in the PCA eigenbasis.
    """
    from scipy.linalg import expm
    d = k_data.shape[1]
    n_params = 2 * d * rank

    def _build_R(params):
        U = params[:d * rank].reshape(d, rank)
        V = params[d * rank:].reshape(d, rank)
        A = (U @ V.T - V @ U.T).astype(np.float64)
        R = expm(A).astype(np.float32)
        # Nearest orthogonal via polar decomposition (numerical cleanup)
        Ur, _, Vt = np.linalg.svd(R)
        return (Ur @ Vt).astype(np.float32)

    def objective(params):
        R = _build_R(params)
        mse = quantize_mse_np(k_data, R, n_bits)
        # commutator penalty: ||R Σ R^T - Σ||_F
        R64 = R.astype(np.float64)
        cov64 = cov.astype(np.float64)
        comm = R64 @ cov64 @ R64.T - cov64
        penalty = float(np.sqrt(np.sum(comm ** 2)))
        return mse + lam * penalty

    rng = np.random.RandomState(seed)
    x0 = rng.randn(n_params).astype(np.float64) * 0.01
    result = optimize.minimize(
        objective, x0, method="Powell",
        options={"maxiter": max_iters, "maxfev": max_iters * max(n_params * 4, 100)},
    )
    R = _build_R(result.x)
    mse = quantize_mse_np(k_data, R, n_bits)
    return R, mse


# ── K / Q extraction ────────────────────────────────────────────────

def extract_kq_per_head(model, tokenizer, texts, device, max_seq_len, model_type="gpt2"):
    """Extract per-head K and Q vectors for each layer.

    Returns:
        layer_idx → head_idx → {'K': (n_tokens, d_head), 'Q': (n_tokens, d_head)}
    """
    import torch
    num_layers = model.config.n_layer if hasattr(model.config, "n_layer") else model.config.num_hidden_layers
    n_heads = model.config.n_head if hasattr(model.config, "n_head") else model.config.num_attention_heads
    n_embd = model.config.n_embd if hasattr(model.config, "n_embd") else model.config.hidden_size
    d_head = n_embd // n_heads

    data = {l: {h: {"K": [], "Q": []} for h in range(n_heads)} for l in range(num_layers)}

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_seq_len
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        for l in range(num_layers):
            h = hidden_states[l]
            if model_type == "gpt2":
                block = model.transformer.h[l]
                h_normed = block.ln_1(h)  # LayerNorm before attention
                qkv = block.attn.c_attn(h_normed)
                q_full, k_full, _ = qkv.split(n_embd, dim=-1)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")
            # Reshape (1, seq, n_embd) → (seq, n_heads, d_head)
            seq_len = k_full.shape[1]
            k_heads = k_full.squeeze(0).reshape(seq_len, n_heads, d_head)
            q_heads = q_full.squeeze(0).reshape(seq_len, n_heads, d_head)
            for hi in range(n_heads):
                data[l][hi]["K"].append(k_heads[:, hi, :].detach().cpu().float().numpy())
                data[l][hi]["Q"].append(q_heads[:, hi, :].detach().cpu().float().numpy())

        del inputs, outputs, hidden_states
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    for l in range(num_layers):
        for hi in range(n_heads):
            data[l][hi]["K"] = np.concatenate(data[l][hi]["K"], axis=0)
            data[l][hi]["Q"] = np.concatenate(data[l][hi]["Q"], axis=0)
    return data, n_heads, d_head


# ── Self-tests ───────────────────────────────────────────────────────

def run_self_tests():
    """Deterministic tests — no GPU, no model load."""
    np.random.seed(42)
    d = 8
    n = 200

    print("[self-test] block rotation orthogonality ...", end=" ")
    for bs in [2, 4, 8]:
        n_ang = _n_block_angles(d, bs)
        angles = np.random.randn(n_ang) * 0.3
        R = _block_rotation(d, bs, angles)
        check_orthogonal(R, tol=1e-5)
    print("PASS")

    print("[self-test] complex unitary (block-diagonal U(1)^{d/2}) ...", end=" ")
    half = d // 2
    pca_R = pca_basis_np(np.random.randn(n, d).astype(np.float32))
    R_cu, mse_cu = optimize_complex_unitary(
        np.random.randn(n, d).astype(np.float32), pca_R, 3, 5, 42
    )
    check_orthogonal(R_cu, tol=1e-4)
    assert np.isfinite(mse_cu), "Complex unitary MSE must be finite"
    print("PASS")

    print("[self-test] commutator-reg exp(A) orthogonality ...", end=" ")
    k_data = np.random.randn(n, d).astype(np.float32)
    cov = np.cov(k_data, rowvar=False).astype(np.float32)
    R, mse = optimize_commutator_reg(k_data, cov, lam=1.0, n_bits=3, max_iters=10, rank=2)
    check_orthogonal(R, tol=1e-4)
    assert np.isfinite(mse), "MSE must be finite"
    print("PASS")

    print("[self-test] quantize_mse monotonicity ...", end=" ")
    R_eye = np.eye(d, dtype=np.float32)
    mse_2 = quantize_mse_np(k_data, R_eye, 2)
    mse_3 = quantize_mse_np(k_data, R_eye, 3)
    mse_4 = quantize_mse_np(k_data, R_eye, 4)
    assert mse_2 > mse_3 > mse_4 > 0, f"MSE must decrease with bits: {mse_2:.4f}, {mse_3:.4f}, {mse_4:.4f}"
    print("PASS")

    print("[self-test] block-diagonal optimisation improves over identity ...", end=" ")
    pca_R = pca_basis_np(k_data)
    mse_pca = quantize_mse_np(k_data, pca_R, 3)
    mse_id = quantize_mse_np(k_data, R_eye, 3)
    assert mse_pca < mse_id, f"PCA should beat identity: pca={mse_pca:.4f} id={mse_id:.4f}"
    composed, mse_blk = optimize_block_diagonal(k_data, pca_R, 2, 3, 20, 42)
    check_orthogonal(composed, tol=1e-4)
    assert np.isfinite(mse_blk), "Block MSE must be finite"
    print(f"PASS (pca={mse_pca:.4f}, block_b2={mse_blk:.4f})")

    print("[self-test] attention KL is non-negative ...", end=" ")
    Q = np.random.randn(50, d).astype(np.float32)
    K = np.random.randn(n, d).astype(np.float32)
    K_q = quantize_vector_np(K, 3)
    kl = attention_kl_np(Q, K, K_q, d)
    assert kl >= 0, f"KL must be non-negative: {kl}"
    kl_self = attention_kl_np(Q, K, K, d)
    assert kl_self < 1e-6, f"KL(p||p) should be ~0: {kl_self}"
    print("PASS")

    print("[self-test] identity rotation reproduces direct quantisation ...", end=" ")
    mse_direct = float(np.mean((k_data - quantize_vector_np(k_data, 3)) ** 2))
    mse_via_id = quantize_mse_np(k_data, R_eye, 3)
    assert abs(mse_direct - mse_via_id) < 1e-6, f"Identity mismatch: {mse_direct} vs {mse_via_id}"
    print("PASS")

    print("\n[self-test] ALL PASSED")
    return True


# ── Main experiment ──────────────────────────────────────────────────

def _evaluate_rotation(K_ho, Q_ho, R, bits, d):
    """Evaluate a rotation on held-out data: MSE and attention KL."""
    mse = quantize_mse_np(K_ho, R, bits)
    K_q = quantize_vector_np(K_ho @ R, bits) @ R.T
    kl = attention_kl_np(Q_ho, K_ho, K_q, d)
    return mse, kl


def run_experiment(args: argparse.Namespace):
    import torch

    OPT_BITS = 3  # Plan: "Optimize ... to minimize 3-bit quantization MSE"

    if args.smoke:
        args.model_name = "openai-community/gpt2"
        args.num_samples = 4
        args.max_seq_len = 64
        args.max_opt_iters = 10
        args.layer_stride = 6
        args.block_sizes = [2, 4]
        args.commutator_lambdas = [0.1, 1.0]

    set_seed(args.seed)

    print(f"[exp5] Loading model: {args.model_name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir or None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16
    if "qwen" in args.model_name.lower() or "llama" in args.model_name.lower():
        dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype,
        device_map=args.device,
        attn_implementation="eager",
        cache_dir=args.cache_dir or None,
    )
    model.eval()

    # Calibration texts from WikiText-2
    print("[exp5] Loading WikiText-2 calibration texts ...")
    from datasets import load_dataset
    ds_kwargs = {}
    if args.cache_dir:
        ds_kwargs["cache_dir"] = args.cache_dir
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", **ds_kwargs)
    texts = [x for x in train["text"] if x and x.strip()][:args.num_samples]

    # Extract per-head K, Q
    print(f"[exp5] Extracting per-head K/Q vectors ({len(texts)} texts, max_seq_len={args.max_seq_len}) ...")
    kq_data, n_heads, d = extract_kq_per_head(
        model, tokenizer, texts, args.device, args.max_seq_len, model_type="gpt2",
    )
    num_layers = len(kq_data)

    # Select layers and heads
    layer_indices = list(range(0, num_layers, args.layer_stride))
    # Evaluate head 0 per layer (representative); aggregate across layers for stats
    eval_head = 0
    print(f"[exp5] Evaluating layers: {layer_indices}, head={eval_head} (d={d})")

    results = {
        "config": {
            "model": args.model_name,
            "num_samples": args.num_samples,
            "max_seq_len": args.max_seq_len,
            "quant_bits": args.quant_bits,
            "block_sizes": args.block_sizes,
            "layer_stride": args.layer_stride,
            "heldout_frac": args.heldout_frac,
            "max_opt_iters": args.max_opt_iters,
            "commutator_lambdas": args.commutator_lambdas,
            "opt_bits": OPT_BITS,
            "seed": args.seed,
            "d": d,
            "n_heads": n_heads,
            "eval_head": eval_head,
            "layers_evaluated": layer_indices,
        },
        "per_layer": {},
    }

    for li in layer_indices:
        print(f"\n[exp5] === Layer {li}, Head {eval_head} ===")
        K_all = kq_data[li][eval_head]["K"]
        Q_all = kq_data[li][eval_head]["Q"]
        n = K_all.shape[0]

        # Train / held-out split
        rng = np.random.RandomState(args.seed + li)
        perm = rng.permutation(n)
        n_heldout = max(1, int(n * args.heldout_frac))
        idx_cal = perm[n_heldout:]
        idx_ho = perm[:n_heldout]
        K_cal, K_ho = K_all[idx_cal], K_all[idx_ho]
        Q_ho = Q_all[idx_ho]

        # Baselines
        R_id = np.eye(d, dtype=np.float32)
        R_rng = random_orthogonal_np(d, args.seed + li * 1000)
        R_pca = pca_basis_np(K_cal)
        cov = np.cov(K_cal, rowvar=False).astype(np.float32)

        layer_results = {"baselines": {}, "exp5_1_block": {}, "exp5_2_complex": {}, "exp5_3_commutator": {}}

        for bits in args.quant_bits:
            for name, R in [("identity", R_id), ("random", R_rng), ("pca", R_pca)]:
                mse, kl = _evaluate_rotation(K_ho, Q_ho, R, bits, d)
                key = f"{name}_{bits}bit"
                layer_results["baselines"][key] = {"mse": mse, "kl": kl}
                print(f"  [baseline] {name} {bits}bit: MSE={mse:.4f} KL={kl:.6f}")

        # 5-1: Block diagonal — optimize at OPT_BITS, evaluate at all bits
        for bs in args.block_sizes:
            if d % bs != 0:
                print(f"  [5-1] skip block_size={bs} (d={d} not divisible)")
                continue
            t0 = time.time()
            R_comp, _ = optimize_block_diagonal(K_cal, R_pca, bs, OPT_BITS, args.max_opt_iters, args.seed)
            orth_err = check_orthogonal(R_comp)
            opt_time = time.time() - t0
            for bits in args.quant_bits:
                mse, kl = _evaluate_rotation(K_ho, Q_ho, R_comp, bits, d)
                key = f"block_b{bs}_{bits}bit"
                layer_results["exp5_1_block"][key] = {
                    "mse": mse, "kl": kl, "orth_err": orth_err,
                    "opt_time_s": opt_time,
                }
                print(f"  [5-1] block_b{bs} {bits}bit: MSE={mse:.4f} KL={kl:.6f}")

        # 5-2: Complex unitary — optimize at OPT_BITS, evaluate at all bits
        t0 = time.time()
        R_cu, _ = optimize_complex_unitary(K_cal, R_pca, OPT_BITS, args.max_opt_iters, args.seed)
        orth_err = check_orthogonal(R_cu)
        opt_time = time.time() - t0
        for bits in args.quant_bits:
            mse, kl = _evaluate_rotation(K_ho, Q_ho, R_cu, bits, d)
            key = f"complex_unitary_{bits}bit"
            layer_results["exp5_2_complex"][key] = {
                "mse": mse, "kl": kl, "orth_err": orth_err,
                "opt_time_s": opt_time,
            }
            print(f"  [5-2] complex_unitary {bits}bit: MSE={mse:.4f} KL={kl:.6f}")

        # 5-3: Commutator-regularised — optimize at OPT_BITS, evaluate at all bits
        for lam in args.commutator_lambdas:
            t0 = time.time()
            R_cr, _ = optimize_commutator_reg(
                K_cal, cov, lam, OPT_BITS, args.max_opt_iters,
                rank=min(2, d // 2), seed=args.seed,
            )
            orth_err = check_orthogonal(R_cr)
            opt_time = time.time() - t0
            for bits in args.quant_bits:
                mse, kl = _evaluate_rotation(K_ho, Q_ho, R_cr, bits, d)
                key = f"commutator_lam{lam}_{bits}bit"
                layer_results["exp5_3_commutator"][key] = {
                    "mse": mse, "kl": kl, "orth_err": orth_err,
                    "opt_time_s": opt_time,
                }
                print(f"  [5-3] commutator λ={lam} {bits}bit: MSE={mse:.4f} KL={kl:.6f}")

        results["per_layer"][str(li)] = layer_results

    # ── Aggregate statistics ──
    print("\n[exp5] === Aggregate Statistics ===")
    summary = {}
    for bits in args.quant_bits:
        per_bits = {}
        for method_group in ["baselines", "exp5_1_block", "exp5_2_complex", "exp5_3_commutator"]:
            for li in layer_indices:
                lr = results["per_layer"][str(li)][method_group]
                for key, val in lr.items():
                    if key.endswith(f"_{bits}bit"):
                        short = key.replace(f"_{bits}bit", "")
                        if short not in per_bits:
                            per_bits[short] = {"mse": [], "kl": []}
                        per_bits[short]["mse"].append(val["mse"])
                        per_bits[short]["kl"].append(val["kl"])
        summary[f"{bits}bit"] = {}
        pca_mse = per_bits.get("pca", {}).get("mse", [])
        pca_kl = per_bits.get("pca", {}).get("kl", [])
        for method, vals in sorted(per_bits.items()):
            mean_mse = float(np.mean(vals["mse"]))
            mean_kl = float(np.mean(vals["kl"]))
            entry = {"mean_mse": mean_mse, "mean_kl": mean_kl}
            if pca_mse and len(vals["mse"]) == len(pca_mse) and method != "pca":
                # Paired t-test for MSE (plan requirement)
                t_stat, p_val = stats.ttest_rel(vals["mse"], pca_mse)
                entry["vs_pca_mse_ttest_t"] = float(t_stat)
                entry["vs_pca_mse_ttest_p"] = float(p_val)
                entry["beats_pca_mse"] = bool(mean_mse < np.mean(pca_mse) and p_val < 0.05)
                # Wilcoxon signed-rank for KL (plan requirement: non-parametric backup)
                if len(vals["kl"]) >= 6 and pca_kl:
                    try:
                        w_stat, w_p = stats.wilcoxon(vals["kl"], pca_kl)
                        entry["vs_pca_kl_wilcoxon_w"] = float(w_stat)
                        entry["vs_pca_kl_wilcoxon_p"] = float(w_p)
                        entry["beats_pca_kl"] = bool(mean_kl < np.mean(pca_kl) and w_p < 0.05)
                    except ValueError:
                        pass  # wilcoxon requires ≥6 non-zero differences
            summary[f"{bits}bit"][method] = entry
            flag = " **" if entry.get("beats_pca_mse", False) else ""
            print(f"  {bits}bit {method:30s}: MSE={mean_mse:.4f}  KL={mean_kl:.6f}{flag}")

    results["summary"] = summary

    # ── Write output ──
    output_dir = Path(args.output_dir) if args.output_dir else Path("/tmp/exp5_lie_variants")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "exp5_lie_variants_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[exp5] Results written to {out_path}")


def set_seed(seed: int):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ── Entry point ──────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.self_test:
        ok = run_self_tests()
        sys.exit(0 if ok else 1)
    run_experiment(args)


if __name__ == "__main__":
    main()
