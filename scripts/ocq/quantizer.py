"""
OCQ — Ontology-Categorical Quantization for LLM K cache.

(Previously drafted as ``oc-FOKVQ``; renamed 2026-04-09 to avoid the
patent-excluded ``FOKVQ`` token. Internal API symbols were also renamed
``oc_fokvq_* -> ocq_*`` at that time.)

Core insight (2026-04-09): In agent inference with an ontology-resolved
context (e.g., "marketer" vs "UI analyst" already decided from session
history), the K-space projections onto the ontology axes carry a
*categorical* decision, not a continuous magnitude. Post-decision, the
residual variance on those axes is noise. Therefore 1-bit quantization
on ontology axes is *noise removal*, not lossy compression.

This module implements six variants from the cross-product:

  Ontology-axis quantization modes:
    1a: sign-only          — store sign(coeff), reconstruct with ±|mean|
    1b: 2-level mean split — k-means (k=2) per axis, store cluster bit
    1c: combined argmax    — one facet index per token (log2(r_ont) bits
                              total for all ontology dims), zero elsewhere

  Residual basis computation modes:
    2a: Σ_res eigh         — (I - B_ont B_ontᵀ) Σ_K (I - B_ont B_ontᵀ),
                              top (d - r_ont) eigenvectors
    2b: full Σ_K minus ont — full Σ_K eigendecomp, Gram-Schmidt project
                              out B_ont directions, keep top (d - r_ont)

Residual coefficients use standard per-token symmetric uniform
quantization in the rotated space.

API:
    build_ocq_basis(sigma_k, B_ont, res_mode) -> B_full
    ocq_quantize(keys, B_full_per_head, r_ont, ont_mode, res_bits)

Shapes (per-head, applied in loops for per-(layer, head)):
    sigma_k:  (d, d) float
    B_ont:    (d, r_ont) float, orthonormal
    B_full:   (d, d) float, orthonormal, first r_ont cols = B_ont
    keys:     (..., d) float
    keys_q:   (..., d) float (reconstruction in original space)

All computation in float32 or float64 for numerical stability; callers
are responsible for casting back to the K cache dtype.

IMPORTANT (2026-04-09 bug fix): the ontology basis B_ont is constructed
from **pre-RoPE** k_proj output (see scripts/ocq/build_qwen_metatool_b_ont.py).
Therefore ocq_quantize must be applied to **pre-RoPE** K tensors — i.e.
via a forward hook on `layer.self_attn.k_proj` that rewrites the tensor
before it is consumed by apply_rotary_pos_emb. Applying OCQ to HF's
`past_key_values` (which stores post-RoPE K) is a space mismatch and
produces incorrect results.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import torch

OntMode = Literal["1a", "1b", "1c"]
ResMode = Literal["2a", "2b"]


# ---------------------------------------------------------------------
# Basis construction
# ---------------------------------------------------------------------

def _gram_schmidt_project_out(V: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Project columns of V onto the orthogonal complement of U's column
    space, then re-orthonormalize.

    Args:
        V: (d, k) — candidate vectors (not necessarily orthonormal).
        U: (d, m) — orthonormal basis whose span is to be removed.

    Returns:
        V_res: (d, k') orthonormal columns spanning V ∩ U^⊥, with
               k' ≤ k (rank may drop if V was partially in span(U)).
    """
    P = V - U @ (U.T @ V)
    Q, R = np.linalg.qr(P, mode="reduced")
    # Drop numerically-zero columns (rank-deficient cases).
    rank = int(np.sum(np.abs(np.diag(R)) > 1e-8))
    return Q[:, :rank]


def build_residual_basis_2a(
    sigma_k: np.ndarray,
    B_ont: np.ndarray,
    target_dim: int | None = None,
) -> np.ndarray:
    """Mode 2a: orthogonal-complement eigendecomposition.

    Computes Σ_res = (I - B_ont B_ontᵀ) Σ_K (I - B_ont B_ontᵀ), then
    returns the top target_dim eigenvectors of Σ_res.
    """
    d = sigma_k.shape[0]
    if target_dim is None:
        target_dim = d - B_ont.shape[1]
    if target_dim <= 0:
        return np.zeros((d, 0), dtype=sigma_k.dtype)
    P_perp = np.eye(d, dtype=sigma_k.dtype) - B_ont @ B_ont.T
    sigma_res = P_perp @ sigma_k @ P_perp
    sigma_res = 0.5 * (sigma_res + sigma_res.T)  # symmetrize
    eigvals, eigvecs = np.linalg.eigh(sigma_res)
    order = np.argsort(-eigvals)
    eigvecs = eigvecs[:, order]
    B_res = eigvecs[:, :target_dim]
    # Re-orthogonalize against B_ont to kill numerical drift.
    return _gram_schmidt_project_out(B_res, B_ont)


def build_residual_basis_2b(
    sigma_k: np.ndarray,
    B_ont: np.ndarray,
    target_dim: int | None = None,
) -> np.ndarray:
    """Mode 2b: full Σ_K PCA with B_ont projection.

    Computes full eigendecomposition of Σ_K, then Gram-Schmidt-projects
    out B_ont from each eigenvector, and keeps the top target_dim of
    the projected set (ordered by original eigenvalue).
    """
    d = sigma_k.shape[0]
    if target_dim is None:
        target_dim = d - B_ont.shape[1]
    if target_dim <= 0:
        return np.zeros((d, 0), dtype=sigma_k.dtype)
    sigma_sym = 0.5 * (sigma_k + sigma_k.T)
    eigvals, eigvecs = np.linalg.eigh(sigma_sym)
    order = np.argsort(-eigvals)
    eigvecs = eigvecs[:, order]  # (d, d), top PCA dir first
    # Project out B_ont from each, preserving eigenvalue order.
    B_res_raw = eigvecs - B_ont @ (B_ont.T @ eigvecs)
    # Orthonormalize while preserving order: use modified Gram-Schmidt
    # instead of QR (QR re-orders by numerical pivoting).
    B_res = _modified_gram_schmidt(B_res_raw, target_dim)
    return B_res


def _modified_gram_schmidt(M: np.ndarray, target_dim: int) -> np.ndarray:
    """Modified Gram-Schmidt keeping column order; drop zero cols.

    Returns a (d, k) matrix with k ≤ target_dim orthonormal columns in
    the order they were provided.
    """
    d, n = M.shape
    out = np.zeros((d, min(n, target_dim)), dtype=M.dtype)
    count = 0
    for j in range(n):
        if count >= target_dim:
            break
        v = M[:, j].copy()
        for i in range(count):
            v -= np.dot(out[:, i], v) * out[:, i]
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            out[:, count] = v / norm
            count += 1
    return out[:, :count]


def build_ocq_basis(
    sigma_k: np.ndarray,
    B_ont: np.ndarray,
    res_mode: ResMode,
) -> np.ndarray:
    """Return full rotation basis B_full = [B_ont | B_res] of shape
    (d, d) where B_res is orthogonal to B_ont and spans the residual
    space according to the selected mode.

    If dimensions don't reach d (rank-deficient residual), B_full has
    fewer columns than d; callers must handle the partial-rank case.
    """
    d = sigma_k.shape[0]
    r_ont = B_ont.shape[1]
    target_res_dim = d - r_ont
    if res_mode == "2a":
        B_res = build_residual_basis_2a(sigma_k, B_ont, target_res_dim)
    elif res_mode == "2b":
        B_res = build_residual_basis_2b(sigma_k, B_ont, target_res_dim)
    else:
        raise ValueError(f"Unknown res_mode: {res_mode}")
    B_full = np.concatenate([B_ont, B_res], axis=1)
    return B_full


# ---------------------------------------------------------------------
# Ontology-axis categorical quantization modes
# ---------------------------------------------------------------------

def _quant_1a_sign(
    coeffs_ont: torch.Tensor,  # (..., r_ont)
) -> torch.Tensor:
    """Mode 1a: store sign per axis, reconstruct with per-axis mean magnitude.

    Bits per token per axis: 1.
    """
    signs = torch.sign(coeffs_ont)
    # Mean |·| over all non-last dims; broadcast back.
    reduce_dims = tuple(range(coeffs_ont.ndim - 1))
    mean_mag = coeffs_ont.abs().mean(dim=reduce_dims, keepdim=True)
    return signs * mean_mag


def _quant_1b_mean_split(
    coeffs_ont: torch.Tensor,  # (..., r_ont)
) -> torch.Tensor:
    """Mode 1b: per-axis 2-level quantization by mean split.

    Each axis is clustered into "above the axis mean" vs "below the
    axis mean"; each bucket is represented by its own centroid
    (mean of values in that bucket). Bits per token per axis: 1.
    """
    reduce_dims = tuple(range(coeffs_ont.ndim - 1))
    # Per-axis split point: mean across all tokens.
    split = coeffs_ont.mean(dim=reduce_dims, keepdim=True)
    mask_hi = coeffs_ont >= split
    mask_lo = ~mask_hi

    # Per-axis centroids: need mean of the values in each bucket.
    # Using masked mean: sum(val * mask) / sum(mask).
    hi_sum = (coeffs_ont * mask_hi.float()).sum(dim=reduce_dims, keepdim=True)
    hi_cnt = mask_hi.float().sum(dim=reduce_dims, keepdim=True).clamp(min=1.0)
    hi_mean = hi_sum / hi_cnt

    lo_sum = (coeffs_ont * mask_lo.float()).sum(dim=reduce_dims, keepdim=True)
    lo_cnt = mask_lo.float().sum(dim=reduce_dims, keepdim=True).clamp(min=1.0)
    lo_mean = lo_sum / lo_cnt

    return torch.where(mask_hi, hi_mean, lo_mean).expand_as(coeffs_ont).contiguous()


def _quant_1c_combined_argmax(
    coeffs_ont: torch.Tensor,  # (..., r_ont)
) -> torch.Tensor:
    """Mode 1c: per-token single-facet selection.

    For each token, identify the ontology axis with the largest |coeff|.
    Store only that coefficient (at full precision conceptually — in
    practice we reuse the original value), zero out the others.

    Bits per token for the whole ontology block: ceil(log2(r_ont)) + ~1
    (argmax index + sign) — much more aggressive than 1a/1b.

    Reconstruction: at the argmax axis, use the original coefficient;
    at all other axes, zero.
    """
    r_ont = coeffs_ont.shape[-1]
    # argmax by magnitude
    argmax = coeffs_ont.abs().argmax(dim=-1, keepdim=True)  # (..., 1)
    # Gather the winning coefficient value
    winning = torch.gather(coeffs_ont, dim=-1, index=argmax)  # (..., 1)
    # Build sparse output
    out = torch.zeros_like(coeffs_ont)
    out.scatter_(dim=-1, index=argmax, src=winning)
    return out


_ONT_MODES = {
    "1a": _quant_1a_sign,
    "1b": _quant_1b_mean_split,
    "1c": _quant_1c_combined_argmax,
}


# ---------------------------------------------------------------------
# Residual axis quantization (variance-proportional uniform)
# ---------------------------------------------------------------------

def _symmetric_quant_per_token(
    coeffs: torch.Tensor,  # (..., k)
    bits: int,
) -> torch.Tensor:
    """Per-token symmetric uniform quantization on the last dim."""
    if bits <= 0:
        return torch.zeros_like(coeffs)
    levels = 2 ** bits
    qmin = -levels // 2
    qmax = (levels + 1) // 2 - 1
    scale = coeffs.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / max(1, -qmin)
    quant = (coeffs / scale).round().clamp(qmin, qmax)
    return quant * scale


# ---------------------------------------------------------------------
# Main quantize function
# ---------------------------------------------------------------------

def ocq_quantize_head(
    keys: torch.Tensor,       # (..., d)
    B_full: torch.Tensor,     # (d, d)
    r_ont: int,
    ont_mode: OntMode,
    res_bits: int,
) -> torch.Tensor:
    """Apply OCQ to K vectors for one head.

    Steps:
    1. Rotate into [B_ont | B_res] basis.
    2. Categorically quantize ontology coefficients.
    3. Variance-proportionally quantize residual coefficients at res_bits.
    4. Rotate back.

    Args:
        keys: (..., d) K vectors (any leading dims, last is feature).
        B_full: (d, d) orthonormal rotation, first r_ont cols = B_ont.
        r_ont: ontology subspace dimension.
        ont_mode: '1a' / '1b' / '1c'.
        res_bits: residual uniform bits (e.g. 2 or 3).

    Returns:
        Reconstructed keys in the original basis, same shape as input.
    """
    if B_full.shape[0] != keys.shape[-1]:
        raise ValueError(
            f"Basis dim {B_full.shape[0]} != keys feature dim {keys.shape[-1]}"
        )
    if B_full.shape[1] != B_full.shape[0]:
        # Partial-rank residual — pad with zero columns so the rotation is
        # square. Padding columns contribute zero to reconstruction.
        d = B_full.shape[0]
        pad = torch.zeros(
            d, d - B_full.shape[1],
            device=B_full.device, dtype=B_full.dtype,
        )
        B_full = torch.cat([B_full, pad], dim=1)

    # Rotate: coeffs = keys @ B_full (trailing-dim matmul)
    coeffs = torch.matmul(keys, B_full)  # (..., d)

    coeffs_ont = coeffs[..., :r_ont]
    coeffs_res = coeffs[..., r_ont:]

    # Categorical quant on ontology axes
    if ont_mode not in _ONT_MODES:
        raise ValueError(f"Unknown ont_mode: {ont_mode}")
    coeffs_ont_q = _ONT_MODES[ont_mode](coeffs_ont)

    # Variance-prop uniform on residual
    coeffs_res_q = _symmetric_quant_per_token(coeffs_res, res_bits)

    coeffs_q = torch.cat([coeffs_ont_q, coeffs_res_q], dim=-1)
    # Inverse rotation: keys_recon = coeffs_q @ B_full.T
    return torch.matmul(coeffs_q, B_full.transpose(-1, -2))


def ocq_quantize(
    keys: torch.Tensor,         # (B, H, S, d)
    B_full_per_head: torch.Tensor,  # (H, d, d)
    r_ont: int,
    ont_mode: OntMode,
    res_bits: int,
) -> torch.Tensor:
    """Apply OCQ over all heads of a K cache tensor.

    Assumes per-head independent basis.
    """
    if keys.ndim != 4:
        raise ValueError(f"Expected (B, H, S, d) keys, got {keys.shape}")
    H = keys.shape[1]
    if B_full_per_head.shape[0] != H:
        raise ValueError(
            f"Basis heads {B_full_per_head.shape[0]} != keys heads {H}"
        )
    # Permute to (H, B, S, d), apply per head, permute back.
    k = keys.permute(1, 0, 2, 3).contiguous()  # (H, B, S, d)
    out = torch.zeros_like(k)
    for h in range(H):
        out[h] = ocq_quantize_head(
            k[h], B_full_per_head[h], r_ont, ont_mode, res_bits,
        )
    return out.permute(1, 0, 2, 3).contiguous()


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def _self_test() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    d = 64
    r_ont = 8
    H = 4
    S = 32
    B = 2

    # Synthetic Σ_K: rank-12 anisotropic Gaussian + noise
    U = np.random.randn(d, 16)
    U, _ = np.linalg.qr(U)
    lam = np.diag([5.0] * 4 + [2.0] * 4 + [1.0] * 4 + [0.3] * (d - 12))
    full_rot = np.random.randn(d, d)
    full_rot, _ = np.linalg.qr(full_rot)
    sigma_k = (full_rot @ lam @ full_rot.T).astype(np.float64)
    sigma_k = 0.5 * (sigma_k + sigma_k.T)

    # Synthetic B_ont: orthonormal r_ont columns
    B_ont_np = np.random.randn(d, r_ont)
    B_ont_np, _ = np.linalg.qr(B_ont_np)

    # Test both residual modes
    for res_mode in ["2a", "2b"]:
        B_full_np = build_ocq_basis(sigma_k, B_ont_np, res_mode=res_mode)
        assert B_full_np.shape[0] == d, f"wrong d: {B_full_np.shape}"
        k_cols = B_full_np.shape[1]
        # Orthogonality check
        BTB = B_full_np.T @ B_full_np
        ortho = float(np.max(np.abs(BTB - np.eye(k_cols))))
        assert ortho < 1e-6, f"{res_mode} orthogonality error {ortho}"
        # B_ont preserved in first r_ont cols
        diff = float(np.max(np.abs(B_full_np[:, :r_ont] - B_ont_np)))
        assert diff < 1e-8, f"{res_mode} B_ont not preserved: {diff}"

        # Quantize keys
        B_full_t = torch.from_numpy(B_full_np.astype(np.float32))
        B_per_head = B_full_t.unsqueeze(0).repeat(H, 1, 1)
        keys = torch.randn(B, H, S, d, dtype=torch.float32)
        for ont_mode in ["1a", "1b", "1c"]:
            for res_bits in [2, 3, 4]:
                k_q = ocq_quantize(
                    keys, B_per_head, r_ont, ont_mode, res_bits,
                )
                assert k_q.shape == keys.shape
                assert torch.isfinite(k_q).all()
                mse = float((keys - k_q).pow(2).mean())
                print(
                    f"  {res_mode} {ont_mode} res_bits={res_bits}: "
                    f"MSE={mse:.4e}"
                )
    print("ocq self-test passed.")


if __name__ == "__main__":
    _self_test()
