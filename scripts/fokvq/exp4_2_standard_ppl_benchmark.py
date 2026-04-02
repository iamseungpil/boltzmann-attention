"""
FOKVQ Experiment 4-2 / 10-1: Standard WikiText-2 PPL Benchmark
===============================================================

Implements the reviewer-grade sliding-window protocol described in
FOKVQ_ADDITIONAL_EXPERIMENT_PLAN_v9.docx.

Protocol:
1. Load WikiText-2 test set via `datasets`
2. Evaluate perplexity with a sliding-window causal LM protocol
3. For quantized runs:
   - build prefix KV cache
   - quantize cached K tensors
   - evaluate the next stride using the quantized cache
4. Compare FP16, uniform bit quantization, and PCA-based FOKVQ

Notes:
- This runner quantizes K caches only. Uniform and FOKVQ share the same
  evaluation path, so comparisons remain internally fair.
- FOKVQ uses a per-layer per-head PCA basis from calibration text and a
  target-bit-preserving high/low allocation schedule:
    2-bit -> 5/1 bits
    3-bit -> 6/2 bits
    4-bit -> 7/3 bits
  over the top 25% principal axes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--self-test", action="store_true")
    pre_args, _ = bootstrap.parse_known_args()

    parser = argparse.ArgumentParser(parents=[bootstrap])
    required = not pre_args.self_test
    parser.add_argument("--model-name", type=str, required=required, default="self-test")
    parser.add_argument("--model-key", type=str, required=required, default="self-test")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--context-len", type=int, required=required, default=32)
    parser.add_argument("--stride", type=int, required=required, default=16)
    parser.add_argument("--calibration-samples", type=int, default=16)
    parser.add_argument("--calibration-max-len", type=int, default=512)
    parser.add_argument("--max-eval-tokens", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=["fp16", "uniform", "fokvq"])
    parser.add_argument("--bits", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--fokvq-topk-frac", type=float, default=0.25)
    parser.add_argument("--fokvq-adaptive-energy-frac", type=float, default=0.9)
    parser.add_argument("--fokvq-clip-quantile", type=float, default=0.995)
    parser.add_argument("--output-dir", type=str, required=required, default="/tmp/exp4_2_self_test")
    parser.add_argument("--cache-dir", type=str, default="")
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


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


def get_tokenizer_and_model(
    model_name: str,
    dtype: torch.dtype,
    device: str,
    cache_dir: Optional[str],
    attn_implementation: str,
):
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation=attn_implementation,
        **kwargs,
    )
    model.eval()
    return tokenizer, model


def load_wikitext_texts(cache_dir: Optional[str]) -> Tuple[List[str], List[str]]:
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", **kwargs)
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", **kwargs)
    train_texts = [x for x in train["text"] if x and x.strip()]
    test_texts = [x for x in test["text"] if x and x.strip()]
    return train_texts, test_texts


def collect_calibration_texts(texts: Sequence[str], n_samples: int) -> List[str]:
    return list(texts[:n_samples])


def encode_joined_text(tokenizer, texts: Sequence[str], device: str) -> torch.Tensor:
    joined = "\n\n".join(texts)
    encoded = tokenizer(joined, return_tensors="pt")
    return encoded["input_ids"].to(device)


def to_legacy_cache(past_key_values):
    if past_key_values is None:
        return None, None
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache(), type(past_key_values)
    return past_key_values, None


def from_legacy_cache(legacy_cache, cache_cls):
    if legacy_cache is None:
        return None
    if cache_cls is DynamicCache or isinstance(legacy_cache, tuple):
        return DynamicCache(legacy_cache)
    if cache_cls is not None and hasattr(cache_cls, "from_legacy_cache"):
        return cache_cls.from_legacy_cache(legacy_cache)
    return legacy_cache


def symmetric_quantize_last_dim(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    if bits <= 0:
        raise ValueError(f"bits must be positive, got {bits}")
    if bits == 1:
        abs_max = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        signs = torch.where(tensor >= 0, torch.ones_like(tensor), -torch.ones_like(tensor))
        return signs * abs_max
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    abs_max = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = abs_max / qmax
    quant = (tensor / scale).round().clamp(qmin, qmax)
    return quant * scale


def asymmetric_quantize_seq_dim(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    if bits <= 0:
        raise ValueError(f"bits must be positive, got {bits}")
    levels = (2 ** bits) - 1
    if levels <= 0:
        return tensor
    t_min = tensor.amin(dim=-2, keepdim=True)
    t_max = tensor.amax(dim=-2, keepdim=True)
    scale = ((t_max - t_min) / levels).clamp(min=1e-8)
    quant = ((tensor - t_min) / scale).round().clamp(0, levels)
    return quant * scale + t_min


def pca_basis_np(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt.T.astype(np.float32)


def effective_average_bits(high_bits: int, low_bits: int, high_frac: float) -> float:
    return high_frac * float(high_bits) + (1.0 - high_frac) * float(low_bits)


def bit_schedule(target_bits: int, high_frac: float) -> Tuple[int, int]:
    if target_bits <= 1:
        return target_bits, target_bits
    if high_frac <= 0.0 or high_frac >= 1.0:
        return target_bits, target_bits

    low_bits = max(1, target_bits - 1)
    high_bits_real = (target_bits - (1.0 - high_frac) * low_bits) / high_frac
    high_bits = int(round(high_bits_real))
    if high_bits < low_bits:
        high_bits = low_bits

    avg_bits = effective_average_bits(high_bits, low_bits, high_frac)
    if abs(avg_bits - float(target_bits)) > 0.05:
        raise ValueError(
            "Requested high_frac does not admit an integer fair mixed-precision schedule: "
            f"target_bits={target_bits}, high_frac={high_frac:.6f}, "
            f"candidate=({high_bits},{low_bits}), effective_avg_bits={avg_bits:.6f}"
        )
    return high_bits, low_bits


def compute_random_orthogonal_basis(dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_matrix = torch.randn(dim, dim, generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(random_matrix, mode="reduced")
    return q.contiguous()


def fit_lloyd_max_codebook(
    samples: torch.Tensor,
    num_levels: int,
    n_iters: int = 30,
) -> torch.Tensor:
    values = samples.reshape(-1).float()
    quantiles = torch.linspace(0.0, 1.0, steps=num_levels + 2, dtype=torch.float32)[1:-1]
    codebook = torch.quantile(values, quantiles).unique(sorted=True)
    if codebook.numel() != num_levels:
        codebook = torch.linspace(values.min(), values.max(), steps=num_levels, dtype=torch.float32)
    for _ in range(n_iters):
        thresholds = 0.5 * (codebook[:-1] + codebook[1:])
        bucket_ids = torch.bucketize(values, thresholds)
        next_codebook = codebook.clone()
        for idx in range(num_levels):
            mask = bucket_ids == idx
            if mask.any():
                next_codebook[idx] = values[mask].mean()
        if torch.allclose(next_codebook, codebook, atol=1e-6, rtol=0.0):
            codebook = next_codebook
            break
        codebook = next_codebook
    return codebook.sort().values.contiguous()


def build_standard_normal_codebooks(
    bits_list: Sequence[int],
    seed: int,
    num_samples: int = 200_000,
) -> Dict[int, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17)
    samples = torch.randn(num_samples, generator=generator, dtype=torch.float32)
    return {
        bits: fit_lloyd_max_codebook(samples, 2 ** bits)
        for bits in bits_list
    }


def quantize_with_codebook(tensor: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    flat = tensor.reshape(-1, 1)
    levels = codebook.to(device=tensor.device, dtype=tensor.dtype).view(1, -1)
    nearest = (flat - levels).abs().argmin(dim=1)
    return levels.squeeze(0)[nearest].reshape_as(tensor)


def scalar_uniform_quantize(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    levels = (2 ** bits) - 1
    if levels <= 0:
        return tensor
    t_min = tensor.amin()
    t_max = tensor.amax()
    scale = ((t_max - t_min) / levels).clamp(min=1e-8)
    quant = ((tensor - t_min) / scale).round().clamp(0, levels)
    return quant * scale + t_min


def global_clip_abs(tensor: torch.Tensor, clip_quantile: float) -> torch.Tensor:
    if not 0.0 < clip_quantile < 1.0:
        raise ValueError(f"clip_quantile must be in (0, 1), got {clip_quantile}")
    clip_value = torch.quantile(tensor.abs().reshape(-1).float(), clip_quantile).clamp(min=1e-8)
    return tensor.clamp(min=-clip_value, max=clip_value)


def select_topk_from_energy(coeffs: torch.Tensor, energy_frac: float) -> int:
    if not 0.0 < energy_frac <= 1.0:
        raise ValueError(f"energy_frac must be in (0, 1], got {energy_frac}")
    dim = coeffs.shape[-1]
    if dim == 0:
        return 0
    energy = coeffs.float().pow(2).sum(dim=tuple(range(coeffs.ndim - 1)))
    total = float(energy.sum().item())
    if total <= 0.0:
        return dim
    cumulative = torch.cumsum(energy, dim=0) / total
    top_k = int(torch.searchsorted(cumulative, energy_frac).item()) + 1
    return max(1, min(dim, top_k))


def snap_topk_to_fair_schedule(dim: int, desired_top_k: int, target_bits: int) -> int:
    desired_top_k = max(1, min(dim, desired_top_k))
    candidates = sorted(range(1, dim + 1), key=lambda x: (abs(x - desired_top_k), x))
    for top_k in candidates:
        high_frac = float(top_k) / float(dim)
        try:
            bit_schedule(target_bits, high_frac)
            return top_k
        except ValueError:
            continue
    return dim


def quantize_nonuniform_with_basis(
    keys: torch.Tensor,
    basis: torch.Tensor,
    target_bits: int,
    topk_frac: float,
    adaptive_energy_frac: Optional[float] = None,
    clip_quantile: Optional[float] = None,
    mean: Optional[torch.Tensor] = None,
    seq_asymmetric: bool = False,
) -> torch.Tensor:
    centered_keys = keys
    if mean is not None:
        centered_keys = keys - mean.view(1, -1, 1, mean.shape[-1])
    coeffs = torch.matmul(centered_keys, basis)
    dim = coeffs.shape[-1]
    if dim == 0:
        return keys
    if clip_quantile is not None:
        coeffs = global_clip_abs(coeffs, clip_quantile)
    if adaptive_energy_frac is not None:
        desired_top_k = select_topk_from_energy(coeffs, adaptive_energy_frac)
    else:
        desired_top_k = max(1, min(dim, int(round(dim * topk_frac))))
    top_k = snap_topk_to_fair_schedule(dim, desired_top_k, target_bits)
    high_frac = float(top_k) / float(dim)
    high_bits, low_bits = bit_schedule(target_bits, high_frac)
    if top_k >= dim:
        coeffs_q = asymmetric_quantize_seq_dim(coeffs, target_bits) if seq_asymmetric else symmetric_quantize_last_dim(coeffs, target_bits)
        recon = torch.matmul(coeffs_q, basis.transpose(-1, -2))
        return recon if mean is None else recon + mean.view(1, -1, 1, mean.shape[-1])
    coeffs_hi = asymmetric_quantize_seq_dim(coeffs[..., :top_k], high_bits) if seq_asymmetric else symmetric_quantize_last_dim(coeffs[..., :top_k], high_bits)
    coeffs_lo = asymmetric_quantize_seq_dim(coeffs[..., top_k:], low_bits) if seq_asymmetric else symmetric_quantize_last_dim(coeffs[..., top_k:], low_bits)
    coeffs_q = torch.cat([coeffs_hi, coeffs_lo], dim=-1)
    recon = torch.matmul(coeffs_q, basis.transpose(-1, -2))
    return recon if mean is None else recon + mean.view(1, -1, 1, mean.shape[-1])


def quantize_variance_keys(keys: torch.Tensor, target_bits: int) -> torch.Tensor:
    token_importance = keys.pow(2).sum(dim=-1, keepdim=True)
    threshold = token_importance.median(dim=-2, keepdim=True).values
    mask = (token_importance >= threshold)
    high_frac = float(mask.float().mean().item())
    high_bits, low_bits = bit_schedule(target_bits, high_frac)
    hi = symmetric_quantize_last_dim(keys, high_bits)
    lo = symmetric_quantize_last_dim(keys, low_bits)
    mask = mask.to(dtype=keys.dtype)
    return hi * mask + lo * (1.0 - mask)


def quantize_turboquant_keys(
    keys: torch.Tensor,
    basis: torch.Tensor,
    codebook: torch.Tensor,
    clip_quantile: Optional[float] = None,
) -> torch.Tensor:
    coeffs = torch.matmul(keys.float(), basis)
    if clip_quantile is not None:
        coeffs = global_clip_abs(coeffs, clip_quantile)
    scale = coeffs.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    coeffs_norm = coeffs / scale
    coeffs_q = quantize_with_codebook(coeffs_norm, codebook)
    return torch.matmul(coeffs_q * scale, basis.transpose(-1, -2))


def run_self_tests(seed: int) -> None:
    set_seed(seed)

    assert bit_schedule(2, 0.25) == (5, 1), "2-bit @ 25% should recover the original fair schedule"
    assert bit_schedule(2, 0.5) == (3, 1), "2-bit @ 50% should use a fair 3/1 schedule"
    assert bit_schedule(3, 0.5) == (4, 2), "3-bit @ 50% should use a fair 4/2 schedule"
    assert bit_schedule(4, 0.5) == (5, 3), "4-bit @ 50% should use a fair 5/3 schedule"

    basis = compute_random_orthogonal_basis(dim=16, seed=seed)
    orth_error = torch.max((basis.transpose(0, 1) @ basis - torch.eye(16)).abs()).item()
    assert orth_error < 1e-4, f"Orthogonality drift too large: {orth_error}"

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    samples = torch.randn(50_000, generator=generator, dtype=torch.float32)
    codebook = fit_lloyd_max_codebook(samples, num_levels=4)
    assert codebook.numel() == 4, f"Unexpected codebook size: {codebook.numel()}"
    assert torch.all(codebook[1:] >= codebook[:-1]), "Codebook must be monotonic"

    heldout = torch.randn(20_000, generator=generator, dtype=torch.float32)
    lloyd_q = quantize_with_codebook(heldout, codebook)
    uniform_q = scalar_uniform_quantize(heldout, bits=2)
    lloyd_mse = torch.mean((heldout - lloyd_q) ** 2).item()
    uniform_mse = torch.mean((heldout - uniform_q) ** 2).item()
    assert lloyd_mse <= uniform_mse * 1.02, (
        f"Lloyd-Max should not regress badly against scalar uniform: "
        f"lloyd={lloyd_mse:.6f}, uniform={uniform_mse:.6f}"
    )

    codebooks = build_standard_normal_codebooks(bits_list=[2, 4], seed=seed)
    random_basis = compute_random_orthogonal_basis(dim=8, seed=seed + 99)
    keys = torch.randn(2, 3, 11, 8, generator=generator, dtype=torch.float32)
    turbo_2 = quantize_turboquant_keys(keys, random_basis, codebooks[2])
    turbo_4 = quantize_turboquant_keys(keys, random_basis, codebooks[4])
    assert turbo_2.shape == keys.shape == turbo_4.shape, "Shape must be preserved"
    assert turbo_2.dtype == keys.dtype == turbo_4.dtype, "Dtype must be preserved"
    assert torch.isfinite(turbo_2).all() and torch.isfinite(turbo_4).all(), "Quantized tensors must be finite"
    mse_2 = torch.mean((keys - turbo_2) ** 2).item()
    mse_4 = torch.mean((keys - turbo_4) ** 2).item()
    assert mse_4 < mse_2, f"Higher-bit turboquant should reduce error: 4-bit={mse_4:.6f}, 2-bit={mse_2:.6f}"

    adaptive_q = quantize_nonuniform_with_basis(
        keys,
        random_basis,
        target_bits=2,
        topk_frac=0.25,
        adaptive_energy_frac=0.9,
    )
    clipped_q = quantize_nonuniform_with_basis(
        keys,
        random_basis,
        target_bits=2,
        topk_frac=0.5,
        clip_quantile=0.95,
    )
    pca_lloyd_q = quantize_turboquant_keys(keys, random_basis, codebooks[2], clip_quantile=0.95)
    assert torch.isfinite(adaptive_q).all() and torch.isfinite(clipped_q).all() and torch.isfinite(pca_lloyd_q).all()
    assert adaptive_q.shape == keys.shape == clipped_q.shape == pca_lloyd_q.shape

    shifted_keys = torch.randn(2, 3, 23, 8, generator=generator, dtype=torch.float32) + 3.0
    shifted_mean = shifted_keys.mean(dim=(0, 2))
    uncentered_shifted = quantize_nonuniform_with_basis(
        shifted_keys,
        random_basis,
        target_bits=2,
        topk_frac=0.5,
    )
    centered_shifted = quantize_nonuniform_with_basis(
        shifted_keys,
        random_basis,
        target_bits=2,
        topk_frac=0.5,
        mean=shifted_mean,
    )
    uncentered_shifted_mse = torch.mean((shifted_keys - uncentered_shifted) ** 2).item()
    centered_shifted_mse = torch.mean((shifted_keys - centered_shifted) ** 2).item()
    assert centered_shifted_mse < uncentered_shifted_mse, (
        f"Centering should help on shifted tensors: centered={centered_shifted_mse:.6f}, "
        f"uncentered={uncentered_shifted_mse:.6f}"
    )

    skewed_keys = torch.exp(torch.randn(2, 3, 19, 8, generator=generator, dtype=torch.float32) * 0.6)
    symmetric_skewed = quantize_nonuniform_with_basis(
        skewed_keys,
        torch.eye(8, dtype=torch.float32),
        target_bits=2,
        topk_frac=1.0,
        mean=None,
        seq_asymmetric=False,
    )
    asymmetric_skewed = quantize_nonuniform_with_basis(
        skewed_keys,
        torch.eye(8, dtype=torch.float32),
        target_bits=2,
        topk_frac=1.0,
        mean=None,
        seq_asymmetric=True,
    )
    symmetric_skewed_mse = torch.mean((skewed_keys - symmetric_skewed) ** 2).item()
    asymmetric_skewed_mse = torch.mean((skewed_keys - asymmetric_skewed) ** 2).item()
    assert asymmetric_skewed_mse < symmetric_skewed_mse, (
        f"Sequence-asymmetric quantization should help on skewed tensors: "
        f"asym={asymmetric_skewed_mse:.6f}, sym={symmetric_skewed_mse:.6f}"
    )

    payload = {
        "self_test": "passed",
        "orthogonality_max_abs_error": orth_error,
        "lloyd_mse": lloyd_mse,
        "uniform_mse": uniform_mse,
        "centered_shifted_mse": {
            "uncentered": uncentered_shifted_mse,
            "centered": centered_shifted_mse,
        },
        "skewed_mse": {
            "symmetric": symmetric_skewed_mse,
            "asymmetric": asymmetric_skewed_mse,
        },
        "turboquant_mse": {
            "2": mse_2,
            "4": mse_4,
        },
    }
    print(json.dumps(payload, indent=2))


@torch.no_grad()
def build_fokvq_bases(
    model,
    tokenizer,
    calibration_texts: Sequence[str],
    device: str,
    max_len: int,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    num_layers = model.config.num_hidden_layers
    per_layer_keys: Dict[int, List[torch.Tensor]] = {idx: [] for idx in range(num_layers)}

    for text in calibration_texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        ).to(device)
        outputs = model(**inputs, use_cache=True)
        legacy_cache, _ = to_legacy_cache(outputs.past_key_values)
        for layer_idx, layer_cache in enumerate(legacy_cache):
            layer_keys = layer_cache[0].detach().float().cpu()  # (b, h, s, d)
            flat = layer_keys.permute(1, 0, 2, 3).reshape(layer_keys.shape[1], -1, layer_keys.shape[-1])
            per_layer_keys[layer_idx].append(flat)
        del inputs, outputs, legacy_cache
        torch.cuda.empty_cache()

    bases = {}
    means = {}
    for layer_idx in range(num_layers):
        head_batches = torch.cat(per_layer_keys[layer_idx], dim=1).numpy()  # (h, n, d)
        head_bases = []
        head_means = []
        for head_idx in range(head_batches.shape[0]):
            head_means.append(torch.from_numpy(head_batches[head_idx].mean(axis=0).astype(np.float32)))
            basis_np = pca_basis_np(head_batches[head_idx])
            head_bases.append(torch.from_numpy(basis_np))
        bases[layer_idx] = torch.stack(head_bases, dim=0).to(device=device, dtype=torch.float32)
        means[layer_idx] = torch.stack(head_means, dim=0).to(device=device, dtype=torch.float32)
    return bases, means


def build_identity_bases(model, device: str) -> Dict[int, torch.Tensor]:
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    eye = torch.eye(head_dim, dtype=torch.float32, device=device)
    return {
        layer_idx: eye.unsqueeze(0).repeat(num_heads, 1, 1)
        for layer_idx in range(num_layers)
    }


def build_random_bases(model, device: str, seed: int) -> Dict[int, torch.Tensor]:
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    bases = {}
    for layer_idx in range(num_layers):
        layer_bases = []
        for head_idx in range(num_heads):
            layer_bases.append(
                compute_random_orthogonal_basis(
                    head_dim,
                    seed + layer_idx * 1000 + head_idx,
                )
            )
        bases[layer_idx] = torch.stack(layer_bases, dim=0).to(device=device, dtype=torch.float32)
    return bases


def quantize_cache(
    past_key_values,
    method: str,
    bits: int,
    fokvq_bases: Optional[Dict[int, torch.Tensor]],
    fokvq_means: Optional[Dict[int, torch.Tensor]],
    fokvq_topk_frac: float,
    fokvq_adaptive_energy_frac: float,
    fokvq_clip_quantile: float,
    turbo_codebooks: Optional[Dict[int, torch.Tensor]] = None,
) -> Tuple[object, Dict[str, float]]:
    legacy_cache, cache_cls = to_legacy_cache(past_key_values)
    quantized_layers = []
    key_sq_error = 0.0
    key_sq_count = 0
    for layer_idx, layer_cache in enumerate(legacy_cache):
        key_cache = layer_cache[0]
        other_items = list(layer_cache[1:])
        if method == "uniform":
            key_quant = symmetric_quantize_last_dim(key_cache, bits)
        elif method == "variance":
            key_quant = quantize_variance_keys(key_cache, bits)
        elif method == "kivi":
            key_quant = asymmetric_quantize_seq_dim(key_cache, bits)
        elif method in {"fokvq", "identity", "random", "fokvq_adaptive", "fokvq_clip", "fokvq_centered", "fokvq_centered_asym"}:
            if fokvq_bases is None:
                raise ValueError(f"Bases are required for method={method}")
            basis = fokvq_bases[layer_idx].to(device=key_cache.device)
            mean = None
            if method in {"fokvq_centered", "fokvq_centered_asym"}:
                if fokvq_means is None:
                    raise ValueError(f"Means are required for method={method}")
                mean = fokvq_means[layer_idx].to(device=key_cache.device)
            key_quant = quantize_nonuniform_with_basis(
                key_cache.float(),
                basis,
                bits,
                fokvq_topk_frac,
                adaptive_energy_frac=fokvq_adaptive_energy_frac if method == "fokvq_adaptive" else None,
                clip_quantile=fokvq_clip_quantile if method == "fokvq_clip" else None,
                mean=mean,
                seq_asymmetric=(method == "fokvq_centered_asym"),
            ).to(dtype=key_cache.dtype)
        elif method == "fokvq_lloyd":
            if fokvq_bases is None or turbo_codebooks is None:
                raise ValueError("fokvq_lloyd requires PCA bases and codebooks")
            basis = fokvq_bases[layer_idx].to(device=key_cache.device)
            codebook = turbo_codebooks[bits]
            key_quant = quantize_turboquant_keys(
                key_cache.float(),
                basis,
                codebook,
                clip_quantile=fokvq_clip_quantile,
            ).to(dtype=key_cache.dtype)
        elif method == "turboquant":
            if fokvq_bases is None or turbo_codebooks is None:
                raise ValueError("turboquant requires rotation bases and codebooks")
            basis = fokvq_bases[layer_idx].to(device=key_cache.device)
            codebook = turbo_codebooks[bits]
            key_quant = quantize_turboquant_keys(
                key_cache.float(),
                basis,
                codebook,
            ).to(dtype=key_cache.dtype)
        else:
            raise ValueError(f"Unsupported quantization method: {method}")
        diff = (key_cache.float() - key_quant.float()).pow(2)
        key_sq_error += float(diff.sum().item())
        key_sq_count += int(diff.numel())
        quantized_layers.append((key_quant,) + tuple(other_items))
    stats = {
        "key_mse_sum": key_sq_error,
        "key_mse_count": key_sq_count,
    }
    return from_legacy_cache(tuple(quantized_layers), cache_cls), stats


def negative_log_likelihood_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, int]:
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    loss = torch.nn.functional.cross_entropy(
        flat_logits,
        flat_labels,
        reduction="sum",
    )
    num_tokens = int(flat_labels.numel())
    return float(loss.item()), num_tokens


@torch.no_grad()
def score_full_window_suffix(
    model,
    window: torch.Tensor,
    suffix_len: int,
) -> Tuple[float, int]:
    outputs = model(window, use_cache=False)
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = window[:, 1:]
    valid_from = max(window.size(1) - suffix_len, 1) - 1
    return negative_log_likelihood_from_logits(
        shift_logits[:, valid_from:, :],
        shift_labels[:, valid_from:],
    )


@torch.no_grad()
def evaluate_fp16_sliding_window(
    model,
    input_ids: torch.Tensor,
    context_len: int,
    stride: int,
) -> Dict[str, float]:
    total_nll = 0.0
    total_tokens = 0
    prev_end = 0
    t0 = time.time()
    key_mse_sum = 0.0
    key_mse_count = 0

    for begin in range(0, input_ids.size(1), stride):
        end = min(begin + context_len, input_ids.size(1))
        trg_len = end - prev_end
        window = input_ids[:, begin:end]
        token_nll, num_tokens = score_full_window_suffix(model, window, trg_len)
        total_nll += token_nll
        total_tokens += num_tokens
        prev_end = end
        if end == input_ids.size(1):
            break
    ppl = math.exp(total_nll / max(total_tokens, 1))
    return {
        "ppl": ppl,
        "total_nll": total_nll,
        "total_tokens": total_tokens,
        "runtime_s": time.time() - t0,
    }


@torch.no_grad()
def evaluate_quantized_sliding_window(
    model,
    input_ids: torch.Tensor,
    context_len: int,
    stride: int,
    method: str,
    bits: int,
    fokvq_bases: Optional[Dict[int, torch.Tensor]],
    fokvq_means: Optional[Dict[int, torch.Tensor]],
    fokvq_topk_frac: float,
    fokvq_adaptive_energy_frac: float,
    fokvq_clip_quantile: float,
    turbo_codebooks: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[str, float]:
    total_nll = 0.0
    total_tokens = 0
    key_mse_sum = 0.0
    key_mse_count = 0
    prev_end = 0
    t0 = time.time()

    for begin in range(0, input_ids.size(1), stride):
        end = min(begin + context_len, input_ids.size(1))
        window = input_ids[:, begin:end]
        if window.size(1) <= 1:
            continue

        trg_len = end - prev_end
        prefix_len = max(0, window.size(1) - trg_len)
        if prefix_len == 0:
            token_nll, num_tokens = score_full_window_suffix(model, window, trg_len)
            total_nll += token_nll
            total_tokens += num_tokens
            prev_end = end
            continue

        prefix_ids = window[:, :prefix_len]
        target_ids = window[:, prefix_len:]

        prefix_outputs = model(prefix_ids, use_cache=True)
        quantized_cache, cache_stats = quantize_cache(
            prefix_outputs.past_key_values,
            method,
            bits,
            fokvq_bases,
            fokvq_means,
            fokvq_topk_frac,
            fokvq_adaptive_energy_frac,
            fokvq_clip_quantile,
            turbo_codebooks,
        )
        key_mse_sum += cache_stats["key_mse_sum"]
        key_mse_count += cache_stats["key_mse_count"]

        full_attention_mask = torch.ones(
            1,
            prefix_len + target_ids.size(1),
            device=target_ids.device,
            dtype=torch.long,
        )
        outputs = model(
            target_ids,
            past_key_values=quantized_cache,
            attention_mask=full_attention_mask,
            use_cache=False,
        )
        combined_logits = torch.cat(
            [
                prefix_outputs.logits[:, -1:, :],
                outputs.logits[:, :-1, :],
            ],
            dim=1,
        )
        token_nll, num_tokens = negative_log_likelihood_from_logits(combined_logits, target_ids)
        total_nll += token_nll
        total_tokens += num_tokens

        del prefix_outputs, quantized_cache, outputs
        torch.cuda.empty_cache()
        prev_end = end

        if end == input_ids.size(1):
            break

    ppl = math.exp(total_nll / max(total_tokens, 1))
    return {
        "ppl": ppl,
        "total_nll": total_nll,
        "total_tokens": total_tokens,
        "runtime_s": time.time() - t0,
        "avg_key_mse": (key_mse_sum / key_mse_count) if key_mse_count > 0 else 0.0,
    }


def maybe_truncate_ids(input_ids: torch.Tensor, max_eval_tokens: int) -> torch.Tensor:
    if max_eval_tokens <= 0:
        return input_ids
    return input_ids[:, :max_eval_tokens]


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

    print(f"Loading datasets for {args.model_key}...")
    train_texts, test_texts = load_wikitext_texts(cache_dir)
    calibration_texts = collect_calibration_texts(train_texts, args.calibration_samples)

    print(f"Loading model {args.model_name} on {args.device}...")
    tokenizer, model = get_tokenizer_and_model(
        args.model_name,
        dtype,
        args.device,
        cache_dir,
        args.attn_implementation,
    )

    test_ids = maybe_truncate_ids(encode_joined_text(tokenizer, test_texts, args.device), args.max_eval_tokens)

    summary = {
        "experiment": "exp4_2_standard_ppl_benchmark",
        "model_key": args.model_key,
        "model_name": args.model_name,
        "device": args.device,
        "dtype": str(dtype),
        "context_len": args.context_len,
        "stride": args.stride,
        "calibration_samples": args.calibration_samples,
        "calibration_max_len": args.calibration_max_len,
        "max_eval_tokens": args.max_eval_tokens,
        "methods": args.methods,
        "bits": args.bits,
        "fokvq_topk_frac": args.fokvq_topk_frac,
        "fokvq_adaptive_energy_frac": args.fokvq_adaptive_energy_frac,
        "fokvq_clip_quantile": args.fokvq_clip_quantile,
        "turboquant_note": "turboquant-style random rotation + Lloyd-Max scalar quantization; not an official full TurboQuant reproduction",
        "smoke": args.smoke,
        "results": {},
    }

    t0 = time.time()
    if "fp16" in args.methods:
        print("Running FP16 baseline...")
        summary["results"]["fp16"] = evaluate_fp16_sliding_window(
            model,
            test_ids,
            args.context_len,
            args.stride,
        )

    fokvq_basis_methods = {"fokvq", "fokvq_adaptive", "fokvq_clip", "fokvq_lloyd", "fokvq_centered", "fokvq_centered_asym"}
    fokvq_bases = None
    fokvq_means = None
    if any(method in fokvq_basis_methods for method in args.methods):
        print("Building FOKVQ PCA bases...")
        fokvq_bases, fokvq_means = build_fokvq_bases(
            model,
            tokenizer,
            calibration_texts,
            args.device,
            args.calibration_max_len,
        )
    identity_bases = build_identity_bases(model, args.device) if "identity" in args.methods else None
    random_bases = build_random_bases(model, args.device, args.seed) if "random" in args.methods else None
    turbo_bases = build_random_bases(model, args.device, args.seed + 10_000) if "turboquant" in args.methods else None
    needs_codebooks = any(method in {"turboquant", "fokvq_lloyd"} for method in args.methods)
    turbo_codebooks = build_standard_normal_codebooks(args.bits, args.seed) if needs_codebooks else None

    for method in [m for m in args.methods if m != "fp16"]:
        method_bases = None
        method_codebooks = None
        if method == "fokvq":
            method_bases = fokvq_bases
        elif method == "fokvq_centered":
            method_bases = fokvq_bases
        elif method == "fokvq_centered_asym":
            method_bases = fokvq_bases
        elif method == "fokvq_adaptive":
            method_bases = fokvq_bases
        elif method == "fokvq_clip":
            method_bases = fokvq_bases
        elif method == "fokvq_lloyd":
            method_bases = fokvq_bases
            method_codebooks = turbo_codebooks
        elif method == "identity":
            method_bases = identity_bases
        elif method == "random":
            method_bases = random_bases
        elif method == "turboquant":
            method_bases = turbo_bases
            method_codebooks = turbo_codebooks
        summary["results"][method] = {}
        for bits in args.bits:
            print(f"Running {method} {bits}-bit...")
            result = evaluate_quantized_sliding_window(
                model,
                test_ids,
                args.context_len,
                args.stride,
                method,
                bits,
                method_bases,
                fokvq_means,
                args.fokvq_topk_frac,
                args.fokvq_adaptive_energy_frac,
                args.fokvq_clip_quantile,
                method_codebooks,
            )
            if not math.isfinite(result["ppl"]):
                raise RuntimeError(f"Non-finite PPL for method={method}, bits={bits}")
            summary["results"][method][str(bits)] = result

    summary["runtime_s"] = time.time() - t0
    summary["peak_memory_gib"] = (
        torch.cuda.max_memory_allocated(torch.device(args.device)) / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )

    out_path = output_dir / f"{args.model_key}_standard_ppl.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    run()
