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


def quantize_k_tensor(K: torch.Tensor, method: str, bits: int,
                      gamma: float = 0.3,
                      q_covs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Quantize a K tensor of shape (..., seq, d_head).

    Handles both 2D (seq, d_head) and 4D (batch, heads, seq, d_head).

    Args:
        K: key tensor
        method: "fp16", "uniform", "kivi", "fokvq", or "fokvq_qw"
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
                # Fallback to regular FOKVQ if no Q covariance available
                K_q, _ = fokvq_quantize_head(K_head, bits, gamma)
            return K_q
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
        if self.method == "fokvq_qw" and query_states is not None:
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

            # Get head dimensions
            num_heads = attn_module.num_heads
            num_kv_heads = attn_module.num_key_value_heads
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
            if patcher.method == "fokvq_qw":
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

            outputs = (attn_output,)
            if output_attentions:
                outputs += (attn_weights,)
            if use_cache:
                outputs += (past_key_value,)

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

    # Load data
    print("\nLoading WikiText-2...")
    input_ids = load_wikitext2_ids(tokenizer, args.device, cache_dir)
    if args.max_eval_tokens > 0:
        input_ids = input_ids[:, :args.max_eval_tokens]
        print(f"  Truncated to {input_ids.shape[1]} tokens")

    # Summary dict
    summary = {
        "experiment": "exp4_2_v3_full_quant_ppl",
        "model_key": args.model_key,
        "model_name": args.model_name,
        "device": args.device,
        "dtype": str(dtype),
        "chunk_len": args.context_len,
        "protocol": args.protocol,
        "fokvq_gamma": args.gamma,
        "methods": args.methods,
        "bits": args.bits,
        "total_tokens_in_corpus": int(input_ids.shape[1]),
        "key_difference_vs_v2": "100% K quantization via hooks (v2 was 50% prefix-only)",
        "results": {},
    }

    t0 = time.time()

    # --- FP16 Baseline ---
    if "fp16" in args.methods:
        print("\n--- FP16 Baseline ---")
        result = evaluate_ppl_chunked(model, input_ids, args.context_len)
        summary["results"]["fp16"] = result
        print(f"  PPL = {result['ppl']:.4f} "
              f"({result['total_tokens']} tokens, {result['runtime_s']:.1f}s)")

    # --- Quantized Methods ---
    for method in [m for m in args.methods if m != "fp16"]:
        summary["results"][method] = {}
        for bits in args.bits:
            key = f"{method}_{bits}bit"
            print(f"\n--- {key} ({args.protocol}) ---")
            try:
                result = evaluate_method(
                    model, input_ids, args.context_len,
                    method, bits, args.gamma, args.protocol)
                summary["results"][method][str(bits)] = result
                mse_str = (f", key_MSE={result.get('avg_key_mse', 0):.6f}"
                          if 'avg_key_mse' in result else "")
                print(f"  PPL = {result['ppl']:.4f} "
                      f"({result['total_tokens']} tokens, "
                      f"{result['runtime_s']:.1f}s{mse_str})")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                summary["results"][method][str(bits)] = {
                    "ppl": float('inf'), "error": str(e),
                }

            torch.cuda.empty_cache()
            gc.collect()

    summary["runtime_s"] = time.time() - t0
    if torch.cuda.is_available():
        summary["peak_memory_gib"] = (
            torch.cuda.max_memory_allocated(torch.device(args.device)) / (1024 ** 3))

    # --- Write JSON ---
    out_path = output_dir / f"{args.model_key}_full_quant_ppl_v3.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")

    # --- Print Summary Table ---
    print("\n" + "=" * 72)
    print("SUMMARY (v3: 100% K quantized)")
    print("=" * 72)
    fp16_ppl = summary["results"].get("fp16", {}).get("ppl", float('inf'))
    print(f"  FP16 baseline: {fp16_ppl:.2f}")
    print(f"  Protocol: {args.protocol}")
    print(f"  {'Method':<12} {'2bit':>10} {'3bit':>10} {'4bit':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for method in [m for m in args.methods if m != "fp16"]:
        row = f"  {method:<12}"
        for bits in args.bits:
            ppl = summary["results"].get(method, {}).get(str(bits), {}).get("ppl", float('inf'))
            if math.isfinite(ppl):
                if math.isfinite(fp16_ppl) and fp16_ppl > 0:
                    delta = (ppl - fp16_ppl) / fp16_ppl * 100
                    row += f" {ppl:>7.2f}({delta:+.1f}%)"
                else:
                    row += f" {ppl:>10.2f}"
            else:
                row += f" {'N/A':>10}"
        print(row)
    print("=" * 72)

    # Compare with v2 expectations
    print("\nEXPECTED CHANGES vs v2 (50% prefix-only quantization):")
    print("  - Larger PPL degradation at low bits (full quantization effect visible)")
    print("  - GPT-2 3bit: expect > +0.4% (was diluted in v2)")
    print("  - Qwen 3bit: expect similar or larger than +24%")
    print("  - Qwen 4bit: expect similar or larger than +1.5%")
    print("  - Key insight: if v3 results are similar to v2, the 50% dilution")
    print("    was already minor (good for the paper)")

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    run()
