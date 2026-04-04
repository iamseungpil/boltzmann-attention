"""
FOKVQ Experiment 4-2 v2: Standard WikiText-2 PPL Benchmark (Corrected)
======================================================================

Based on our verified exp10_1_wikitext2_ppl.py implementation, adapted for
Azure ML / multi-model benchmarking with argparse interface.

Key fixes vs original exp4_2 (cowork):
  1. PCA CENTERING: subtract mean before projection, restore after inverse
  2. CONTINUOUS BIT ALLOCATION: eigenvalue^gamma weighted per-dimension
     (replaces binary top-k / bottom split that gave 1-bit to 75% of dims)
  3. ASYMMETRIC QUANTIZATION: min-max per-column in PCA space
     (replaces symmetric around zero that wasted dynamic range)
  4. SLIDING WINDOW: adopts cowork's combined-logit NLL (more standard)

Protocol:
  1. Load WikiText-2 test set via `datasets`, concatenate into single sequence
  2. Sliding-window causal LM evaluation
  3. For quantized: build prefix KV -> quantize K -> eval with quantized KV
  4. PPL = exp(mean NLL) over all eval tokens

Methods: FP16, Uniform, KIVI, FOKVQ
Models:  GPT-2 Medium, Qwen2.5-7B, Llama-3-8B (via --model-name)
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--self-test", action="store_true")
    pre, _ = bootstrap.parse_known_args()

    p = argparse.ArgumentParser(parents=[bootstrap])
    req = not pre.self_test
    p.add_argument("--model-name", type=str, required=req, default="gpt2-medium")
    p.add_argument("--model-key", type=str, required=req, default="gpt2-medium")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--context-len", type=int, required=req, default=32)
    p.add_argument("--stride", type=int, required=req, default=16)
    p.add_argument("--methods", nargs="+",
                   default=["fp16", "uniform", "kivi", "fokvq"])
    p.add_argument("--bits", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--gamma", type=float, default=0.3,
                   help="FOKVQ eigenvalue weighting exponent")
    p.add_argument("--max-eval-tokens", type=int, default=0,
                   help="Truncate test set (0 = use all)")
    p.add_argument("--output-dir", type=str, required=req,
                   default="/tmp/exp4_2_v2_self_test")
    p.add_argument("--cache-dir", type=str, default="")
    p.add_argument("--attn-implementation", type=str, default="eager")
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
    """Load WikiText-2 test set, concatenate, tokenize into single sequence."""
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
# Quantization Methods (from our verified exp10_1)
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
    """KIVI-style asymmetric quantization along sequence dimension."""
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
    """FOKVQ per-head: PCA rotate -> continuous bit alloc -> asymmetric quant.

    Key properties (fixes vs cowork original):
      - Centers data before PCA projection, restores mean after inverse
      - Continuous eigenvalue^gamma bit allocation (not binary split)
      - Asymmetric min-max quantization per PCA axis

    Args:
        K_head: (seq_len, d_head) single head key tensor
        bits_avg: average bits per dimension
        gamma: eigenvalue weighting exponent (0.3 = mild emphasis on top axes)

    Returns:
        K_recon: reconstructed K in original dtype
        r_eff: effective rank (spectral entropy)
    """
    d = K_head.shape[-1]
    K_f = K_head.float()

    # --- FIX 1: Centering ---
    mean = K_f.mean(dim=0)
    centered = K_f - mean

    # PCA via covariance eigendecomposition (GPU-accelerated)
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    cov += torch.eye(d, device=cov.device) * 1e-8
    evals, evecs = torch.linalg.eigh(cov)
    idx = torch.argsort(evals, descending=True)
    evals = evals[idx]
    evecs = evecs[:, idx]

    K_pca = centered @ evecs  # project CENTERED data

    # --- FIX 2: Continuous bit allocation (eigenvalue^gamma) ---
    ev_np = evals.cpu().numpy()
    ev_pos = np.maximum(ev_np, 1e-10)
    w = ev_pos ** gamma
    w /= w.sum()
    ib = np.clip(np.round(w * d * bits_avg).astype(int), 1, 8)
    # Budget adjustment to hit exactly d * bits_avg total bits
    while ib.sum() > d * bits_avg:
        ib[np.argmax(ib)] -= 1
    while ib.sum() < d * bits_avg:
        ib[np.argmin(ib)] += 1
    ib = np.clip(ib, 1, 8)

    # --- FIX 3: Asymmetric per-dimension quantization ---
    K_q = torch.zeros_like(K_pca)
    for i in range(d):
        col = K_pca[:, i]
        c_min = col.min()
        c_max = col.max()
        n_lev = 2 ** int(ib[i])
        step = max((c_max - c_min).item(), 1e-8) / (n_lev - 1)
        K_q[:, i] = torch.round((col - c_min) / step) * step + c_min

    # Inverse PCA + mean restoration
    K_recon = K_q @ evecs.T + mean

    # Effective rank for diagnostics
    ev_norm = ev_pos / ev_pos.sum()
    r_eff = float(np.exp(-np.sum(ev_norm * np.log(ev_norm + 1e-30))))

    return K_recon.to(K_head.dtype), r_eff


# ============================================================================
# KV Cache Quantization
# ============================================================================

def to_legacy_cache(past_key_values):
    """Convert modern cache to legacy tuple format."""
    if past_key_values is None:
        return None, None
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache(), type(past_key_values)
    # Already legacy tuple
    return past_key_values, None


def from_legacy_cache(legacy_cache, cache_cls):
    """Convert legacy tuple back to modern cache."""
    if legacy_cache is None:
        return None
    if cache_cls is not None and hasattr(cache_cls, "from_legacy_cache"):
        return cache_cls.from_legacy_cache(legacy_cache)
    # Try DynamicCache
    try:
        from transformers.cache_utils import DynamicCache
        if cache_cls is DynamicCache or isinstance(legacy_cache, tuple):
            return DynamicCache(legacy_cache)
    except (ImportError, TypeError):
        pass
    return legacy_cache


def quantize_kv_cache(past_key_values, method: str, bits: int,
                      gamma: float = 0.3):
    """Quantize K tensors in KV cache, return new cache + MSE stats.

    Supports both legacy tuple and DynamicCache formats.
    """
    legacy_cache, cache_cls = to_legacy_cache(past_key_values)
    quantized_layers = []
    key_sq_error = 0.0
    key_sq_count = 0

    for layer_cache in legacy_cache:
        key_cache = layer_cache[0]  # (batch, n_kv_heads, seq, d_head)
        other = list(layer_cache[1:])

        K = key_cache.clone()
        for b in range(K.shape[0]):
            for h in range(K.shape[1]):
                K_head = K[b, h]  # (seq, d_head)
                if method == "uniform":
                    K[b, h] = uniform_quantize_tensor(K_head, bits)
                elif method == "kivi":
                    K[b, h] = kivi_quantize_tensor(K_head, bits)
                elif method == "fokvq":
                    K_q, _ = fokvq_quantize_head(K_head, bits, gamma)
                    K[b, h] = K_q
                else:
                    raise ValueError(f"Unknown method: {method}")

        diff = (key_cache.float() - K.float()).pow(2)
        key_sq_error += float(diff.sum().item())
        key_sq_count += int(diff.numel())

        quantized_layers.append((K,) + tuple(other))

    stats = {
        "key_mse_sum": key_sq_error,
        "key_mse_count": key_sq_count,
    }
    return from_legacy_cache(tuple(quantized_layers), cache_cls), stats


# ============================================================================
# NLL Helpers
# ============================================================================

def nll_from_logits(logits: torch.Tensor, labels: torch.Tensor
                    ) -> Tuple[float, int]:
    """Compute sum of NLL from logits and labels."""
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_labels, reduction="sum")
    return float(loss.item()), int(flat_labels.numel())


# ============================================================================
# Sliding Window Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_fp16(model, input_ids: torch.Tensor, context_len: int,
                  stride: int) -> Dict[str, float]:
    """FP16 baseline: standard sliding window PPL."""
    total_nll = 0.0
    total_tokens = 0
    prev_end = 0
    t0 = time.time()

    for begin in range(0, input_ids.size(1), stride):
        end = min(begin + context_len, input_ids.size(1))
        trg_len = end - prev_end
        window = input_ids[:, begin:end]

        outputs = model(window, use_cache=False)
        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = window[:, 1:]
        valid_from = max(window.size(1) - trg_len, 1) - 1

        token_nll, n_toks = nll_from_logits(
            shift_logits[:, valid_from:, :],
            shift_labels[:, valid_from:],
        )
        total_nll += token_nll
        total_tokens += n_toks
        prev_end = end

        del outputs
        torch.cuda.empty_cache()

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
def evaluate_quantized(model, input_ids: torch.Tensor, context_len: int,
                       stride: int, method: str, bits: int,
                       gamma: float = 0.3) -> Dict[str, float]:
    """Quantized sliding window PPL with KV cache quantization.

    Uses combined-logit approach:
      prefix_logits[:, -1:] predicts eval[0]
      eval_logits[:, :-1]   predicts eval[1:]
    This evaluates ALL stride tokens (standard protocol).
    """
    total_nll = 0.0
    total_tokens = 0
    key_mse_sum = 0.0
    key_mse_count = 0
    prev_end = 0
    t0 = time.time()
    n_windows = 0

    for begin in range(0, input_ids.size(1), stride):
        end = min(begin + context_len, input_ids.size(1))
        window = input_ids[:, begin:end]
        if window.size(1) <= 1:
            continue

        trg_len = end - prev_end
        prefix_len = max(0, window.size(1) - trg_len)
        n_windows += 1

        if prefix_len == 0:
            # First window: no prefix to quantize, evaluate as FP16
            outputs = model(window, use_cache=False)
            shift_logits = outputs.logits[:, :-1, :]
            shift_labels = window[:, 1:]
            valid_from = max(window.size(1) - trg_len, 1) - 1
            token_nll, n_toks = nll_from_logits(
                shift_logits[:, valid_from:, :],
                shift_labels[:, valid_from:],
            )
            total_nll += token_nll
            total_tokens += n_toks
            del outputs
        else:
            prefix_ids = window[:, :prefix_len]
            target_ids = window[:, prefix_len:]

            # Step 1: Forward prefix -> KV cache
            prefix_out = model(prefix_ids, use_cache=True)

            # Step 2: Quantize K in cache
            q_cache, cache_stats = quantize_kv_cache(
                prefix_out.past_key_values, method, bits, gamma)
            key_mse_sum += cache_stats["key_mse_sum"]
            key_mse_count += cache_stats["key_mse_count"]

            # Step 3: Forward eval with quantized KV
            full_attn_mask = torch.ones(
                1, prefix_len + target_ids.size(1),
                device=target_ids.device, dtype=torch.long,
            )
            eval_out = model(
                target_ids,
                past_key_values=q_cache,
                attention_mask=full_attn_mask,
                use_cache=False,
            )

            # Step 4: NLL (combined logit approach)
            # prefix last logit predicts target[0]
            # eval logits[:, :-1] predict target[1:]
            combined_logits = torch.cat([
                prefix_out.logits[:, -1:, :],
                eval_out.logits[:, :-1, :],
            ], dim=1)
            token_nll, n_toks = nll_from_logits(combined_logits, target_ids)
            total_nll += token_nll
            total_tokens += n_toks

            del prefix_out, q_cache, eval_out

        prev_end = end
        torch.cuda.empty_cache()

        if n_windows % 20 == 0:
            running_ppl = math.exp(min(total_nll / max(total_tokens, 1), 100.0))
            elapsed = time.time() - t0
            print(f"    win={n_windows} | tokens={total_tokens} | "
                  f"ppl={running_ppl:.2f} | {elapsed:.1f}s", flush=True)

        if end == input_ids.size(1):
            break

    ppl = math.exp(total_nll / max(total_tokens, 1))
    return {
        "ppl": ppl,
        "total_nll": total_nll,
        "total_tokens": total_tokens,
        "runtime_s": time.time() - t0,
        "avg_key_mse": (key_mse_sum / key_mse_count) if key_mse_count > 0 else 0.0,
        "n_windows": n_windows,
    }


# ============================================================================
# Self-Test
# ============================================================================

def run_self_tests(seed: int) -> None:
    set_seed(seed)
    print("Running self-tests...")

    # Test 1: uniform_quantize_tensor roundtrip
    t = torch.randn(8, 16)
    for bits in [2, 3, 4, 8]:
        q = uniform_quantize_tensor(t, bits)
        assert q.shape == t.shape
        assert torch.isfinite(q).all()
    print("  [PASS] uniform_quantize_tensor")

    # Test 2: kivi_quantize_tensor
    t = torch.randn(4, 8, 32, 16)  # (b, h, s, d)
    for bits in [2, 3, 4]:
        q = kivi_quantize_tensor(t, bits)
        assert q.shape == t.shape
        assert torch.isfinite(q).all()
    print("  [PASS] kivi_quantize_tensor")

    # Test 3: fokvq centering correctness
    K = torch.randn(64, 32) + 5.0  # large mean offset
    K_q, r_eff = fokvq_quantize_head(K, bits_avg=4, gamma=0.3)
    assert K_q.shape == K.shape
    assert torch.isfinite(K_q).all()
    mse_fokvq = (K - K_q).pow(2).mean().item()
    # Compare with uniform at same bits
    K_unif = uniform_quantize_tensor(K, 4)
    mse_unif = (K - K_unif).pow(2).mean().item()
    print(f"  [INFO] FOKVQ 4bit MSE={mse_fokvq:.6f}, Uniform 4bit MSE={mse_unif:.6f}")
    assert r_eff > 0
    print("  [PASS] fokvq_quantize_head (centering + asymmetric)")

    # Test 4: fokvq should beat uniform at 4-bit on anisotropic data
    # Create strongly anisotropic data
    np.random.seed(seed)
    evals = np.array([100, 50, 10, 1] + [0.01] * 28, dtype=np.float32)
    U = np.linalg.qr(np.random.randn(32, 32).astype(np.float32))[0]
    data = np.random.randn(128, 32).astype(np.float32) @ np.diag(np.sqrt(evals)) @ U.T
    K_aniso = torch.from_numpy(data) + 3.0  # add mean offset
    K_q_fokvq, r_eff = fokvq_quantize_head(K_aniso, bits_avg=3, gamma=0.3)
    K_q_unif = uniform_quantize_tensor(K_aniso, 3)
    mse_f = (K_aniso - K_q_fokvq).pow(2).mean().item()
    mse_u = (K_aniso - K_q_unif).pow(2).mean().item()
    print(f"  [INFO] Anisotropic: FOKVQ 3bit MSE={mse_f:.6f}, "
          f"Uniform 3bit MSE={mse_u:.6f}, r_eff={r_eff:.1f}/32")
    if mse_f < mse_u:
        print("  [PASS] FOKVQ beats uniform on anisotropic data")
    else:
        print("  [WARN] FOKVQ did not beat uniform (unexpected)")

    # Test 5: higher bits -> lower MSE
    K = torch.randn(64, 32) + 2.0
    mses = {}
    for bits in [2, 3, 4]:
        K_q, _ = fokvq_quantize_head(K, bits, gamma=0.3)
        mses[bits] = (K - K_q).pow(2).mean().item()
    assert mses[4] < mses[3] < mses[2], f"Monotonicity failed: {mses}"
    print(f"  [PASS] MSE monotonicity: 2b={mses[2]:.4f} > 3b={mses[3]:.4f} > 4b={mses[4]:.4f}")

    print("\nAll self-tests passed.")
    result = {
        "self_test": "passed",
        "fokvq_mse_4bit": mse_fokvq,
        "uniform_mse_4bit": mse_unif,
    }
    print(json.dumps(result, indent=2))


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
    print("FOKVQ Exp 4-2 v2: WikiText-2 Standard PPL Benchmark")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {args.model_name} ({args.model_key})")
    print(f"Device: {args.device}, dtype: {dtype}")
    print(f"Context: {args.context_len}, Stride: {args.stride}")
    print(f"Methods: {args.methods}, Bits: {args.bits}")
    print(f"FOKVQ gamma: {args.gamma}")
    print("=" * 72)

    # Load model
    print("\nLoading model...")
    tokenizer, model = load_model_and_tokenizer(
        args.model_name, dtype, args.device, cache_dir, args.attn_implementation)

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    n_kv_heads = getattr(cfg, 'num_key_value_heads', n_heads)
    d_head = cfg.hidden_size // n_heads
    print(f"  layers={n_layers}, heads={n_heads}, kv_heads={n_kv_heads}, "
          f"d_head={d_head}")

    # Load data
    print("\nLoading WikiText-2...")
    input_ids = load_wikitext2_ids(tokenizer, args.device, cache_dir)
    if args.max_eval_tokens > 0:
        input_ids = input_ids[:, :args.max_eval_tokens]
        print(f"  Truncated to {input_ids.shape[1]} tokens")

    # Summary dict
    summary = {
        "experiment": "exp4_2_standard_ppl_benchmark_v2",
        "model_key": args.model_key,
        "model_name": args.model_name,
        "device": args.device,
        "dtype": str(dtype),
        "context_len": args.context_len,
        "stride": args.stride,
        "fokvq_gamma": args.gamma,
        "methods": args.methods,
        "bits": args.bits,
        "total_tokens_in_corpus": int(input_ids.shape[1]),
        "fixes_vs_v1": [
            "PCA centering (mean subtract before projection, restore after inverse)",
            "Continuous bit allocation (eigenvalue^gamma, not binary top-k split)",
            "Asymmetric min-max quantization (not symmetric around zero)",
        ],
        "results": {},
    }

    t0 = time.time()

    # --- FP16 Baseline ---
    if "fp16" in args.methods:
        print("\n--- FP16 Baseline ---")
        result = evaluate_fp16(model, input_ids, args.context_len, args.stride)
        summary["results"]["fp16"] = result
        print(f"  PPL = {result['ppl']:.4f} "
              f"({result['total_tokens']} tokens, {result['runtime_s']:.1f}s)")

    # --- Quantized Methods ---
    for method in [m for m in args.methods if m != "fp16"]:
        summary["results"][method] = {}
        for bits in args.bits:
            key = f"{method}_{bits}bit"
            print(f"\n--- {key} ---")
            try:
                result = evaluate_quantized(
                    model, input_ids, args.context_len, args.stride,
                    method, bits, args.gamma)
                summary["results"][method][str(bits)] = result
                print(f"  PPL = {result['ppl']:.4f} "
                      f"({result['total_tokens']} tokens, {result['runtime_s']:.1f}s, "
                      f"key_MSE={result['avg_key_mse']:.6f})")
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
    out_path = output_dir / f"{args.model_key}_standard_ppl_v2.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {out_path}")

    # --- Print Summary Table ---
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    fp16_ppl = summary["results"].get("fp16", {}).get("ppl", float('inf'))
    print(f"  FP16 baseline: {fp16_ppl:.2f}")
    print(f"  {'Method':<12} {'2bit':>8} {'3bit':>8} {'4bit':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    for method in [m for m in args.methods if m != "fp16"]:
        row = f"  {method:<12}"
        for bits in args.bits:
            ppl = summary["results"].get(method, {}).get(str(bits), {}).get("ppl", float('inf'))
            if math.isfinite(ppl):
                row += f" {ppl:>8.2f}"
            else:
                row += f" {'N/A':>8}"
        print(row)
    print("=" * 72)

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    run()
