#!/usr/bin/env python3
"""
Downstream task evaluation with quantized KV cache.
Uses lm-evaluation-harness with k_proj forward hooks.

Usage:
  python run_downstream.py --model meta-llama/Llama-3.1-8B --method pre_pca_wf2 --bits 2 --tasks mmlu --device cuda:0
  python run_downstream.py --model meta-llama/Llama-3.1-8B --method fp16 --tasks mmlu --device cuda:0
  python run_downstream.py --model meta-llama/Llama-3.1-8B --method turbo --bits 2 --tasks gsm8k --device cuda:0
  python run_downstream.py --self-test
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

OUTDIR = Path(__file__).parent / "verification_results"
HARNESS_SCOPE = "scope-limited same-harness downstream ablation"
RUNNER_NAME = "run_downstream.py"
RUNNER_VERSION = "v2_provenance_hardened"
HARNESS_LIMITATIONS = [
    "Only decoder models with model.layers[*].self_attn.{q_proj,k_proj} are supported.",
    "Quantized entries are same-harness methods or same-harness proxy controls only.",
    "This runner is not an official reproduction of KIVI, TurboQuant, or exp4_2_v3 full-quant PPL.",
    "Method aliases retained for plan vocabulary may map to simpler runner-local implementations.",
]
CALIBRATION_ARTIFACT_IDENTITY = {
    "dataset": "wikitext",
    "subset": "wikitext-2-raw-v1",
    "split": "train",
    "text_selection": "first 50 non-empty texts joined by double newlines",
    "max_length": 2048,
    "space": "pre-RoPE k_proj outputs",
}

METHOD_SPECS = {
    "fp16": {
        "display_name": "FP16",
        "canonical_name": "fp16",
        "baseline_family": "full_precision",
        "implementation_key": "fp16_reference",
        "implementation_source": "run_downstream.py",
        "implementation_scope": "no quantization",
        "official_reproduction": False,
        "pilot_only": False,
        "method_provenance": "identity forward without KV quantization",
        "support_level": "supported",
        "claim_label": "FP16 reference",
    },
    "no_rot": {
        "display_name": "NoRot Uniform",
        "canonical_name": "no_rot",
        "baseline_family": "same_harness_control",
        "implementation_key": "identity_basis_uniform_ref",
        "implementation_source": "scripts/exp4_2_v3_full_quant_ppl.py",
        "implementation_scope": "identity-basis quantization shared with the v3 full-quant harness",
        "official_reproduction": False,
        "pilot_only": False,
        "method_provenance": "same-harness control imported from the shared v3 quantizer module",
        "support_level": "supported",
        "claim_label": "No-rotation uniform control",
    },
    "turbo": {
        "display_name": "Turbo Proxy",
        "canonical_name": "turboquant_rand",
        "baseline_family": "turboquant_proxy",
        "implementation_key": "turboquant_rand_proxy_ref",
        "implementation_source": "scripts/exp4_2_v3_full_quant_ppl.py",
        "implementation_scope": "single random orthogonal rotation plus uniform scalar quantization",
        "official_reproduction": False,
        "pilot_only": True,
        "method_provenance": "TurboQuant-inspired same-harness proxy imported from the shared v3 quantizer module",
        "support_level": "supported_alias",
        "claim_label": "TurboQuant-inspired proxy",
    },
    "turboquant_rand": {
        "display_name": "TurboQuant Proxy",
        "canonical_name": "turboquant_rand",
        "baseline_family": "turboquant_proxy",
        "implementation_key": "turboquant_rand_proxy_ref",
        "implementation_source": "scripts/exp4_2_v3_full_quant_ppl.py",
        "implementation_scope": "single random orthogonal rotation plus uniform scalar quantization",
        "official_reproduction": False,
        "pilot_only": True,
        "method_provenance": "TurboQuant-inspired same-harness proxy imported from the shared v3 quantizer module",
        "support_level": "supported",
        "claim_label": "TurboQuant-inspired proxy",
    },
    "pre_pca_uni": {
        "display_name": "Pre-RoPE PCA + Uniform",
        "canonical_name": "pre_pca_uni",
        "baseline_family": "fokvq_core",
        "implementation_key": "fokvq_uniform_ref",
        "implementation_source": "scripts/exp4_2_v3_full_quant_ppl.py",
        "implementation_scope": "shared same-harness FOKVQ uniform quantizer imported from the v3 module",
        "official_reproduction": False,
        "pilot_only": False,
        "method_provenance": "same-harness candidate imported from the shared v3 quantizer module",
        "support_level": "supported",
        "claim_label": "Pre-RoPE PCA + uniform",
    },
    "pre_pca_wf2": {
        "display_name": "Pre-RoPE PCA + WF(floor=2)",
        "canonical_name": "pre_pca_wf2",
        "baseline_family": "fokvq_extension",
        "implementation_key": "pre_rope_pca_waterfill_uniform",
        "implementation_source": "run_downstream.py",
        "implementation_scope": "per-head pre-RoPE PCA rotation plus waterfilling-style bit allocation",
        "official_reproduction": False,
        "pilot_only": False,
        "method_provenance": "local FOKVQ extension in the downstream harness",
        "support_level": "supported",
        "claim_label": "Pre-RoPE PCA + WF(floor=2)",
    },
    "fokvq_e2": {
        "display_name": "FOKVQ-E2",
        "canonical_name": "fokvq_e2",
        "baseline_family": "fokvq_candidate",
        "implementation_key": "fokvq_e2_ref",
        "implementation_source": "scripts/exp4_2_v3_full_quant_ppl.py",
        "implementation_scope": "shared same-harness FOKVQ-E2 quantizer imported from the v3 module",
        "official_reproduction": False,
        "pilot_only": False,
        "method_provenance": "same-harness candidate imported from the shared v3 quantizer module",
        "support_level": "supported",
        "claim_label": "FOKVQ-E2 same-harness candidate",
    },
    "fokvq_e2_residual": {
        "display_name": "FOKVQ-E2 Residual",
        "canonical_name": "fokvq_e2_residual",
        "baseline_family": "fokvq_candidate",
        "implementation_key": "fokvq_e2_residual_ref",
        "implementation_source": "scripts/exp4_2_v3_full_quant_ppl.py",
        "implementation_scope": "shared same-harness FOKVQ-E2 residual quantizer imported from the v3 module",
        "official_reproduction": False,
        "pilot_only": False,
        "method_provenance": "same-harness candidate imported from the shared v3 quantizer module",
        "support_level": "supported",
        "claim_label": "FOKVQ-E2 residual same-harness candidate",
    },
    "kivi_residual": {
        "display_name": "KIVI-style Residual Proxy",
        "canonical_name": "kivi_residual",
        "baseline_family": "kivi_proxy",
        "implementation_key": "kivi_residual_proxy_ref",
        "implementation_source": "scripts/exp4_2_v3_full_quant_ppl.py",
        "implementation_scope": "grouped prefix quantization plus FP residual tail",
        "official_reproduction": False,
        "pilot_only": True,
        "method_provenance": "KIVI-style same-harness proxy imported from the shared v3 quantizer module, not the official KIVI implementation",
        "support_level": "supported",
        "claim_label": "KIVI-style proxy",
    },
}


def get_method_spec(method: str) -> dict:
    if method not in METHOD_SPECS:
        supported = ", ".join(sorted(METHOD_SPECS))
        raise ValueError(f"Unsupported downstream quantization method: {method}. Supported: {supported}")
    spec = dict(METHOD_SPECS[method])
    spec["requested_name"] = method
    return spec


def resolve_dtype(model_name: str, dtype_arg: str) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float32":
        return torch.float32
    lowered = model_name.lower()
    if "qwen" in lowered or "llama" in lowered or "mistral" in lowered:
        return torch.bfloat16
    return torch.float16


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


_REFERENCE_QUANTIZERS = None


def load_reference_quantizers():
    global _REFERENCE_QUANTIZERS
    if _REFERENCE_QUANTIZERS is not None:
        return _REFERENCE_QUANTIZERS

    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "exp4_2_v3_full_quant_ppl.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("exp4_2_v3_full_quant_ppl_ref", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load reference quantizer module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _REFERENCE_QUANTIZERS = module
    return module


def resolve_attention_stack(model):
    """Return attention-layer metadata for architectures with per-layer k_proj.

    This downstream runner is intentionally scoped to decoder architectures
    whose attention blocks expose `self_attn.k_proj` and `self_attn.q_proj`.
    That matches the current Qwen/Llama/Mistral experiments. GPT-2 style
    combined-qkv blocks are out of scope for this harness and should fail with
    a clear error instead of an attribute crash during calibration.
    """
    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    d_head = cfg.hidden_size // n_heads

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise ValueError(
            "run_downstream.py currently supports decoder models with "
            "`model.layers[*].self_attn.{q_proj,k_proj}` only. "
            f"Unsupported model_type={getattr(cfg, 'model_type', 'unknown')}."
        )

    sample_block = layers[0]
    attn = getattr(sample_block, "self_attn", None)
    if attn is None or not hasattr(attn, "k_proj") or not hasattr(attn, "q_proj"):
        raise ValueError(
            "Unsupported attention block for downstream KV quantization harness: "
            "expected `self_attn.k_proj` and `self_attn.q_proj`."
        )

    return {
        "cfg": cfg,
        "n_heads": n_heads,
        "n_kv": n_kv,
        "d_head": d_head,
        "n_layers": cfg.num_hidden_layers,
        "layers": layers,
    }

# ============================================================================
# Quantization (reuse from verify_v15)
# ============================================================================
def uniform_quant_col(col, bits):
    nl = 2 ** bits
    vmin, vmax = col.min(), col.max()
    if vmax - vmin < 1e-10:
        return col.copy()
    s = (vmax - vmin) / (nl - 1)
    q = np.clip(np.round((col - vmin) / s).astype(int), 0, nl - 1)
    return q * s + vmin


def compute_wf_alloc(variances, avg_bits, min_bits=2):
    d = len(variances)
    max_bits = 4
    log_var = np.log2(np.maximum(variances, 1e-10))
    mean_log_var = np.mean(log_var)
    b_alloc = avg_bits + 0.5 * (log_var - mean_log_var)
    b_alloc = np.clip(b_alloc, min_bits, max_bits)
    total_target = d * avg_bits
    b_alloc = b_alloc * (total_target / max(b_alloc.sum(), 1e-10))
    b_alloc = np.maximum(np.round(b_alloc), min_bits)
    b_alloc = np.minimum(b_alloc, max_bits)
    deficit = int(total_target - b_alloc.sum())
    if deficit > 0:
        headroom = max_bits - b_alloc
        idx = np.argsort(-headroom)
        for i in idx[:deficit]:
            if b_alloc[i] < max_bits:
                b_alloc[i] += 1
    elif deficit < 0:
        slack = b_alloc - min_bits
        idx = np.argsort(slack)
        for i in idx[:abs(deficit)]:
            if b_alloc[i] > min_bits:
                b_alloc[i] -= 1
    return b_alloc.astype(int)


def quantize_grouped_residual_prefix(Kh: np.ndarray,
                                     bits: int,
                                     group_size: int = 32,
                                     residual_length: int = 32) -> np.ndarray:
    """Simple same-harness KIVI-style proxy: grouped prefix quant + FP tail."""
    if Kh.shape[0] <= residual_length:
        return Kh.copy()

    prefix = Kh[:-residual_length].copy()
    tail = Kh[-residual_length:].copy()
    out = prefix.copy()
    for start in range(0, prefix.shape[0], group_size):
        end = min(start + group_size, prefix.shape[0])
        group = prefix[start:end]
        for j in range(group.shape[1]):
            out[start:end, j] = uniform_quant_col(group[:, j], bits)
    return np.concatenate([out, tail], axis=0)


def quantize_head_with_method(Kh: np.ndarray,
                              method: str,
                              bits: int,
                              layer_idx: int,
                              head_idx: int,
                              pca_bases: dict | None,
                              pca_means: dict | None,
                              random_rot: np.ndarray | None,
                              wf_alloc: dict | None,
                              residual_length: int = 32) -> np.ndarray:
    """Quantize one head with method names aligned to the unified plan."""
    ref = load_reference_quantizers()
    kh_t = torch.from_numpy(Kh).to(dtype=torch.float32)

    if method in ("fp16",):
        return Kh.copy()

    if method in ("no_rot",):
        return ref.identity_basis_quantize_head(kh_t, bits).cpu().numpy()

    if method in ("kivi_residual",):
        return ref.kivi_residual_quantize_head(
            kh_t, bits, group_size=32, residual_length=residual_length
        ).cpu().numpy()

    if method in ("turbo", "turboquant_rand"):
        return ref.turbo_quantize_random_head(kh_t, bits).cpu().numpy()

    if method in ("pre_pca_uni",):
        k_q, _ = ref.fokvq_quantize_head(kh_t, bits, gamma=0.3)
        return k_q.cpu().numpy()

    if method in ("fokvq_e2",):
        k_q, _ = ref.fokvq_e2_quantize_head(kh_t, bits, gamma=0.3)
        return k_q.cpu().numpy()

    if method in ("fokvq_e2_residual",):
        k_q, _ = ref.fokvq_e2_residual_quantize_head(
            kh_t, bits, gamma=0.3, residual_length=residual_length
        )
        return k_q.cpu().numpy()

    if method in ("pre_pca_wf2",):
        R = pca_bases[(layer_idx, head_idx)]
        mean = pca_means[(layer_idx, head_idx)] if pca_means is not None else np.zeros(Kh.shape[1], dtype=Kh.dtype)
    else:
        raise ValueError(f"Unsupported downstream quantization method: {method}")

    Kh_centered = Kh - mean
    Kr = Kh_centered @ R if R is not None else Kh_centered.copy()
    quant_rows = Kr.shape[0]
    if method == "fokvq_e2_residual":
        quant_rows = max(0, Kr.shape[0] - residual_length)
    for j in range(Kr.shape[1]):
        b = bits
        if method in ("pre_pca_wf2",):
            b = int(wf_alloc[(layer_idx, head_idx)][j])
            b = max(min(b, 4), 1)
        if quant_rows > 0:
            Kr[:quant_rows, j] = uniform_quant_col(Kr[:quant_rows, j], b)
    Krec = Kr @ R.T if R is not None else Kr
    return Krec + mean


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--self-test", action="store_true")
    pre, _ = bootstrap.parse_known_args()

    parser = argparse.ArgumentParser(parents=[bootstrap])
    req = not pre.self_test
    parser.add_argument("--model", type=str, required=req)
    parser.add_argument("--method", type=str, default="fp16",
                        choices=[
                            "fp16", "no_rot", "turbo", "turboquant_rand",
                            "pre_pca_uni", "pre_pca_wf2", "fokvq_e2",
                            "fokvq_e2_residual", "kivi_residual"
                        ])
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--tasks", type=str, nargs="+", default=["mmlu"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--cache-dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per task (for testing)")
    return parser.parse_args()


def build_output_metadata(args: argparse.Namespace,
                          model,
                          tokenizer,
                          stack: dict,
                          lm_eval_version: str | None = None) -> dict:
    import transformers

    method_spec = get_method_spec(args.method)
    dtype = str(next(model.parameters()).dtype)
    return {
        "generated_at": datetime.now().isoformat(),
        "hostname": platform.node(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "lm_eval_version": lm_eval_version,
        "git_head": get_git_head(),
        "harness_name": RUNNER_NAME,
        "harness_version": RUNNER_VERSION,
        "model": args.model,
        "model_type": getattr(model.config, "model_type", "unknown"),
        "model_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("revision"),
        "method": args.method,
        "bits": args.bits,
        "display_name": method_spec["display_name"],
        "requested_method": method_spec["requested_name"],
        "canonical_method": method_spec["canonical_name"],
        "claim_label": method_spec["claim_label"],
        "support_level": method_spec["support_level"],
        "baseline_family": method_spec["baseline_family"],
        "implementation_key": method_spec["implementation_key"],
        "implementation_source": method_spec["implementation_source"],
        "method_provenance": method_spec["method_provenance"],
        "implementation_scope": method_spec["implementation_scope"],
        "official_reproduction": method_spec["official_reproduction"],
        "pilot_only": method_spec["pilot_only"],
        "harness_scope": HARNESS_SCOPE,
        "harness_limitations": HARNESS_LIMITATIONS,
        "same_harness_only": True,
        "overclaim_guard": (
            "Do not cite this as an official KIVI/TurboQuant/full-v3 reproduction; "
            "interpret only within this downstream runner's same-harness scope."
        ),
        "architecture_support": "decoder models with model.layers[*].self_attn.{q_proj,k_proj}",
        "calibration_source": "wikitext-2-raw-v1 train split, first 50 non-empty texts, max_length=2048",
        "calibration_dataset_revision": "wikitext-2-raw-v1",
        "calibration_seed": args.seed,
        "quantization_seed": args.seed,
        "calibration_artifact_identity": {
            **CALIBRATION_ARTIFACT_IDENTITY,
            "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", args.model),
        },
        "calibration_tokens_max_length": 2048,
        "tasks_requested": list(args.tasks),
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "device": args.device,
        "dtype": dtype,
        "attn_implementation": args.attn_implementation,
        "cache_dir": args.cache_dir or None,
        "pilot_scope": args.limit is not None,
        "prompt_scoring_path": "lm_eval.simple_evaluate -> HFLM(batch_size=1)",
        "kv_heads": stack["n_kv"],
        "attention_heads": stack["n_heads"],
        "d_head": stack["d_head"],
        "n_layers": stack["n_layers"],
    }


# ============================================================================
# Hook factory
# ============================================================================
def make_k_quant_hook(layer_idx, bits, n_kv, d_head,
                      pca_bases=None, pca_means=None, random_rot=None, wf_alloc=None):
    def hook_fn(module, input, output):
        k = output[0] if isinstance(output, tuple) else output
        k_np = k.detach().cpu().float().numpy()
        orig_shape = k_np.shape
        k_flat = k_np.reshape(-1, n_kv, d_head)

        for h in range(n_kv):
            Kh = k_flat[:, h, :]
            method = hook_fn.method
            k_flat[:, h, :] = quantize_head_with_method(
                Kh, method, bits, layer_idx, h,
                pca_bases=pca_bases, pca_means=pca_means, random_rot=random_rot, wf_alloc=wf_alloc
            )

        return torch.tensor(k_flat.reshape(orig_shape), dtype=k.dtype, device=k.device)
    return hook_fn


# ============================================================================
# Calibration + hook installation
# ============================================================================
def calibrate_and_install_hooks(model, method, bits, device):
    """Calibrate PCA bases and install quantization hooks."""
    from transformers import AutoTokenizer
    from datasets import load_dataset

    method_spec = get_method_spec(method)
    stack = resolve_attention_stack(model)
    cfg = stack["cfg"]
    n_kv = stack["n_kv"]
    n_layers = stack["n_layers"]
    d_head = stack["d_head"]
    layers = stack["layers"]

    if method == 'fp16':
        return []  # No hooks

    # Calibration: extract pre-RoPE keys
    kwargs = {}
    cache_dir = getattr(model, "_downstream_cache_dir", None)
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, trust_remote_code=True, **kwargs)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", **kwargs)
    text = "\n\n".join([t for t in ds["text"] if t.strip()][:50])
    cal_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=2048).to(device)

    pre_capture = {}
    cal_hooks = []

    def make_cal_hook(li):
        def fn(mod, args_t, kwargs_t):
            hs = args_t[0] if args_t else kwargs_t.get('hidden_states')
            if hs is not None:
                k = mod.k_proj(hs)[0].detach().cpu().float().numpy().reshape(-1, n_kv, d_head)
                for h in range(n_kv):
                    pre_capture[(li, h)] = k[:, h, :]
        return fn

    for l in range(n_layers):
        cal_hooks.append(layers[l].self_attn.register_forward_pre_hook(make_cal_hook(l), with_kwargs=True))

    with torch.no_grad():
        model(cal_ids, use_cache=False)

    for hook in cal_hooks:
        hook.remove()

    # Compute PCA bases
    pca_bases = {}
    pca_means = {}
    eigenvalues = {}
    for (l, h), K in pre_capture.items():
        mean = K.mean(0)
        Kc = K - mean
        cov = Kc.T @ Kc / K.shape[0]
        eigvals, eigvecs = np.linalg.eigh(cov)
        pca_bases[(l, h)] = eigvecs
        pca_means[(l, h)] = mean
        eigenvalues[(l, h)] = np.maximum(eigvals, 1e-10)

    np.random.seed(getattr(model, "_downstream_seed", 42))
    R_random = np.linalg.qr(np.random.randn(d_head, d_head))[0]

    # Choose rotation and WF
    pca = None
    rand_rot = None
    wf_alloc = None

    if method in ('pre_pca_uni', 'pre_pca_wf2', 'fokvq_e2', 'fokvq_e2_residual'):
        pca = pca_bases
    elif method in ('turbo', 'turboquant_rand'):
        rand_rot = R_random

    if method == 'pre_pca_wf2':
        wf_alloc = {k: compute_wf_alloc(v, bits, min_bits=2) for k, v in eigenvalues.items()}

    # Install hooks
    hook_handles = []
    for l in range(n_layers):
        k_proj = layers[l].self_attn.k_proj
        hook_fn = make_k_quant_hook(
            l, bits, n_kv, d_head,
            pca_bases=pca, pca_means=pca_means if pca is not None else None,
            random_rot=rand_rot, wf_alloc=wf_alloc
        )
        hook_fn.method = method
        hk = k_proj.register_forward_hook(hook_fn)
        hook_handles.append(hk)

    print(
        "  Installed "
        f"{len(hook_handles)} hooks for method={method_spec['requested_name']} "
        f"(impl={method_spec['implementation_key']}), bits={bits}"
    )
    return hook_handles


def run_self_test() -> int:
    """Supported quantized tiny-Llama smoke plus unsupported GPT-2 failure."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    device = "cpu"
    print("Running downstream self-test...")

    method_spec = get_method_spec("fokvq_e2")
    if method_spec["implementation_key"] != "fokvq_e2_ref":
        raise RuntimeError("Expected fokvq_e2 to remain backed by the shared v3 quantizer implementation.")
    if method_spec["official_reproduction"]:
        raise RuntimeError("Expected fokvq_e2 provenance to remain non-official.")
    print("  [PASS] provenance metadata keeps fokvq_e2 as a same-harness shared implementation")
    run_reference_parity_smoke()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device).eval()
    model._downstream_seed = 42
    model._downstream_cache_dir = None

    hooks = calibrate_and_install_hooks(model, "pre_pca_wf2", bits=2, device=device)
    try:
        batch = tokenizer("KV cache downstream self-test.", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**batch, use_cache=False)
        if not torch.isfinite(outputs.logits).all():
            raise RuntimeError("Non-finite logits produced by downstream self-test.")
        print(f"  [PASS] finite quantized logits on tiny Llama: shape={tuple(outputs.logits.shape)}")
    finally:
        for hk in hooks:
            hk.remove()
        del model
        gc.collect()

    run_unsupported_architecture_smoke()
    print("Downstream self-test passed.")
    return 0


def run_unsupported_architecture_smoke() -> int:
    """Verify unsupported GPT-2-like architectures fail with a clear error."""
    from transformers import AutoModelForCausalLM

    model_name = "sshleifer/tiny-gpt2"
    device = "cpu"
    print("Running unsupported-architecture smoke...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device).eval()
    try:
        try:
            calibrate_and_install_hooks(model, "pre_pca_uni", bits=2, device=device)
        except ValueError as exc:
            msg = str(exc)
            if "currently supports decoder models" not in msg and "Unsupported attention block" not in msg:
                raise RuntimeError(f"Unexpected error message for unsupported architecture: {msg}") from exc
            print(f"  [PASS] clean unsupported-architecture failure: {msg}")
            return 0
        raise RuntimeError("Unsupported architecture smoke did not fail as expected.")
    finally:
        del model
        gc.collect()


def run_reference_parity_smoke() -> int:
    """Check overlapping methods agree with v3 reference quantizers on a toy head."""
    ref = load_reference_quantizers()
    rng = np.random.RandomState(42)
    Kh = rng.randn(64, 32).astype(np.float32)
    methods = {
        "no_rot": lambda x: ref.identity_basis_quantize_head(torch.from_numpy(x), 2).cpu().numpy(),
        "pre_pca_uni": lambda x: ref.fokvq_quantize_head(torch.from_numpy(x), 2, gamma=0.3)[0].cpu().numpy(),
        "fokvq_e2": lambda x: ref.fokvq_e2_quantize_head(torch.from_numpy(x), 2, gamma=0.3)[0].cpu().numpy(),
        "fokvq_e2_residual": lambda x: ref.fokvq_e2_residual_quantize_head(
            torch.from_numpy(x), 2, gamma=0.3, residual_length=32
        )[0].cpu().numpy(),
        "kivi_residual": lambda x: ref.kivi_residual_quantize_head(
            torch.from_numpy(x), 2, group_size=32, residual_length=32
        ).cpu().numpy(),
        "turboquant_rand": lambda x: ref.turbo_quantize_random_head(torch.from_numpy(x), 2).cpu().numpy(),
    }
    for method, ref_fn in methods.items():
        got = quantize_head_with_method(
            Kh, method, bits=2, layer_idx=0, head_idx=0,
            pca_bases=None, pca_means=None, random_rot=None, wf_alloc=None
        )
        want = ref_fn(Kh)
        max_abs = float(np.max(np.abs(got - want)))
        if max_abs > 1e-5:
            raise RuntimeError(f"Reference parity smoke failed for {method}: max_abs={max_abs}")
        print(f"  [PASS] reference parity: {method} max_abs={max_abs:.3e}")
    return 0


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()
    if args.self_test:
        raise SystemExit(run_self_test())

    import lm_eval
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(args.seed)
    method_spec = get_method_spec(args.method)
    dtype = resolve_dtype(args.model, args.dtype)
    cache_dir = args.cache_dir or None

    print(f"\n{'#'*70}")
    print(f"# Downstream Eval: {args.model}")
    print(f"# Method: {args.method}, Bits: {args.bits}")
    print(f"# Tasks: {args.tasks}")
    print(f"{'#'*70}")
    print(f"  Runner: {RUNNER_NAME} ({RUNNER_VERSION})")
    print(f"  Scope: {HARNESS_SCOPE}")
    print(
        "  Provenance: "
        f"{method_spec['claim_label']} | {method_spec['method_provenance']} | "
        f"impl={method_spec['implementation_key']}"
    )
    if method_spec["support_level"] == "supported_alias":
        print(f"  Alias note: requested {args.method} maps to runner-local impl {method_spec['implementation_key']}")
    print("  Over-claim guard: not an official KIVI/TurboQuant/full-v3 reproduction.")

    t0 = time.time()

    # Load model
    load_kwargs = {}
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        **load_kwargs,
    ).to(args.device).eval()
    model._downstream_seed = args.seed
    model._downstream_cache_dir = cache_dir

    stack = resolve_attention_stack(model)
    print(
        "  Harness scope: "
        f"model_type={getattr(model.config, 'model_type', 'unknown')}, "
        f"layers={stack['n_layers']}, kv_heads={stack['n_kv']}, d_head={stack['d_head']}, "
        f"dtype={dtype}, attn_impl={args.attn_implementation}, seed={args.seed}"
    )

    # Install hooks
    hook_handles = calibrate_and_install_hooks(model, args.method, args.bits, args.device)

    # Run lm-eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
    )

    # Remove hooks
    for hk in hook_handles:
        hk.remove()

    # Print results
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Results: {args.model} | {args.method} {args.bits}bit")
    print(f"{'='*70}")

    output = build_output_metadata(args, model, tokenizer, stack, lm_eval.__version__)
    output.update({
        'model': args.model,
        'method': args.method,
        'bits': args.bits,
        'harness_scope': HARNESS_SCOPE,
        'tasks': {},
        'elapsed': elapsed,
    })

    for task_name, task_results in results['results'].items():
        metric_name = 'acc,none' if 'acc,none' in task_results else ('acc_norm,none' if 'acc_norm,none' in task_results else None)
        stderr_name = 'acc_stderr,none' if 'acc_stderr,none' in task_results else ('acc_norm_stderr,none' if 'acc_norm_stderr,none' in task_results else None)
        acc = task_results.get(metric_name, None) if metric_name else None
        acc_stderr = task_results.get(stderr_name, None) if stderr_name else None
        print(f"  {task_name}: {metric_name}={acc}")
        output['tasks'][task_name] = {
            'acc': acc,
            'acc_stderr': acc_stderr,
            'primary_metric_name': metric_name,
            'stderr_metric_name': stderr_name,
            'all_metrics': {k: v for k, v in task_results.items() if isinstance(v, (int, float))},
        }

    # Save
    OUTDIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.replace("/", "_")
    task_tag = "_".join(args.tasks)
    outfile = OUTDIR / f"downstream_{model_tag}_{args.method}_{args.bits}b_{task_tag}_{timestamp}.json"
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved: {outfile}")
    print(f"  Time: {elapsed:.1f}s")

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    main()
