#!/usr/bin/env python3
"""
Hook-mode WikiText-2 PPL evaluator for OCQ / KIVI / KVSink (WIP) / uniform.

Replaces the legacy ``scripts/fokvq/exp4_2_standard_ppl_benchmark.py``
``evaluate_quantized_sliding_window`` path, which had two defects:

  Bug 1 (known 2026-04-09 AM): ``stride == context_len`` fell through
    to an fp16 path because ``prefix_len = 0`` when every window is a
    non-overlap chunk. KIVI's published protocol uses exactly that
    setting, so no quantization was applied in a KIVI-comparable run.

  Bug 2 (discovered 2026-04-09 PM): the legacy driver quantized
    ``past_key_values`` which for Llama/Qwen HF models holds
    **post-RoPE** K tensors, while ``build_qwen_metatool_b_ont.py``
    collects ``k_proj`` output which is **pre-RoPE**. Applying a
    pre-RoPE basis to post-RoPE K is a space mismatch and the
    resulting PPL numbers are not interpretable.

This driver fixes both by registering a forward hook on every
``layer.self_attn.k_proj`` module that rewrites the K output tensor
**before** ``apply_rotary_pos_emb`` runs. Every K used in attention
(self-attention within the window AND cross-chunk via ``past_key_values``)
is therefore the quantized-then-dequantized tensor.

Eval protocol
-------------
KIVI-style non-overlapping WT2 PPL:
  * concat ``"\n\n".join(test_texts)`` once
  * tokenize once; split into windows of ``context_len`` tokens
    (last ragged window is dropped if ``--drop-last``; default keeps it)
  * for each window: fresh ``model(window, use_cache=False)`` with the
    quant hook active → compute NLL over all but the first token
  * sum NLL / sum tokens → PPL

Supported methods
-----------------
  fp16       — no-op quantization (upper baseline)
  uniform    — per-head per-token symmetric uniform on pre-RoPE K
  kivi       — per-channel asymmetric quant on pre-RoPE K (reproduces
               KIVI's "key is quantized per-channel" design; residual
               length ``R`` = last-N-tokens-fp16 controlled by ``--kivi-R``)
  ocq        — Ontology-Categorical Quantization on pre-RoPE K, loading
               per-(layer, head) B_ont from a ``.pt`` produced by
               ``scripts/ocq/build_qwen_metatool_b_ont.py``

Residual modes (ocq only): 2a or 2b (see quantizer.py).
Ontology modes (ocq only): 1a, 1b, 1c (see quantizer.py).

CLI example
-----------
    source /home/woori/workspace_common/CDP/poc/set.env && \\
    python scripts/ocq/eval_hook_mode.py \\
        --model Qwen/Qwen2.5-7B \\
        --device cuda:0 \\
        --context-len 2048 \\
        --max-windows 16 \\
        --methods fp16 uniform kivi ocq \\
        --ocq-b-ont external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt \\
        --ocq-ont-mode 1b --ocq-res-mode 2a --ocq-res-bits 2 \\
        --bits 2 3 4 \\
        --out /tmp/ocq_hook_mode_smoke.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from quantizer import build_ocq_basis, ocq_quantize  # noqa: E402


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--context-len", type=int, default=2048,
                   help="Window size in tokens. KIVI protocol = 2048.")
    p.add_argument("--max-windows", type=int, default=0,
                   help="Hard cap on number of windows (0 = use all).")
    p.add_argument("--drop-last", action="store_true",
                   help="Drop final ragged window instead of padding it.")
    p.add_argument("--methods", nargs="+",
                   default=["fp16", "uniform", "kivi", "ocq"])
    p.add_argument("--bits", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--kivi-R", type=int, default=128,
                   help="KIVI residual length (last R tokens fp16).")
    # OCQ options
    p.add_argument("--ocq-b-ont", type=str, default="",
                   help="Path to B_ont .pt file (required for ocq method).")
    p.add_argument("--ocq-ont-mode", default="1b", choices=["1a", "1b", "1c"])
    p.add_argument("--ocq-res-mode", default="2a", choices=["2a", "2b"])
    p.add_argument("--ocq-res-bits", type=int, default=0,
                   help="Residual axis bits. 0 = follow --bits sweep.")
    p.add_argument("--ocq-r-ont", type=int, default=0,
                   help="Override r_ont. 0 = use B_ont tensor's r_ont.")
    # Data
    p.add_argument("--dataset", default="wikitext-2-raw-v1")
    p.add_argument("--cache-dir", default="")
    # Output
    p.add_argument("--out", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    # Smoke / self-test
    p.add_argument("--self-test", action="store_true",
                   help="Run a synthetic self-test (no HF model load).")
    return p.parse_args()


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_wikitext_test(cache_dir: Optional[str]) -> str:
    ds = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="test",
        cache_dir=cache_dir or None,
    )
    texts = [r["text"] for r in ds if r["text"].strip()]
    return "\n\n".join(texts)


def tokenize_and_chunk(
    text: str, tokenizer, context_len: int, drop_last: bool,
) -> List[torch.Tensor]:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]  # (T,)
    total = ids.shape[0]
    n_full = total // context_len
    windows = [ids[i * context_len : (i + 1) * context_len]
               for i in range(n_full)]
    tail = ids[n_full * context_len:]
    if tail.shape[0] > 1 and not drop_last:
        windows.append(tail)
    return windows


# ---------------------------------------------------------------------
# Quant primitives (pre-RoPE K)
# ---------------------------------------------------------------------
#
# Hook input/output shape from k_proj (Llama/Qwen style, GQA):
#   input:  (B, T, hidden_size)
#   output: (B, T, n_kv_heads * head_dim)   — pre-RoPE, BEFORE reshape
#
# We reshape to (B, T, n_kv, head_dim) inside the hook, quantize in
# (B, n_kv, T, head_dim) (to match the B,H,S,d convention used by the
# rest of the code and by KIVI's per-channel description), then reshape
# back to the flattened (B, T, n_kv*head_dim) layout that the caller
# expects to apply apply_rotary_pos_emb on.

def _symmetric_quant_last(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-token symmetric uniform on the last dim."""
    if bits <= 0:
        return torch.zeros_like(tensor)
    levels = 2 ** bits
    qmin = -levels // 2
    qmax = (levels + 1) // 2 - 1
    scale = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / max(1, -qmin)
    q = (tensor / scale).round().clamp(qmin, qmax)
    return q * scale


def _asymmetric_per_channel(tensor: torch.Tensor, bits: int,
                            residual_R: int = 0) -> torch.Tensor:
    """KIVI-style per-channel asymmetric quant on K.

    tensor shape: (B, H, T, d). KIVI's key quant reduces over the T axis
    per channel (d axis). With ``residual_R > 0``, the last R tokens are
    kept at the original dtype (fp16/bf16) — the KIVI residual length.
    """
    if bits <= 0:
        return torch.zeros_like(tensor)
    B, H, T, d = tensor.shape
    if residual_R >= T:
        return tensor  # entire window is residual
    bulk = tensor[:, :, : T - residual_R, :] if residual_R > 0 else tensor
    tail = tensor[:, :, T - residual_R:, :] if residual_R > 0 else None
    # Per-channel stats over the bulk tokens
    mn = bulk.amin(dim=2, keepdim=True)  # (B, H, 1, d)
    mx = bulk.amax(dim=2, keepdim=True)
    levels = 2 ** bits
    scale = (mx - mn).clamp(min=1e-8) / (levels - 1)
    q = ((bulk - mn) / scale).round().clamp(0, levels - 1)
    bulk_q = q * scale + mn
    if tail is not None:
        return torch.cat([bulk_q, tail], dim=2)
    return bulk_q


def _ocq_quant(
    tensor: torch.Tensor,              # (B, H, T, d)
    B_full_per_head: torch.Tensor,     # (H, d, d) — first r_ont cols = B_ont
    r_ont: int,
    ont_mode: str,
    res_bits: int,
) -> torch.Tensor:
    return ocq_quantize(tensor, B_full_per_head, r_ont=r_ont,
                        ont_mode=ont_mode, res_bits=res_bits)


# ---------------------------------------------------------------------
# Quant callbacks and hook installer
# ---------------------------------------------------------------------

QuantFn = Callable[[torch.Tensor, int], torch.Tensor]
# (key_tensor_BHTd, layer_idx) -> quantized key_tensor_BHTd


def build_quant_fn(
    method: str,
    bits: int,
    kivi_R: int,
    # OCQ-specific
    ocq_bases: Optional[torch.Tensor],        # (L, H, d, d) B_full per layer per head
    ocq_r_ont: int,
    ocq_ont_mode: str,
    ocq_res_bits: int,
) -> QuantFn:
    if method == "fp16":
        def f(k, layer_idx):
            return k
        return f
    if method == "uniform":
        def f(k, layer_idx):
            return _symmetric_quant_last(k, bits)
        return f
    if method == "kivi":
        def f(k, layer_idx):
            return _asymmetric_per_channel(k, bits, residual_R=kivi_R)
        return f
    if method == "ocq":
        if ocq_bases is None:
            raise ValueError("ocq method requires --ocq-b-ont")
        effective_res_bits = ocq_res_bits if ocq_res_bits > 0 else bits
        r_ont = ocq_r_ont
        ont_mode = ocq_ont_mode
        def f(k, layer_idx):
            B_full = ocq_bases[layer_idx].to(device=k.device, dtype=k.dtype)
            return _ocq_quant(
                k, B_full, r_ont=r_ont, ont_mode=ont_mode,
                res_bits=effective_res_bits,
            )
        return f
    raise ValueError(f"unknown method: {method}")


@contextmanager
def install_k_proj_hooks(model, quant_fn: QuantFn, n_kv: int, head_dim: int):
    """Register forward hooks on every layer's k_proj.

    Hook rewrites k_proj output in-place. Shape handling:
      raw k_proj output: (B, T, n_kv * head_dim)
      reshape to: (B, T, n_kv, head_dim) -> permute (B, n_kv, T, head_dim)
      quantize
      permute + reshape back to (B, T, n_kv * head_dim)
    """
    handles = []
    try:
        for layer_idx, layer in enumerate(model.model.layers):
            k_proj = layer.self_attn.k_proj

            def make_hook(li):
                def hook(module, inputs, output):
                    # output: (B, T, n_kv*head_dim)  OR  (B, T, hidden_size)
                    # For GQA, n_kv*head_dim != hidden_size.
                    B, T, D = output.shape
                    if D != n_kv * head_dim:
                        # Unexpected layout; give up quietly.
                        return output
                    k = output.view(B, T, n_kv, head_dim).permute(0, 2, 1, 3).contiguous()
                    # k: (B, n_kv, T, head_dim), pre-RoPE
                    orig_dtype = k.dtype
                    k_q = quant_fn(k.float(), li).to(dtype=orig_dtype)
                    out = k_q.permute(0, 2, 1, 3).contiguous().view(B, T, D)
                    return out
                return hook

            handles.append(k_proj.register_forward_hook(make_hook(layer_idx)))
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_ppl(
    model,
    windows: List[torch.Tensor],
    device: str,
) -> Dict[str, float]:
    total_nll = 0.0
    total_tok = 0
    t0 = time.time()
    for win in windows:
        ids = win.unsqueeze(0).to(device)
        out = model(ids, use_cache=False)
        logits = out.logits[:, :-1, :]  # (1, T-1, V)
        labels = ids[:, 1:]             # (1, T-1)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="sum",
        )
        n = int(labels.numel())
        total_nll += float(loss.item())
        total_tok += n
    return {
        "ppl": math.exp(total_nll / max(total_tok, 1)),
        "total_nll": total_nll,
        "total_tokens": total_tok,
        "runtime_s": time.time() - t0,
    }


# ---------------------------------------------------------------------
# OCQ basis construction (build per-layer per-head B_full from saved B_ont)
# ---------------------------------------------------------------------
#
# The saved B_ont.pt has shape (L, H, d, r_ont) and stores ONLY the
# ontology basis, not the residual basis. We construct B_full at load
# time by combining with the per-(layer, head) pre-RoPE Σ_K computed
# from a short calibration pass.
#
# Calibration: run a handful of wikitext training sentences through
# the model with a forward hook that collects k_proj output, accumulate
# per-(layer, head) Σ_K over all content tokens.

@torch.no_grad()
def build_ocq_full_bases(
    model,
    tokenizer,
    B_ont: torch.Tensor,               # (L, H, d, r_ont)
    res_mode: str,                     # "2a" or "2b"
    device: str,
    n_calib_samples: int = 32,
    calib_max_len: int = 512,
) -> torch.Tensor:                     # (L, H, d, d)
    L, H, d, r_ont = B_ont.shape
    # Accumulate Σ_K per (layer, head)
    sigma_sum = np.zeros((L, H, d, d), dtype=np.float64)
    sigma_cnt = np.zeros((L, H), dtype=np.int64)
    buf: List[Optional[torch.Tensor]] = [None] * L

    def make_hook(li):
        def hook(module, inputs, output):
            buf[li] = output.detach()  # (B, T, H*d) pre-RoPE
        return hook

    handles = [
        model.model.layers[li].self_attn.k_proj.register_forward_hook(make_hook(li))
        for li in range(L)
    ]
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                          split="train")
        collected = 0
        for row in ds:
            txt = row["text"].strip()
            if not txt or len(txt) < 40:
                continue
            enc = tokenizer(
                txt, return_tensors="pt",
                truncation=True, max_length=calib_max_len,
                add_special_tokens=False,
            )
            ids = enc["input_ids"].to(device)
            if ids.shape[1] < 4:
                continue
            model(ids, use_cache=False)
            for li in range(L):
                k = buf[li]
                if k is None:
                    continue
                # (1, T, H*d) -> (T, H, d) -> (H, T, d)
                T = k.shape[1]
                k_htd = k.view(1, T, H, d)[0].permute(1, 0, 2).float()  # (H, T, d)
                # skip BOS-like first token for calib stability
                k_htd = k_htd[:, 1:, :] if T > 1 else k_htd
                for h in range(H):
                    kh = k_htd[h].cpu().numpy().astype(np.float64)  # (T-1, d)
                    sigma_sum[li, h] += kh.T @ kh
                    sigma_cnt[li, h] += kh.shape[0]
            collected += 1
            if collected >= n_calib_samples:
                break
    finally:
        for h in handles:
            h.remove()

    # Build full bases
    B_full = np.zeros((L, H, d, d), dtype=np.float32)
    for li in range(L):
        for h in range(H):
            cnt = max(int(sigma_cnt[li, h]), 1)
            sig = sigma_sum[li, h] / cnt
            B_ont_lh = B_ont[li, h].cpu().numpy().astype(np.float64)  # (d, r_ont)
            B_full_lh = build_ocq_basis(sig, B_ont_lh, res_mode=res_mode)
            # If partial-rank, pad to d with zeros on the right
            k_cols = B_full_lh.shape[1]
            if k_cols < d:
                pad = np.zeros((d, d - k_cols), dtype=B_full_lh.dtype)
                B_full_lh = np.concatenate([B_full_lh, pad], axis=1)
            B_full[li, h] = B_full_lh.astype(np.float32)
    return torch.from_numpy(B_full)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    # auto: bf16 on Ampere+, otherwise fp16
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _self_test() -> None:
    """Synthetic hook-mode smoke test using a tiny nn module stack."""
    torch.manual_seed(0)
    # Fake a 2-layer model.model.layers[].self_attn.k_proj interface
    d_head = 8
    n_kv = 2
    hidden = n_kv * d_head
    L = 2
    T = 16

    class FakeAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.k_proj = torch.nn.Linear(hidden, hidden, bias=False)

    class FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = FakeAttn()

    class FakeStack(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([FakeLayer() for _ in range(L)])

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeStack()

    model = FakeModel().eval()
    x = torch.randn(1, T, hidden)

    # Reference K outputs pre-hook
    ref_out = []
    for li in range(L):
        ref_out.append(model.model.layers[li].self_attn.k_proj(x).detach().clone())

    # uniform 2-bit hook test
    def quant_fn(k, li):
        return _symmetric_quant_last(k, 2)

    with install_k_proj_hooks(model, quant_fn, n_kv=n_kv, head_dim=d_head):
        hooked_out = []
        for li in range(L):
            hooked_out.append(model.model.layers[li].self_attn.k_proj(x).detach().clone())

    for li in range(L):
        ref = ref_out[li]
        hk = hooked_out[li]
        # Shapes match
        assert ref.shape == hk.shape, f"shape mismatch layer {li}"
        # Hooked is different from ref (quantized) but same reconstruction magnitude
        mse = float((ref - hk).pow(2).mean())
        assert mse > 1e-8, f"hook had no effect at layer {li}"
        assert mse < float((ref ** 2).mean()), (
            f"quantization destroyed signal at layer {li}"
        )
    print("eval_hook_mode self-test passed.")


def run() -> None:
    args = parse_args()
    if args.self_test:
        _self_test()
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dtype = resolve_dtype(args.dtype)
    print(f"[load] model={args.model} dtype={dtype} device={args.device}",
          flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[load] took {time.time()-t0:.1f}s", flush=True)

    cfg = model.config
    n_kv = cfg.num_key_value_heads
    n_q = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_q)
    L = cfg.num_hidden_layers
    print(f"[cfg] L={L} n_kv={n_kv} n_q={n_q} head_dim={head_dim}", flush=True)

    # Data
    print(f"[data] loading wikitext test split", flush=True)
    text = load_wikitext_test(args.cache_dir or None)
    windows = tokenize_and_chunk(text, tok, args.context_len, args.drop_last)
    if args.max_windows > 0:
        windows = windows[: args.max_windows]
    n_win = len(windows)
    total_tokens = sum(w.shape[0] for w in windows)
    print(f"[data] {n_win} windows, {total_tokens} total tokens, "
          f"ctx={args.context_len}", flush=True)

    # Optional OCQ bases
    ocq_bases_tensor: Optional[torch.Tensor] = None
    if "ocq" in args.methods:
        if not args.ocq_b_ont:
            raise ValueError("--ocq-b-ont is required when 'ocq' is in --methods")
        print(f"[ocq] loading B_ont from {args.ocq_b_ont}", flush=True)
        b_ont_payload = torch.load(args.ocq_b_ont, map_location="cpu",
                                    weights_only=False)
        if isinstance(b_ont_payload, dict) and "B_ont" in b_ont_payload:
            B_ont = b_ont_payload["B_ont"]
        else:
            B_ont = b_ont_payload
        if B_ont.ndim != 4:
            raise ValueError(f"B_ont must have shape (L,H,d,r), got {B_ont.shape}")
        if B_ont.shape[:2] != (L, n_kv):
            raise ValueError(
                f"B_ont (L,H)=({B_ont.shape[0]},{B_ont.shape[1]}) "
                f"does not match model (L,n_kv)=({L},{n_kv})"
            )
        if B_ont.shape[2] != head_dim:
            raise ValueError(
                f"B_ont head_dim={B_ont.shape[2]} != model head_dim={head_dim}"
            )
        r_ont = args.ocq_r_ont or B_ont.shape[3]
        print(f"[ocq] B_ont shape {tuple(B_ont.shape)}, r_ont={r_ont}",
              flush=True)
        print(f"[ocq] calibrating Σ_K for residual basis (res_mode={args.ocq_res_mode})",
              flush=True)
        t0 = time.time()
        ocq_bases_tensor = build_ocq_full_bases(
            model, tok, B_ont.float(), res_mode=args.ocq_res_mode,
            device=args.device,
        )
        print(f"[ocq] B_full built in {time.time()-t0:.1f}s, "
              f"shape {tuple(ocq_bases_tensor.shape)}", flush=True)

    # Sweep
    results = []
    for method in args.methods:
        if method == "fp16":
            bit_list = [0]  # single run
        else:
            bit_list = list(args.bits)
        for bits in bit_list:
            tag = f"{method}_{bits}" if method != "fp16" else "fp16"
            print(f"\n[eval] {tag}", flush=True)
            quant_fn = build_quant_fn(
                method, bits, args.kivi_R,
                ocq_bases=ocq_bases_tensor,
                ocq_r_ont=args.ocq_r_ont or (
                    ocq_bases_tensor.shape[-1] - (
                        ocq_bases_tensor.shape[-1] - B_ont.shape[3]
                    ) if "ocq" in args.methods else 0
                ),
                ocq_ont_mode=args.ocq_ont_mode,
                ocq_res_bits=args.ocq_res_bits,
            )
            with install_k_proj_hooks(model, quant_fn, n_kv=n_kv,
                                       head_dim=head_dim):
                out = evaluate_ppl(model, windows, args.device)
            out["method"] = method
            out["bits"] = bits
            out["tag"] = tag
            print(f"[eval] {tag}: PPL={out['ppl']:.4f} "
                  f"nll={out['total_nll']:.2f} tok={out['total_tokens']} "
                  f"t={out['runtime_s']:.1f}s", flush=True)
            results.append(out)

    payload = {
        "model": args.model,
        "context_len": args.context_len,
        "n_windows": n_win,
        "total_tokens": total_tokens,
        "kivi_R": args.kivi_R,
        "ocq": {
            "b_ont_path": args.ocq_b_ont,
            "ont_mode": args.ocq_ont_mode,
            "res_mode": args.ocq_res_mode,
            "res_bits": args.ocq_res_bits,
            "r_ont_override": args.ocq_r_ont,
        } if "ocq" in args.methods else None,
        "results": results,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}", flush=True)
    else:
        print("\n" + json.dumps(payload, indent=2))


if __name__ == "__main__":
    run()
