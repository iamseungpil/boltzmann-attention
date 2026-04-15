#!/usr/bin/env python3
"""measure_ocq_entropy_stack.py — predict OCQ + entropy-coding compression ratio.

Stacks DEFLATE / LZMA2 entropy coding (à la KVTC) on top of OCQ quantized
indices and reports the additional compression. Two modes:

  --mode synthetic: generate synthetic K matching (H-cat) — fast (~30 s).
  --mode real:      extract K from a Qwen2.5-7B forward pass on a small
                    WT2 calibration set, more representative (~5 min).

Outputs:
  reports/ocq_entropy_stack/result.json with per-method compression ratios.

Usage:
  python scripts/ocq/measure_ocq_entropy_stack.py --mode synthetic --n-tokens 4096
  python scripts/ocq/measure_ocq_entropy_stack.py --mode real --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
import lzma
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--n-tokens", type=int, default=4096,
                   help="Number of K tokens for the calibration set.")
    p.add_argument("--n-heads", type=int, default=4,
                   help="Number of KV heads (4 for Qwen2.5-7B).")
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--r-ont", type=int, default=24,
                   help="Ontology rank (24 for Qwen).")
    p.add_argument("--res-bits", type=int, default=2,
                   help="Bits per residual channel (2 for OCQ 1b+2a).")
    p.add_argument("--facet-separation", type=float, default=5.0,
                   help="(H-cat) bimodal separation parameter s.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="reports/ocq_entropy_stack/result.json")
    return p.parse_args()


def synthesize_k(args) -> torch.Tensor:
    """Generate K respecting (H-cat) bimodal-channel structure.
    Returns (n_heads, n_tokens, head_dim)."""
    torch.manual_seed(args.seed)
    H, S, d, r = args.n_heads, args.n_tokens, args.head_dim, args.r_ont
    s = args.facet_separation
    # Facet channels: bimodal with separation s + small Gaussian noise
    facet_signs = torch.bernoulli(torch.full((H, S, r), 0.5)) * 2 - 1
    facet_coords = facet_signs * s + torch.randn(H, S, r) * 1.0
    # Build a B_ont per head (orthonormal)
    Bs = torch.zeros(H, d, r)
    for h in range(H):
        rand = torch.randn(d, r)
        q, _ = torch.linalg.qr(rand)
        Bs[h] = q[:, :r]
    # Reconstruct K = B_h @ facet_coords + residual (Gaussian)
    K = torch.zeros(H, S, d)
    for h in range(H):
        K[h] = facet_coords[h] @ Bs[h].t() + torch.randn(S, d) * 1.5
    # Add temporal smoothness (real K has token-to-token correlation)
    # Apply small AR(1) smoothing to mimic adjacent-token similarity
    rho = 0.3
    for s_idx in range(1, S):
        K[:, s_idx, :] = rho * K[:, s_idx - 1, :] + (1 - rho) * K[:, s_idx, :]
    return K, Bs


def extract_real_k(args):
    """Run Qwen2.5-7B forward on small WT2 calibration; return K from
    a representative middle layer (L=13)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"[real] loading {model_id}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="eager", low_cpu_mem_usage=True,
    )
    model.eval()
    # Load B_ont for layer 13
    B_path = "external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt"
    bdict = torch.load(B_path, map_location="cpu", weights_only=False)
    B_full = bdict["B_ont"][13].to(torch.float32)  # (H, d, r)
    print(f"[real] B_ont layer 13: {B_full.shape}", flush=True)

    # Hook to capture K at layer 13 k_proj output
    captured = {}
    def hook(_m, _inp, out):
        captured["K"] = out.detach().cpu().float()
    handle = model.model.layers[13].self_attn.k_proj.register_forward_hook(hook)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"][:50])  # ~few KB
    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=args.n_tokens).to(args.device)
    with torch.no_grad():
        model(**ids)
    handle.remove()

    K_flat = captured["K"]  # (1, S, n_kv*d)
    n_kv = model.config.num_key_value_heads
    d = model.config.hidden_size // model.config.num_attention_heads
    S = K_flat.shape[1]
    K = K_flat[0].view(S, n_kv, d).permute(1, 0, 2).contiguous()
    print(f"[real] K shape: {K.shape}", flush=True)
    return K, B_full


def quantize_ocq(K: torch.Tensor, B: torch.Tensor, r_ont: int,
                 res_bits: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply OCQ 1b + R-bit asymmetric residual.
    Returns (ont_indices: 1-bit, res_indices: res_bits, meta).
    ont_indices: (H, S, r_ont) uint8 (0 or 1)
    res_indices: (H, S, d) uint8 in [0, 2^res_bits)
    """
    H, S, d = K.shape
    K = K.float()
    # Project to ontology coefficients: K @ B_h gives (S, r_ont) per head
    ont_coords = torch.einsum("hsd,hdr->hsr", K, B)  # (H, S, r)
    # Reconstruct ontology component
    K_ont = torch.einsum("hsr,hdr->hsd", ont_coords, B)
    K_res = K - K_ont
    # 1-bit categorical: per-channel mean split (1b mean-split)
    means = ont_coords.mean(dim=1, keepdim=True)  # (H, 1, r)
    ont_idx = (ont_coords > means).to(torch.uint8)
    # res-bit asymmetric: per-channel quantile-based bins
    n_levels = 1 << res_bits
    res_idx = torch.zeros_like(K_res, dtype=torch.uint8)
    for h in range(H):
        for c in range(d):
            x = K_res[h, :, c]
            qs = torch.linspace(1.0 / n_levels, 1.0 - 1.0 / n_levels,
                                n_levels - 1)
            thresholds = torch.quantile(x, qs)
            idx = torch.zeros_like(x, dtype=torch.uint8)
            for level in range(1, n_levels):
                idx[x >= thresholds[level - 1]] = level
            res_idx[h, :, c] = idx
    return ont_idx.numpy(), res_idx.numpy(), {
        "H": H, "S": S, "d": d, "r_ont": r_ont, "res_bits": res_bits,
    }


def pack_bits_to_bytes(ont_idx: np.ndarray, res_idx: np.ndarray,
                       res_bits: int) -> bytes:
    """Tightly pack 1-bit ontology indices + R-bit residual indices."""
    # 1-bit ontology
    ont_flat = ont_idx.reshape(-1).astype(np.uint8)
    ont_bytes = np.packbits(ont_flat).tobytes()
    # R-bit residual: pack (8 // R) values per byte
    res_flat = res_idx.reshape(-1).astype(np.uint8)
    if res_bits == 1:
        return ont_bytes + np.packbits(res_flat).tobytes()
    elif res_bits == 2:
        n_pad = (-len(res_flat)) % 4
        res_padded = np.concatenate([res_flat,
                                     np.zeros(n_pad, dtype=np.uint8)])
        packed = (res_padded[0::4]
                  | (res_padded[1::4] << 2)
                  | (res_padded[2::4] << 4)
                  | (res_padded[3::4] << 6))
        return ont_bytes + packed.tobytes()
    elif res_bits == 4:
        n_pad = (-len(res_flat)) % 2
        res_padded = np.concatenate([res_flat,
                                     np.zeros(n_pad, dtype=np.uint8)])
        packed = res_padded[0::2] | (res_padded[1::2] << 4)
        return ont_bytes + packed.tobytes()
    elif res_bits == 8:
        return ont_bytes + res_flat.tobytes()
    else:
        raise ValueError(f"unsupported res_bits={res_bits}")


def main():
    args = parse_args()
    torch.set_num_threads(8)
    t0 = time.time()

    if args.mode == "synthetic":
        print("[mode] synthetic K with (H-cat) structure", flush=True)
        K, B = synthesize_k(args)
    else:
        print("[mode] real K from Qwen2.5-7B WT2 calibration", flush=True)
        K, B = extract_real_k(args)

    print(f"[load] K {tuple(K.shape)} dtype={K.dtype} elapsed={time.time()-t0:.1f}s",
          flush=True)

    # Apply OCQ
    t1 = time.time()
    ont_idx, res_idx, meta = quantize_ocq(K, B, args.r_ont, args.res_bits)
    print(f"[ocq] quantize done {time.time()-t1:.1f}s | "
          f"ont {ont_idx.shape}, res {res_idx.shape}", flush=True)

    # Pack to bytes
    raw_bytes = pack_bits_to_bytes(ont_idx, res_idx, args.res_bits)
    n_elems = meta["H"] * meta["S"] * meta["d"]
    fp16_bytes = n_elems * 2
    print(f"[pack] raw {len(raw_bytes):,} B | fp16 baseline {fp16_bytes:,} B",
          flush=True)

    # OCQ alone
    ocq_bits_per_elem = 8 * len(raw_bytes) / n_elems
    ocq_ratio = fp16_bytes / len(raw_bytes)

    # OCQ + DEFLATE (zlib level 9)
    t2 = time.time()
    deflate_bytes = zlib.compress(raw_bytes, level=9)
    print(f"[deflate] {len(deflate_bytes):,} B in {time.time()-t2:.1f}s",
          flush=True)
    deflate_bits_per_elem = 8 * len(deflate_bytes) / n_elems
    deflate_ratio = fp16_bytes / len(deflate_bytes)

    # OCQ + LZMA2
    t3 = time.time()
    lzma_bytes = lzma.compress(raw_bytes, preset=9 | lzma.PRESET_EXTREME)
    print(f"[lzma] {len(lzma_bytes):,} B in {time.time()-t3:.1f}s",
          flush=True)
    lzma_bits_per_elem = 8 * len(lzma_bytes) / n_elems
    lzma_ratio = fp16_bytes / len(lzma_bytes)

    # Per-channel entropy (Shannon lower bound for ontology + residual)
    def shannon_entropy(idx_array: np.ndarray, n_levels: int) -> float:
        flat = idx_array.reshape(-1).astype(np.int64)
        counts = np.bincount(flat, minlength=n_levels)
        p = counts.astype(np.float64) / counts.sum()
        nz = p > 0
        return float(-(p[nz] * np.log2(p[nz])).sum())

    H_ont = shannon_entropy(ont_idx, 2)
    H_res = shannon_entropy(res_idx, 1 << args.res_bits)
    n_ont = ont_idx.size
    n_res = res_idx.size
    shannon_bits_per_elem = (H_ont * n_ont + H_res * n_res) / n_elems
    shannon_ratio = 16.0 / shannon_bits_per_elem if shannon_bits_per_elem > 0 else float("inf")

    result = {
        "mode": args.mode,
        "n_heads": meta["H"], "n_tokens": meta["S"], "head_dim": meta["d"],
        "r_ont": meta["r_ont"], "res_bits": meta["res_bits"],
        "facet_separation": args.facet_separation if args.mode == "synthetic" else None,
        "n_elems": n_elems,
        "fp16_bytes": fp16_bytes,
        "ocq": {
            "bytes": len(raw_bytes),
            "bits_per_elem": ocq_bits_per_elem,
            "compression_ratio": ocq_ratio,
        },
        "ocq_plus_deflate": {
            "bytes": len(deflate_bytes),
            "bits_per_elem": deflate_bits_per_elem,
            "compression_ratio": deflate_ratio,
        },
        "ocq_plus_lzma2": {
            "bytes": len(lzma_bytes),
            "bits_per_elem": lzma_bits_per_elem,
            "compression_ratio": lzma_ratio,
        },
        "shannon_lower_bound": {
            "H_ontology_per_bit": H_ont,
            "H_residual_per_symbol": H_res,
            "bits_per_elem": shannon_bits_per_elem,
            "compression_ratio": shannon_ratio,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== SUMMARY ({args.mode}) ===", flush=True)
    print(f"  Method                         Bytes    Bits/elem   Ratio", flush=True)
    print(f"  fp16 baseline                  {fp16_bytes:>10,}    16.000     1.00×", flush=True)
    print(f"  OCQ alone                      {len(raw_bytes):>10,}     {ocq_bits_per_elem:.3f}     {ocq_ratio:.2f}×", flush=True)
    print(f"  OCQ + DEFLATE (zlib)           {len(deflate_bytes):>10,}     {deflate_bits_per_elem:.3f}     {deflate_ratio:.2f}×", flush=True)
    print(f"  OCQ + LZMA2 (preset 9 extreme) {len(lzma_bytes):>10,}     {lzma_bits_per_elem:.3f}     {lzma_ratio:.2f}×", flush=True)
    print(f"  Shannon lower bound            {'(N/A)':>10}     {shannon_bits_per_elem:.3f}     {shannon_ratio:.2f}×", flush=True)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
