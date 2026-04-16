#!/usr/bin/env python3
"""measure_thm618_attn_weighted_bits.py — Thm 6.18 empirical WT2 PPL.

Implements attention-weighted bit allocation:
  b*(t,f) = 0.5 * log2(lambda* * pi(t,f) * sigma_f^2)_+
  s.t. sum_{t,f} b*(t,f) <= B

where:
  pi(t,f) = E_q[attn(q, k_t) * g_f(k_t)]   (facet attention mass)
  sigma_f^2 = E_k[||B_f^T k||^2]            (per-facet variance)

Compared against:
  - KIVI uniform (2.00 bits)
  - OCQ 1b+2a uniform (1.81 bits)
  - fp16

Usage:
  python scripts/ocq/measure_thm618_attn_weighted_bits.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt \
    --calib-tokens 1024 \
    --eval-tokens 8192 \
    --target-avg-bits 1.81 2.0 2.5 3.0 4.0 \
    --out reports/thm618/ppl_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from eval_hook_mode import (
    install_k_proj_hooks, evaluate_ppl,
    tokenize_and_chunk, load_wikitext_test,
    _asymmetric_per_channel,
)

REPO = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--b-ont", required=True)
    p.add_argument("--target-layers", type=int, nargs="+",
                   default=list(range(28)),
                   help="Layers to apply K-quantization (default: all 28 Qwen layers).")
    p.add_argument("--calib-tokens", type=int, default=1024)
    p.add_argument("--eval-tokens", type=int, default=8192,
                   help="Eval tokens for PPL. WT2 non-overlap ctx=2048 chunks.")
    p.add_argument("--ctx-len", type=int, default=2048)
    p.add_argument("--target-avg-bits", type=float, nargs="+",
                   default=[1.81, 2.0, 2.5, 3.0, 4.0])
    p.add_argument("--out", required=True)
    return p.parse_args()


def solve_reverse_water_filling(pi_sigma2: np.ndarray, B_budget: float) -> np.ndarray:
    """Given pi_sigma2 (N,), solve reverse water-filling:
       b_i = 0.5 * log2(lambda * pi_sigma2_i)_+
       s.t. sum b_i <= B_budget, b_i >= 0.
    Returns b_star (N,) with sum <= B_budget.
    Uses bisection on lambda."""
    if pi_sigma2.max() <= 0:
        return np.zeros_like(pi_sigma2)
    # Upper bound: full spend on top elem; lower: 0
    lam_hi = 4.0 / pi_sigma2.max()  # gives b_max ~ 1 per element for top
    lam_lo = 1.0 / (pi_sigma2.max() * 1e6)  # all zero

    for _ in range(100):
        lam = 0.5 * (lam_hi + lam_lo)
        b = 0.5 * np.log2(np.maximum(lam * pi_sigma2, 1.0))
        b_total = b.sum()
        if b_total > B_budget:
            lam_hi = lam
        else:
            lam_lo = lam
        if abs(b_total - B_budget) < 1e-3:
            break
    return b


def quantize_per_elem(K: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    """Per-element scalar quantization with variable bit counts.
    K: (B, H, T, d)
    bits: (H, T, d) or broadcastable — fractional bits treated as min(floor, ceil) weighted.
    Returns reconstructed K (same shape) after quantize-dequantize.

    Each (h, t, d) channel quantized independently to 2^b levels, asymmetric min-max per channel.
    """
    Bs, H, T, D = K.shape
    K_out = K.clone().float()
    bits_int = bits.clamp(0, 8).round().to(torch.int32)
    for h in range(H):
        for t in range(T):
            for d in range(D):
                nb = int(bits_int[h, t, d].item()) if bits_int.ndim == 3 else int(bits_int.item())
                if nb <= 0:
                    # Drop to mean
                    K_out[:, h, t, d] = K_out[:, h, t, d].mean()
                    continue
                x = K_out[:, h, t, d]
                x_min = x.min()
                x_max = x.max()
                if x_max - x_min < 1e-8:
                    continue
                levels = (1 << nb) - 1
                q = ((x - x_min) / (x_max - x_min) * levels).round()
                x_recon = q / levels * (x_max - x_min) + x_min
                K_out[:, h, t, d] = x_recon
    return K_out.to(K.dtype)


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"[load] {args.model}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).eval()
    print(f"[load] {time.time() - t0:.1f}s", flush=True)

    # Load B_ont
    bdict = torch.load(args.b_ont, map_location="cpu", weights_only=False)
    B_ont = bdict["B_ont"] if isinstance(bdict, dict) else bdict  # (L, H_kv, d, r)
    L_total, H_kv, d, R = B_ont.shape
    r_per_pair = bdict.get("r_per_pair", None) if isinstance(bdict, dict) else None
    print(f"[bont] {(L_total, H_kv, d, R)}, per-facet ranks available: {r_per_pair is not None}",
          flush=True)

    # WT2 data
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    txt = "\n\n".join(ds["text"])[:200000]  # ~50K tokens
    ids_full = tok(txt, return_tensors="pt")["input_ids"][0]
    print(f"[data] total {len(ids_full)} tokens", flush=True)

    calib_ids = ids_full[:args.calib_tokens].unsqueeze(0).to(args.device)
    eval_ids = ids_full[args.calib_tokens:args.calib_tokens + args.eval_tokens]

    # ---- Calibration pass: collect pi(t,f) and sigma_f^2 per layer ----
    # For simplicity, aggregate across heads (treat per-layer)
    print("\n[calib] computing pi(t,f) and sigma_f^2 via forward hooks", flush=True)
    calib_stats = {}  # layer_idx -> dict with pi, sigma

    k_captures = {}  # layer -> K tensor
    def make_k_hook(l):
        def h(_m, _in, out):
            k_captures[l] = out.detach()
        return h

    hooks = []
    for l in args.target_layers:
        hooks.append(model.model.layers[l].self_attn.k_proj.register_forward_hook(make_k_hook(l)))

    with torch.no_grad():
        model(calib_ids, output_attentions=True)

    for h in hooks: h.remove()

    # Extract attention outputs by running forward with output_attentions
    # Redo: capture attention weights
    t1 = time.time()
    with torch.no_grad():
        out = model(calib_ids, output_attentions=True, use_cache=False)
    print(f"[calib] forward {time.time()-t1:.1f}s", flush=True)

    attentions = out.attentions  # tuple of (B, n_q, T, T)
    n_q = model.config.num_attention_heads
    g = n_q // H_kv
    head_dim = d

    for l in args.target_layers:
        K = k_captures[l]  # (1, T, n_kv*d)
        T = K.shape[1]
        K = K.view(1, T, H_kv, d).permute(0, 2, 1, 3).float()  # (1, H_kv, T, d)
        B_l = B_ont[l].to(args.device, dtype=torch.float32)  # (H_kv, d, R)

        # Facet structure from r_per_pair
        if r_per_pair is not None:
            # Use first head's ranks as template (assume consistent)
            rs = r_per_pair.get(f"L{l}_H0", [R // 4, R // 4, R // 4, R // 4])
            facet_bounds = [0]
            for rk in rs:
                facet_bounds.append(facet_bounds[-1] + rk)
            facet_bounds = [min(b, R) for b in facet_bounds][:5]  # up to 4 facets
        else:
            step = R // 4
            facet_bounds = [0, step, 2*step, 3*step, R]

        # Compute ||B_f^T k||^2 per token per head per facet
        # coeffs[0, h, t, r] = (B_l[h, :, r]^T @ K[0, h, t, :])
        coeffs = torch.einsum("bhtd,hdr->bhtr", K, B_l)  # (1, H_kv, T, R)
        # gate per facet: energy ratio
        k_norm_sq = (K ** 2).sum(-1) + 1e-8  # (1, H_kv, T)

        # attention: average over Q heads sharing this KV head
        # attentions[l]: (1, n_q, T, T)
        A = attentions[l][0].float()  # (n_q, T, T)
        # Average Q heads per KV group
        A_kv = A.view(H_kv, g, T, T).mean(dim=1)  # (H_kv, T, T)
        # Attention mass onto k_t = sum_{t_q} A_kv[h, t_q, t_k] / T
        attn_mass = A_kv.mean(dim=1).cpu()  # (H_kv, T) — avg attn received by k_t across queries

        # pi(t, f) = attn_mass(t) * g_f(k_t) where g_f = ||B_f^T k||^2 / ||k||^2
        pi_tf_layer = np.zeros((H_kv, T, 4))
        sigma2_f_layer = np.zeros((H_kv, 4))
        for h in range(H_kv):
            for f in range(4):
                lo, hi = facet_bounds[f], facet_bounds[f + 1]
                if hi <= lo:
                    continue
                coeffs_f = coeffs[0, h, :, lo:hi]  # (T, r_f)
                energy_f = (coeffs_f ** 2).sum(-1)  # (T,)
                gate_f = (energy_f / k_norm_sq[0, h]).cpu()
                pi_tf_layer[h, :, f] = attn_mass[h].cpu().numpy() * gate_f.numpy()
                sigma2_f_layer[h, f] = energy_f.mean().cpu().item()

        calib_stats[l] = {
            "pi_tf": pi_tf_layer, "sigma2_f": sigma2_f_layer,
            "facet_bounds": facet_bounds, "T": T,
        }
        if l in args.target_layers[:3]:
            print(f"  L{l}: pi mean {pi_tf_layer.mean():.4g}, "
                  f"sigma2 max {sigma2_f_layer.max():.4g}", flush=True)

    # ---- Compute b*(t,f) per bit budget ----
    print("\n[allocate] Computing reverse water-filling b*(t,f)", flush=True)
    allocations = {}  # avg_bits -> {layer -> {head -> {t, f -> bits}}}

    for target_bits in args.target_avg_bits:
        print(f"  target avg {target_bits} bits", flush=True)
        total_budget = target_bits * sum(
            calib_stats[l]["T"] * H_kv * d for l in args.target_layers
        )
        # Flatten pi*sigma2 across all (l, h, t, f) -> per element via facet mapping
        all_pi_sigma2 = []
        index_map = []
        for l in args.target_layers:
            s = calib_stats[l]
            for h in range(H_kv):
                for f in range(4):
                    for t in range(s["T"]):
                        val = s["pi_tf"][h, t, f] * s["sigma2_f"][h, f]
                        all_pi_sigma2.append(val)
                        # Map to d channels within facet
                        lo, hi = s["facet_bounds"][f], s["facet_bounds"][f + 1]
                        index_map.append((l, h, t, f, lo, hi))
        all_pi_sigma2 = np.array(all_pi_sigma2)

        # Normalize budget — water-fill over facet-level aggregates, then distribute within facet uniformly
        # For simplicity, allocate per-(l,h,t,f) block
        n_blocks = len(all_pi_sigma2)
        n_elems_per_block = np.array([r[5] - r[4] for r in index_map])
        block_budgets = target_bits * n_elems_per_block
        # Water-fill on block avg
        b_block = solve_reverse_water_filling(all_pi_sigma2, total_budget / n_elems_per_block.mean())
        allocations[target_bits] = (b_block, index_map, n_elems_per_block)

    # ---- Collapse b*(t,f) -> per-(l,h,f) average bits (position-independent) ----
    print("\n[collapse] Averaging b*(t,f) over token positions -> b*(l,h,f)", flush=True)
    alloc_lhf = {}
    alloc_summary = {}

    for target_bits in args.target_avg_bits:
        b_block, index_map, n_elems = allocations[target_bits]
        bits_sum, bits_cnt = {}, {}
        for idx, (l, h, t, f, lo, hi) in enumerate(index_map):
            key = (l, h, f)
            bits_sum[key] = bits_sum.get(key, 0.0) + b_block[idx]
            bits_cnt[key] = bits_cnt.get(key, 0) + 1

        lhf_arr = np.zeros((len(args.target_layers), H_kv, 4), dtype=np.float64)
        for li_idx, l in enumerate(args.target_layers):
            for h in range(H_kv):
                for f in range(4):
                    key = (l, h, f)
                    if bits_cnt.get(key, 0) > 0:
                        lhf_arr[li_idx, h, f] = bits_sum[key] / bits_cnt[key]

        achieved_total = (b_block * n_elems).sum()
        achieved_avg = achieved_total / n_elems.sum()
        alloc_lhf[target_bits] = lhf_arr
        alloc_summary[f"avg_bits_{target_bits}"] = {
            "target_avg_bits": target_bits,
            "achieved_avg_bits": float(achieved_avg),
            "lhf_bits_mean": float(lhf_arr.mean()),
            "lhf_bits_max": float(lhf_arr.max()),
        }
        print(f"  target {target_bits}: achieved {achieved_avg:.3f}, "
              f"lhf range [{lhf_arr.min():.2f}, {lhf_arr.max():.2f}]", flush=True)

    # ---- PPL Evaluation via hook-mode quantized forward ----
    print(f"\n[eval] Preparing WT2 test data (eval_tokens={args.eval_tokens})", flush=True)
    eval_text = load_wikitext_test(None)
    eval_windows_full = tokenize_and_chunk(eval_text, tok, args.ctx_len, drop_last=True)
    max_windows = max(1, args.eval_tokens // args.ctx_len)
    eval_windows = eval_windows_full[:max_windows]
    eval_total_tok = sum(w.shape[0] for w in eval_windows)
    print(f"[eval] {len(eval_windows)} windows, {eval_total_tok} tokens, ctx={args.ctx_len}", flush=True)

    cfg = model.config
    n_kv_eval = cfg.num_key_value_heads
    n_q_eval = cfg.num_attention_heads
    head_dim_eval = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_q_eval)

    def make_thm618_quant_fn(target_bits_val):
        lhf = alloc_lhf[target_bits_val]
        layer_to_idx = {l: i for i, l in enumerate(args.target_layers)}
        def quant_fn(k_4d, layer_idx):
            if layer_idx not in layer_to_idx:
                return k_4d
            li = layer_to_idx[layer_idx]
            B_l = B_ont[layer_idx].to(device=k_4d.device, dtype=torch.float32)
            s = calib_stats[layer_idx]
            fb = s["facet_bounds"]
            Bs, H_q, T_e, D = k_4d.shape
            k_out = torch.zeros_like(k_4d)
            for h in range(H_q):
                K_h = k_4d[:, h].float()
                B_h = B_l[h]
                coeffs = K_h @ B_h
                coeffs_q = torch.zeros_like(coeffs)
                for f_idx in range(min(4, len(fb) - 1)):
                    lo, hi = fb[f_idx], fb[f_idx + 1]
                    if hi <= lo:
                        continue
                    nb = max(0, min(8, round(lhf[li, h, f_idx])))
                    cf = coeffs[:, :, lo:hi]
                    if nb <= 0:
                        coeffs_q[:, :, lo:hi] = 0.0
                    else:
                        levels = (1 << nb) - 1
                        c_min = cf.amin(dim=1, keepdim=True)
                        c_max = cf.amax(dim=1, keepdim=True)
                        scale = (c_max - c_min).clamp(min=1e-8) / levels
                        q = ((cf - c_min) / scale).round().clamp(0, levels)
                        coeffs_q[:, :, lo:hi] = q * scale + c_min
                K_ont_q = coeffs_q @ B_h.T
                K_res = K_h - (coeffs @ B_h.T)
                K_res_q = _asymmetric_per_channel(K_res.unsqueeze(0), 2, residual_R=0).squeeze(0)
                k_out[:, h] = (K_ont_q + K_res_q).to(k_4d.dtype)
            return k_out
        return quant_fn

    def make_kivi_fn(bits):
        def quant_fn(k_4d, layer_idx):
            return _asymmetric_per_channel(k_4d, bits, residual_R=0)
        return quant_fn

    fp16_fn = lambda k, li: k

    print("\n" + "=" * 60, flush=True)
    print("[PPL EVAL] Starting quantized KV cache PPL measurement", flush=True)
    print("=" * 60, flush=True)
    ppl_results = []

    print("\n[ppl] fp16 (no quantization)", flush=True)
    with install_k_proj_hooks(model, fp16_fn, n_kv=n_kv_eval, head_dim=head_dim_eval):
        r = evaluate_ppl(model, eval_windows, args.device)
    r["method"] = "fp16"; r["target_avg_bits"] = 16.0
    print(f"  PPL = {r['ppl']:.4f}", flush=True)
    ppl_results.append(r)

    print("\n[ppl] KIVI uniform 2-bit", flush=True)
    with install_k_proj_hooks(model, make_kivi_fn(2), n_kv=n_kv_eval, head_dim=head_dim_eval):
        r = evaluate_ppl(model, eval_windows, args.device)
    r["method"] = "kivi_uniform"; r["target_avg_bits"] = 2.0
    print(f"  PPL = {r['ppl']:.4f}", flush=True)
    ppl_results.append(r)

    for target_bits in args.target_avg_bits:
        print(f"\n[ppl] Thm 6.18 weighted @ {target_bits} avg bits", flush=True)
        with install_k_proj_hooks(model, make_thm618_quant_fn(target_bits), n_kv=n_kv_eval, head_dim=head_dim_eval):
            r = evaluate_ppl(model, eval_windows, args.device)
        r["method"] = "thm618_weighted"; r["target_avg_bits"] = target_bits
        print(f"  PPL = {r['ppl']:.4f}", flush=True)
        ppl_results.append(r)

    print("\n" + "=" * 60, flush=True)
    print(f"{'Method':<25} {'Avg Bits':>10} {'PPL':>10}", flush=True)
    print("-" * 60, flush=True)
    for r in ppl_results:
        print(f"{r['method']:<25} {r['target_avg_bits']:>10.2f} {r['ppl']:>10.4f}", flush=True)
    print("=" * 60, flush=True)

    out_blob = {
        "model": args.model,
        "target_layers": list(args.target_layers),
        "calib_tokens": args.calib_tokens,
        "eval_tokens": eval_total_tok,
        "ctx_len": args.ctx_len,
        "allocations": alloc_summary,
        "ppl_results": ppl_results,
    }
    with open(out_path, "w") as f:
        json.dump(out_blob, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
