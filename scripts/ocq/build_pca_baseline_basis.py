#!/usr/bin/env python3
"""build_pca_baseline_basis.py — PCA-of-K basis ablation builder.

The steering paper argues that the catalog-derived ontology basis `B_ont`
is the reason query-side contraction has the correct sign, and that a
rank-matched data-driven subspace does not reproduce the effect. This
script builds the ablation basis: stack K activations across calibration
prompts per (layer, kv_head), take the top-r right-singular vectors, and
save the result in the same (L, H_kv, d, r) shape as `B_ont` so it drops
into every steering evaluator via `--b-ont`.

Usage:
  python scripts/ocq/build_pca_baseline_basis.py \\
    --model Qwen/Qwen2.5-7B-Instruct \\
    --dataset /tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json \\
    --n-calib 256 --max-tokens 512 --rank 24 \\
    --out reports/pca_baseline_bases/qwen25_7b_subtask4_r24.pt

Result schema:
  {
    "B_ont": (L, H_kv, head_dim, rank) float32 tensor with orthonormal columns,
    "basis_source": "pca_calibration",
    "model": <model name>,
    "n_calib": <int>,
    "max_tokens": <int>,
    "rank": <int>,
    "dataset": <path>,
  }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ocq"))

from eval_metatool_subtask1 import resolve_dtype
from eval_metatool_subtask4 import build_fc_prompt
from eval_metatool_subtask1 import parse_candidates


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default=None,
                   help="bfloat16/float16/float32 override. Default uses resolve_dtype().")
    p.add_argument("--dataset",
                   default="/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json")
    p.add_argument("--n-calib", type=int, default=256,
                   help="Number of calibration prompts.")
    p.add_argument("--max-tokens", type=int, default=512,
                   help="Max sequence length used when collecting K activations.")
    p.add_argument("--rank", type=int, default=24,
                   help="Number of singular vectors kept per (layer, kv_head).")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def collect_k_activations(model, tok, prompts, device, max_tokens):
    """Run the model once per prompt, capture K-projection output per layer.

    Returns a list of length L with each item shaped (H_kv, T_total, head_dim).
    """
    cfg = model.config
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    L = cfg.num_hidden_layers

    # Accumulate K activations into per-layer lists first, then concat.
    per_layer_chunks = [[] for _ in range(L)]
    handles = []

    def make_k_capture(li):
        def hook(_mod, _inp, out):
            B, T, D = out.shape
            if D != n_kv * head_dim:
                return
            K = out.view(B, T, n_kv, head_dim).detach().float().cpu()
            # Flatten batch and T into a single token axis.
            K = K.permute(2, 0, 1, 3).reshape(n_kv, B * T, head_dim)
            per_layer_chunks[li].append(K)
        return hook

    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_k_capture(li)))

    try:
        model.eval()
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                ids = tok(prompt, return_tensors="pt", truncation=True,
                          max_length=max_tokens).to(device)
                model(**ids)
                if (i + 1) % 32 == 0 or i == len(prompts) - 1:
                    print(f"  [calib] {i+1}/{len(prompts)}", flush=True)
    finally:
        for h in handles:
            h.remove()

    # Concat per layer: (H_kv, N_total_tokens, head_dim)
    per_layer = []
    for li in range(L):
        stack = torch.cat(per_layer_chunks[li], dim=1)
        per_layer.append(stack)
    return per_layer, L, n_kv, head_dim


def pca_basis_per_head(K_stack, rank):
    """K_stack: (H_kv, N, head_dim) float32. Returns (H_kv, head_dim, rank)
    with orthonormal columns via SVD on each head's token matrix."""
    H_kv, N, d = K_stack.shape
    basis = torch.zeros(H_kv, d, rank, dtype=torch.float32)
    for h in range(H_kv):
        X = K_stack[h]
        # Center
        X = X - X.mean(dim=0, keepdim=True)
        # SVD: X = U S V^T, with V^T having shape (min(N,d), d).
        try:
            _, _, Vh = torch.linalg.svd(X, full_matrices=False)
        except RuntimeError:
            Vh = torch.linalg.svd(X.to(torch.float64), full_matrices=False)[2].to(torch.float32)
        # Top-r right-singular vectors (rows of Vh) are the PCA directions.
        top_r = Vh[:rank].T  # (d, rank)
        if top_r.shape[1] < rank:
            # Pad with zeros if rank exceeds data dimension.
            pad = torch.zeros(d, rank - top_r.shape[1])
            top_r = torch.cat([top_r, pad], dim=1)
        basis[h] = top_r.contiguous()
    return basis


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"[load] {args.model}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if args.dtype is None else {
        "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32
    }[args.dtype]
    try:
        dtype = resolve_dtype(args.model, args.dtype) if args.dtype else dtype
    except Exception:
        pass
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device,
        attn_implementation="eager", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[load] done in {time.time()-t0:.1f}s", flush=True)

    with open(args.dataset) as f:
        data = json.load(f)
    data = data[: args.n_calib]
    prompts = []
    for entry in data:
        action_prompt = entry.get("action_prompt") or entry.get("prompt", "")
        cands = parse_candidates(action_prompt) if action_prompt else []
        if not action_prompt:
            continue
        fc = build_fc_prompt(tok, action_prompt, cands)
        prompts.append(fc)
    print(f"[calib] {len(prompts)} prompts", flush=True)

    per_layer, L, n_kv, head_dim = collect_k_activations(
        model, tok, prompts, args.device, args.max_tokens
    )

    # Free GPU memory before doing SVD on CPU.
    del model
    torch.cuda.empty_cache()

    print(f"[svd] computing per-(layer, head) PCA bases rank={args.rank}", flush=True)
    B_out = torch.zeros(L, n_kv, head_dim, args.rank, dtype=torch.float32)
    for li in range(L):
        B_out[li] = pca_basis_per_head(per_layer[li], args.rank)
        if (li + 1) % 4 == 0 or li == L - 1:
            print(f"  [svd] layer {li+1}/{L}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "B_ont": B_out,
        "basis_source": "pca_calibration",
        "model": args.model,
        "n_calib": len(prompts),
        "max_tokens": args.max_tokens,
        "rank": args.rank,
        "dataset": args.dataset,
    }
    torch.save(payload, out_path)
    print(f"[write] {out_path}  shape={tuple(B_out.shape)}", flush=True)


if __name__ == "__main__":
    main()
