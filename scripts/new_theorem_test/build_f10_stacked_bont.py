#!/usr/bin/env python3
"""F10 — Build stacked V+D B_ont with facet_split metadata for query-conditional gating.

Motivation (NEW_THEOREM_TEST.md §5 Phase F10):
  Per-token energy-ratio gating delta_K_t = alpha * sum_f alpha_f(K_t) * B_f B_f^T K_t
  requires B_V and B_D as separate blocks within a single B_ont tensor, with
  facet_split metadata so the eval hook can recover them.

Output shape: (L, n_kv, head_dim, R=24) where columns 0:12 = V, 12:24 = D.
Metadata: facet_split = {"V": [0, 12], "D": [12, 24]}.

Reuses build_metatool_verb_domain_bont.py helpers (capture_last_token_K,
build_group_bont) for K extraction and per-group SVD.

Output: external/SEKA/seka_projections/f10-qwen25-7b-metatool-stacked/B_ont.pt

Usage:
  source /home/woori/venvs/seka_env/bin/activate
  CUDA_VISIBLE_DEVICES=1 python3 scripts/new_theorem_test/build_f10_stacked_bont.py \\
    --model Qwen/Qwen2.5-7B-Instruct --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from afod_probe_metatool_toolbench import extract_verb, extract_domain
from build_metatool_verb_domain_bont import (
    capture_last_token_K, build_group_bont, METATOOL_INFO,
)

R_PER_FACET = 12  # V and D each get 12 columns -> total R = 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="external/SEKA/seka_projections")
    ap.add_argument("--tag", default="f10-qwen25-7b-metatool-stacked")
    args = ap.parse_args()

    info = json.load(open(METATOOL_INFO))
    plugins: Dict[str, dict] = {}
    for p in info:
        n = p.get("name_for_model") or p.get("name_for_human")
        if not n:
            continue
        desc = (p.get("description_for_model") or "").strip()
        if not desc:
            continue
        plugins[n] = {
            "desc": desc[:400],
            "verb": extract_verb(desc),
            "domain": extract_domain(desc),
        }
    plugin_names = sorted(plugins.keys())
    print(f"[plugins] {len(plugin_names)} with non-empty descriptions")

    verb_map = {n: plugins[n]["verb"] for n in plugin_names}
    domain_map = {n: plugins[n]["domain"] for n in plugin_names}
    print(f"[verb clusters]   {len(set(verb_map.values()))}")
    print(f"[domain clusters] {len(set(domain_map.values()))}")

    print(f"[load model] {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
    ).to(args.device)
    model.eval()
    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // n_q
    L_total = len(model.model.layers)
    sel_layers = list(range(L_total))
    print(f"[config] n_q={n_q} n_kv={n_kv} d={head_dim} L_total={L_total}")

    K_by_plugin: Dict[str, torch.Tensor] = {}
    t0 = time.time()
    for k, n in enumerate(plugin_names):
        K = capture_last_token_K(model, tok, plugins[n]["desc"],
                                 sel_layers, n_kv, head_dim)
        if K is None:
            continue
        K_by_plugin[n] = K
        if (k + 1) % 50 == 0 or k == 0:
            elapsed = time.time() - t0
            eta = elapsed / (k + 1) * (len(plugin_names) - k - 1)
            print(f"  [{k+1}/{len(plugin_names)}] t={elapsed:.1f}s eta={eta:.1f}s cached={len(K_by_plugin)}", flush=True)
    print(f"[K cache] {len(K_by_plugin)}")

    # Free model GPU memory before SVD
    del model
    torch.cuda.empty_cache()

    print("\n=== build B_V (verb groups, rank 12) ===")
    B_V = build_group_bont(K_by_plugin, verb_map, L_total, n_kv, head_dim, r_bont=R_PER_FACET)
    print(f"  B_V shape={tuple(B_V.shape)}")

    print("\n=== build B_D (domain groups, rank 12) ===")
    B_D = build_group_bont(K_by_plugin, domain_map, L_total, n_kv, head_dim, r_bont=R_PER_FACET)
    print(f"  B_D shape={tuple(B_D.shape)}")

    # Stack along last dim: B_stacked[..., 0:12] = V, B_stacked[..., 12:24] = D
    B_stacked = torch.cat([B_V, B_D], dim=-1)
    R = B_stacked.shape[-1]
    print(f"\n[stacked] shape={tuple(B_stacked.shape)} R={R}")

    out_path = REPO / args.out_dir / args.tag / "B_ont.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "B_ont": B_stacked,
        "model": args.model,
        "r_ont": R,
        "n_layers": L_total,
        "n_kv": n_kv,
        "head_dim": head_dim,
        "target_layers": sel_layers,
        "facet_order": ["V", "D"],
        "facet_split": {"V": [0, R_PER_FACET], "D": [R_PER_FACET, R]},
        "construction": "f10_metatool_stacked_VD",
    }, out_path)
    print(f"[saved] {out_path}")
    print(f"[facet_split] V=[0:{R_PER_FACET}], D=[{R_PER_FACET}:{R}]")


if __name__ == "__main__":
    main()
