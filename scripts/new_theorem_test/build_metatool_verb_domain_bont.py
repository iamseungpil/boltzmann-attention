#!/usr/bin/env python3
"""F9 — Build MetaTool B_ont along the NMI-verified verb × domain orthogonal axes.

Motivation:
  F8b/c NMI probe confirmed that A_verb × D_domain is the ONLY orthogonal
  facet pair on MetaTool (NMI 0.185, < 0.3 threshold).  F9 tests whether
  this NMI-verified orthogonal structure translates into downstream tool-
  selection accuracy on Subtask1.

Four variants (all shape [L_total, n_kv, head_dim, R=12]):
  V    — per-verb direction: group plugins by verb, mean K per verb,
         stack across verbs, SVD → top-12 basis
  D    — per-domain direction: group plugins by domain, mean K per domain,
         stack across domains, SVD → top-12 basis
  V+D  — concat V and D columns + QR → top-12
  flat — all plugins one pool, SVD → top-12 (re-baseline)

Uses MetaTool `plugin_info.json` for anchor source (description_for_model)
and `afod_probe_metatool_toolbench.py` heuristics for verb/domain labels.

Output: external/SEKA/seka_projections/f9-qwen25-7b-metatool-{V,D,VD,flat}/B_ont.pt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

# Reuse the labelling heuristics from the probe
from afod_probe_metatool_toolbench import (
    extract_verb, extract_domain,
)

R_BONT = 12
METATOOL_INFO = "/tmp/MetaTool/dataset/plugin_info.json"


@torch.no_grad()
def capture_last_token_K(model, tok, text, sel_layers, n_kv, head_dim):
    enc = tok(text, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(next(model.parameters()).device)
    caps = {}

    def make_hook(i_layer):
        def _hook(mod, inputs, output):
            out = output
            if out.dim() == 3:
                B, T, D = out.shape
                if D != n_kv * head_dim:
                    return
                out_view = out.view(B, T, n_kv, head_dim)
            elif out.dim() == 4:
                out_view = out
            else:
                return
            caps[i_layer] = out_view[0, -1, :, :].detach().float().cpu()
        return _hook

    handles = []
    for i, L in enumerate(sel_layers):
        attn = model.model.layers[L].self_attn
        module = attn.k_norm if hasattr(attn, "k_norm") else attn.k_proj
        handles.append(module.register_forward_hook(make_hook(i)))
    try:
        model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    if len(caps) != len(sel_layers):
        return None
    return torch.stack([caps[i] for i in range(len(sel_layers))])  # (L, n_kv, head_dim)


def _svd_topk(mat: torch.Tensor, k: int) -> torch.Tensor:
    if mat.shape[1] == 0:
        return torch.zeros(mat.shape[0], k)
    U, _, _ = torch.linalg.svd(mat, full_matrices=False)
    r_eff = min(k, U.shape[1])
    out = torch.zeros(mat.shape[0], k)
    out[:, :r_eff] = U[:, :r_eff]
    return out


def build_group_bont(K_by_plugin, group_map, L, H, d, r_bont=R_BONT) -> torch.Tensor:
    """For each group, compute mean K across members, stack across groups per-(L,h),
    SVD → top-r_bont left singular vectors."""
    B = torch.zeros(L, H, d, r_bont)
    groups = sorted(set(group_map.values()))
    for li in range(L):
        for h in range(H):
            group_means = []
            for g in groups:
                members = [p for p in K_by_plugin if group_map.get(p) == g]
                if not members:
                    continue
                mean = torch.stack([K_by_plugin[m][li, h, :] for m in members]).mean(dim=0)
                group_means.append(mean)
            if not group_means:
                continue
            M = torch.stack(group_means, dim=1)  # (d, n_groups)
            B[li, h] = _svd_topk(M, r_bont)
    return B


def build_flat_bont(K_by_plugin, L, H, d, r_bont=R_BONT) -> torch.Tensor:
    """Flat: all plugin K vectors stacked, SVD per-(L,h)."""
    B = torch.zeros(L, H, d, r_bont)
    plugins = sorted(K_by_plugin.keys())
    for li in range(L):
        for h in range(H):
            M = torch.stack([K_by_plugin[p][li, h, :] for p in plugins], dim=1)
            B[li, h] = _svd_topk(M, r_bont)
    return B


def build_union_bont(B_V, B_D, r_bont=R_BONT) -> torch.Tensor:
    L, H, d, _ = B_V.shape
    B = torch.zeros(L, H, d, r_bont)
    for li in range(L):
        for h in range(H):
            cat = torch.cat([B_V[li, h], B_D[li, h]], dim=1)
            norms = cat.norm(dim=0)
            cat = cat[:, norms > 1e-6]
            if cat.shape[1] == 0:
                continue
            Q, _ = torch.linalg.qr(cat)
            r_eff = min(r_bont, Q.shape[1])
            B[li, h, :, :r_eff] = Q[:, :r_eff]
    return B


def save_bont(B, out_path: Path, label: str, model_name: str,
              L_total, n_kv, head_dim, sel_layers):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "B_ont": B,
        "model": model_name,
        "r_ont": R_BONT,
        "n_layers": L_total,
        "n_kv": n_kv,
        "head_dim": head_dim,
        "target_layers": sel_layers,
        "facet_order": [label],
        "construction": f"f9_metatool_{label}",
    }, out_path)
    print(f"[saved {label}] {out_path}  shape={tuple(B.shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="external/SEKA/seka_projections")
    ap.add_argument("--variants", default="V,D,VD,flat")
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

    # Extract K for each plugin description
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

    out_dir = REPO / args.out_dir
    variants = [v.strip() for v in args.variants.split(",")]

    def _path(lbl):
        return out_dir / f"f9-qwen25-7b-metatool-{lbl}" / "B_ont.pt"

    B_V = B_D = None
    if "V" in variants:
        print("\n=== variant V (verb groups) ===")
        B_V = build_group_bont(K_by_plugin, verb_map, L_total, n_kv, head_dim)
        save_bont(B_V, _path("V"), "V", args.model, L_total, n_kv, head_dim, sel_layers)
    if "D" in variants:
        print("\n=== variant D (domain groups) ===")
        B_D = build_group_bont(K_by_plugin, domain_map, L_total, n_kv, head_dim)
        save_bont(B_D, _path("D"), "D", args.model, L_total, n_kv, head_dim, sel_layers)
    if "VD" in variants:
        print("\n=== variant V+D (union) ===")
        if B_V is None:
            B_V = build_group_bont(K_by_plugin, verb_map, L_total, n_kv, head_dim)
        if B_D is None:
            B_D = build_group_bont(K_by_plugin, domain_map, L_total, n_kv, head_dim)
        B_VD = build_union_bont(B_V, B_D)
        save_bont(B_VD, _path("VD"), "VD", args.model, L_total, n_kv, head_dim, sel_layers)
    if "flat" in variants:
        print("\n=== variant flat (all plugins) ===")
        B_flat = build_flat_bont(K_by_plugin, L_total, n_kv, head_dim)
        save_bont(B_flat, _path("flat"), "flat", args.model, L_total, n_kv, head_dim, sel_layers)

    print("\n[done] all variants saved")


if __name__ == "__main__":
    main()
