#!/usr/bin/env python3
"""F7 — Build ontology-structured B_ont (variants H, C, R, F, flat) + 3 shuffled.

Pipeline:
    1. Load anchor_sentences_{domain}.json and ontology_graph_{domain}.json.
    2. Run Qwen2.5-7B forward pass over every unique sentence, cache K at
       the final content token for every (L, h): K_cache[sent_id] shape (L, H, d).
    3. Build seven B_ont.pt variants that all have shape (L, H, d, R=12):
         - flat       : naive SVD over all class sentences (re-baseline for Phase C)
         - H          : nested filtration by is-a depth (parents first, residualize)
         - C          : block-diagonal per part-of edge (one SVD per edge, concat)
         - R          : relation deltas δ_r = mean(K_tgt) − mean(K_src), stacked SVD
         - F          : concat of H, C, R → Gram-Schmidt → top 12 cols
         - H_shuffled : shuffle is-a depth assignments before residualization
         - C_shuffled : shuffle whole/part edge assignments before block SVD
         - R_shuffled : shuffle (src,tgt) pairs across relations before delta
       (All within the SAME fixed K_cache — only structural metadata changes.)
    4. Save each to external/SEKA/seka_projections/f7-<variant>/B_ont.pt.

Usage:
    python scripts/new_theorem_test/build_ontology_structured_bont.py \\
        --domain telecom --device cuda:0 \\
        --out-dir external/SEKA/seka_projections
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

R_BONT = 12  # match Phase C / F6 baseline


# ---------------------------------------------------------------------
# K extraction (shared across all variants)
# ---------------------------------------------------------------------

@torch.no_grad()
def capture_last_token_K(model, tok, text, sel_layers, n_kv, head_dim):
    """Forward pass on `text`, return K at final token for every sel_layer, shape (L, n_kv, head_dim)."""
    enc = tok(text, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(next(model.parameters()).device)
    caps: Dict[int, torch.Tensor] = {}

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


# ---------------------------------------------------------------------
# Build cache: all unique sentences → K tensor
# ---------------------------------------------------------------------

def collect_unique_sentences(anchors: dict) -> List[Tuple[str, str]]:
    """Return [(sent_id, text), ...] for all class + tool + part-of sentences."""
    out = []
    seen = set()
    for cname, entry in anchors["per_class"].items():
        for i, s in enumerate(entry["sentences"]):
            sid = f"class::{cname}::{i}"
            if s and s not in seen:
                out.append((sid, s))
                seen.add(s)
    for tname, entry in anchors["per_tool"].items():
        for i, s in enumerate(entry["sentences"]):
            sid = f"tool::{tname}::{i}"
            if s and s not in seen:
                out.append((sid, s))
                seen.add(s)
    for pkey, entry in anchors["per_part_of"].items():
        for i, s in enumerate(entry["sentences"]):
            sid = f"part::{pkey}::{i}"
            if s and s not in seen:
                out.append((sid, s))
                seen.add(s)
    return out


# ---------------------------------------------------------------------
# Variant builders — all take shared K_by_id dict, produce (L, H, d, R) tensor.
# ---------------------------------------------------------------------

def _svd_topk(mat: torch.Tensor, k: int) -> torch.Tensor:
    """Return top-k left singular vectors of mat (d × N). If N < k, pad with zeros."""
    if mat.shape[1] == 0:
        return torch.zeros(mat.shape[0], k)
    U, _, _ = torch.linalg.svd(mat, full_matrices=False)
    r_eff = min(k, U.shape[1])
    out = torch.zeros(mat.shape[0], k)
    out[:, :r_eff] = U[:, :r_eff]
    return out


def _gather_K(K_by_id, ids_pattern, L_idx, h):
    """For each sent_id matching pattern (prefix), return a (d, N) matrix."""
    cols = []
    for sid in ids_pattern:
        if sid in K_by_id:
            cols.append(K_by_id[sid][L_idx, h, :])
    if not cols:
        return torch.zeros(K_by_id[next(iter(K_by_id))].shape[-1], 0)
    return torch.stack(cols, dim=1)


def build_flat(K_by_id: Dict[str, torch.Tensor], anchors: dict, L: int, H: int, d: int) -> torch.Tensor:
    """Baseline: ALL class sentences, no structure."""
    B = torch.zeros(L, H, d, R_BONT)
    sent_ids = [sid for sid in K_by_id if sid.startswith("class::")]
    for li in range(L):
        for h in range(H):
            mat = _gather_K(K_by_id, sent_ids, li, h)
            B[li, h] = _svd_topk(mat, R_BONT)
    return B


def build_hierarchy(K_by_id, anchors, L, H, d, depth_override=None) -> torch.Tensor:
    """Variant H: nested filtration by is-a depth.
    depth_override: if provided, use this dict instead of anchors[per_class][*]['depth']."""
    per_class = anchors["per_class"]
    classes_by_depth = defaultdict(list)
    for cname, entry in per_class.items():
        d_c = depth_override[cname] if depth_override else entry["depth"]
        classes_by_depth[d_c].append(cname)
    ordered_classes: List[str] = []
    for d_c in sorted(classes_by_depth.keys()):
        ordered_classes.extend(classes_by_depth[d_c])

    B = torch.zeros(L, H, d, R_BONT)
    for li in range(L):
        for h in range(H):
            accum_basis = torch.zeros(d, 0)  # already-accounted-for directions
            accum_cols: List[torch.Tensor] = []
            for cname in ordered_classes:
                n_sent = len(per_class[cname]["sentences"])
                sent_ids = [f"class::{cname}::{i}" for i in range(n_sent)]
                mat = _gather_K(K_by_id, sent_ids, li, h)
                if mat.shape[1] == 0:
                    continue
                # Residualize against accum_basis
                if accum_basis.shape[1] > 0:
                    mat = mat - accum_basis @ (accum_basis.T @ mat)
                # Add top-1 direction from residual
                U = _svd_topk(mat, 1)[:, 0:1]
                if U.norm() < 1e-8:
                    continue
                accum_cols.append(U)
                accum_basis = torch.cat([accum_basis, U], dim=1)
                if accum_basis.shape[1] >= R_BONT:
                    break
            if accum_cols:
                merged = torch.cat(accum_cols, dim=1)
                r_eff = min(R_BONT, merged.shape[1])
                B[li, h, :, :r_eff] = merged[:, :r_eff]
    return B


def build_composition(K_by_id, anchors, L, H, d, edge_override=None) -> torch.Tensor:
    """Variant C: block-diagonal per part-of edge, concat.
    edge_override: remap part-of key → (whole, part, via_field) for shuffled control."""
    per_part = anchors["per_part_of"]
    edge_keys = list(per_part.keys())
    # Allocate dims: we'll give 1 dim per edge, cap at R_BONT
    B = torch.zeros(L, H, d, R_BONT)
    for li in range(L):
        for h in range(H):
            cols: List[torch.Tensor] = []
            for pkey in edge_keys:
                # If shuffled, use override key's sentences; but always use K from real key
                real_key = pkey
                n_sent = len(per_part[real_key]["sentences"])
                sent_ids = [f"part::{real_key}::{i}" for i in range(n_sent)]
                # For shuffled: the sent_ids stay same (we can't rebuild K); only logical
                # assignment changes. To produce a truly shuffled basis here, we should
                # permute which edge's K goes into which slot.
                if edge_override is not None:
                    src_pkey = edge_override[pkey]
                    n_sent = len(per_part[src_pkey]["sentences"])
                    sent_ids = [f"part::{src_pkey}::{i}" for i in range(n_sent)]
                mat = _gather_K(K_by_id, sent_ids, li, h)
                if mat.shape[1] == 0:
                    continue
                U = _svd_topk(mat, 1)[:, 0:1]
                # Orthogonalize against previously collected cols
                if cols:
                    prev = torch.cat(cols, dim=1)
                    U = U - prev @ (prev.T @ U)
                    if U.norm() < 1e-8:
                        continue
                    U = U / U.norm()
                cols.append(U)
                if len(cols) >= R_BONT:
                    break
            if cols:
                merged = torch.cat(cols, dim=1)
                r_eff = min(R_BONT, merged.shape[1])
                B[li, h, :, :r_eff] = merged[:, :r_eff]
    return B


def build_relation(K_by_id, anchors, L, H, d, rel_override=None) -> torch.Tensor:
    """Variant R: per-relation δ_r = mean(K of target class sents) − mean(K of source class sents).
    rel_override: mapping tool_name → (source_types_list, target_type) for shuffled control."""
    per_relation = anchors["per_relation"]
    per_class = anchors["per_class"]

    rel_keys = list(per_relation.keys())
    B = torch.zeros(L, H, d, R_BONT)
    for li in range(L):
        for h in range(H):
            deltas: List[torch.Tensor] = []
            for rkey in rel_keys:
                if rel_override is not None:
                    src_types, tgt = rel_override[rkey]
                else:
                    src_types = per_relation[rkey]["source_types"]
                    tgt = per_relation[rkey]["target_type"]

                # Gather K for source + target CLASSES
                src_ids: List[str] = []
                for src in src_types:
                    if src in per_class:
                        n = len(per_class[src]["sentences"])
                        src_ids.extend(f"class::{src}::{i}" for i in range(n))
                tgt_ids: List[str] = []
                if tgt in per_class:
                    n = len(per_class[tgt]["sentences"])
                    tgt_ids = [f"class::{tgt}::{i}" for i in range(n)]

                K_src = _gather_K(K_by_id, src_ids, li, h)
                K_tgt = _gather_K(K_by_id, tgt_ids, li, h)
                if K_src.shape[1] == 0 or K_tgt.shape[1] == 0:
                    continue
                delta = K_tgt.mean(dim=1) - K_src.mean(dim=1)  # (d,)
                if delta.norm() < 1e-8:
                    continue
                deltas.append(delta.unsqueeze(1))

            if deltas:
                D = torch.cat(deltas, dim=1)  # (d, n_rel)
                U = _svd_topk(D, R_BONT)
                B[li, h] = U
    return B


def build_full(B_H: torch.Tensor, B_C: torch.Tensor, B_R: torch.Tensor) -> torch.Tensor:
    """Concat H, C, R columns and re-orthogonalize via QR → keep top R_BONT cols."""
    L, H, d, _ = B_H.shape
    B_F = torch.zeros(L, H, d, R_BONT)
    for li in range(L):
        for h in range(H):
            cat = torch.cat([B_H[li, h], B_C[li, h], B_R[li, h]], dim=1)  # (d, 3*R)
            # Strip zero cols
            norms = cat.norm(dim=0)
            cat = cat[:, norms > 1e-6]
            if cat.shape[1] == 0:
                continue
            Q, _ = torch.linalg.qr(cat)
            r_eff = min(R_BONT, Q.shape[1])
            B_F[li, h, :, :r_eff] = Q[:, :r_eff]
    return B_F


# ---------------------------------------------------------------------
# Shuffled-control generators (deterministic, seeded)
# ---------------------------------------------------------------------

def shuffle_depths(anchors: dict, seed: int) -> Dict[str, int]:
    rng = random.Random(seed)
    classes = list(anchors["per_class"].keys())
    orig_depths = [anchors["per_class"][c]["depth"] for c in classes]
    rng.shuffle(orig_depths)
    return dict(zip(classes, orig_depths))


def shuffle_part_edges(anchors: dict, seed: int) -> Dict[str, str]:
    """Produce mapping pkey -> different pkey (whose K will be read in its slot)."""
    rng = random.Random(seed)
    keys = list(anchors["per_part_of"].keys())
    shuf = list(keys)
    rng.shuffle(shuf)
    return dict(zip(keys, shuf))


def shuffle_relations(anchors: dict, seed: int) -> Dict[str, Tuple[List[str], str]]:
    """Randomly permute (source_types, target_type) across tool names."""
    rng = random.Random(seed)
    keys = list(anchors["per_relation"].keys())
    pairs = [(anchors["per_relation"][k]["source_types"],
              anchors["per_relation"][k]["target_type"]) for k in keys]
    perm = list(pairs)
    rng.shuffle(perm)
    return dict(zip(keys, perm))


# ---------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------

def save_bont(B: torch.Tensor, out_path: Path, label: str, model_name: str,
              L_total: int, n_kv: int, head_dim: int, sel_layers: List[int]):
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
        "construction": f"f7_{label}",
    }, out_path)
    print(f"[saved {label}] {out_path}  shape={tuple(B.shape)}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="telecom")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--anchors", default="reports/new_theorem_test/phase_f7/anchor_sentences_{domain}.json")
    ap.add_argument("--out-dir", default="external/SEKA/seka_projections")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variants", default="flat,H,C,R,F,H_shuffled,C_shuffled,R_shuffled",
                    help="Comma-separated list of variants to build.")
    args = ap.parse_args()

    anchor_path = REPO / args.anchors.format(domain=args.domain)
    anchors = json.load(open(anchor_path))
    print(f"[load anchors] {anchor_path}")
    print(f"  per_class={len(anchors['per_class'])}  per_tool={len(anchors['per_tool'])}  "
          f"per_part={len(anchors['per_part_of'])}  per_relation={len(anchors['per_relation'])}")

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

    # --- Extract K for all unique sentences ---
    sent_pairs = collect_unique_sentences(anchors)
    print(f"[sentences] {len(sent_pairs)} unique anchor sentences")
    K_by_id: Dict[str, torch.Tensor] = {}
    t0 = time.time()
    for k, (sid, text) in enumerate(sent_pairs):
        K = capture_last_token_K(model, tok, text, sel_layers, n_kv, head_dim)
        if K is None:
            continue
        K_by_id[sid] = K
        if (k + 1) % 50 == 0 or k == 0:
            elapsed = time.time() - t0
            eta = elapsed / (k + 1) * (len(sent_pairs) - k - 1)
            print(f"  [{k+1}/{len(sent_pairs)}] t={elapsed:.1f}s eta={eta:.1f}s cached={len(K_by_id)}", flush=True)
    print(f"[K cache] {len(K_by_id)} sentences captured")

    # --- Build each variant ---
    variants = [v.strip() for v in args.variants.split(",")]
    out_dir = REPO / args.out_dir
    B_H = B_C = B_R = None

    def _out_path(label: str) -> Path:
        return out_dir / f"f7-qwen25-7b-tau2-{args.domain}-{label}" / "B_ont.pt"

    if "flat" in variants:
        print("\n=== variant: flat ===")
        B = build_flat(K_by_id, anchors, L_total, n_kv, head_dim)
        save_bont(B, _out_path("flat"), "flat", args.model, L_total, n_kv, head_dim, sel_layers)

    if "H" in variants:
        print("\n=== variant: H (nested filtration) ===")
        B_H = build_hierarchy(K_by_id, anchors, L_total, n_kv, head_dim)
        save_bont(B_H, _out_path("H"), "H", args.model, L_total, n_kv, head_dim, sel_layers)

    if "C" in variants:
        print("\n=== variant: C (block-diagonal composition) ===")
        B_C = build_composition(K_by_id, anchors, L_total, n_kv, head_dim)
        save_bont(B_C, _out_path("C"), "C", args.model, L_total, n_kv, head_dim, sel_layers)

    if "R" in variants:
        print("\n=== variant: R (relation deltas) ===")
        B_R = build_relation(K_by_id, anchors, L_total, n_kv, head_dim)
        save_bont(B_R, _out_path("R"), "R", args.model, L_total, n_kv, head_dim, sel_layers)

    if "F" in variants:
        print("\n=== variant: F (union) ===")
        if B_H is None: B_H = build_hierarchy(K_by_id, anchors, L_total, n_kv, head_dim)
        if B_C is None: B_C = build_composition(K_by_id, anchors, L_total, n_kv, head_dim)
        if B_R is None: B_R = build_relation(K_by_id, anchors, L_total, n_kv, head_dim)
        B_F = build_full(B_H, B_C, B_R)
        save_bont(B_F, _out_path("F"), "F", args.model, L_total, n_kv, head_dim, sel_layers)

    if "H_shuffled" in variants:
        print("\n=== variant: H_shuffled (depth permutation) ===")
        dep = shuffle_depths(anchors, args.seed)
        B = build_hierarchy(K_by_id, anchors, L_total, n_kv, head_dim, depth_override=dep)
        save_bont(B, _out_path("H_shuffled"), "H_shuffled", args.model, L_total, n_kv, head_dim, sel_layers)

    if "C_shuffled" in variants:
        print("\n=== variant: C_shuffled (edge permutation) ===")
        edge = shuffle_part_edges(anchors, args.seed)
        B = build_composition(K_by_id, anchors, L_total, n_kv, head_dim, edge_override=edge)
        save_bont(B, _out_path("C_shuffled"), "C_shuffled", args.model, L_total, n_kv, head_dim, sel_layers)

    if "R_shuffled" in variants:
        print("\n=== variant: R_shuffled (src/tgt pair permutation) ===")
        rel = shuffle_relations(anchors, args.seed)
        B = build_relation(K_by_id, anchors, L_total, n_kv, head_dim, rel_override=rel)
        save_bont(B, _out_path("R_shuffled"), "R_shuffled", args.model, L_total, n_kv, head_dim, sel_layers)

    print("\n[done] all variants saved")


if __name__ == "__main__":
    main()
