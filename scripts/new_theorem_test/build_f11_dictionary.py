#!/usr/bin/env python3
"""F11 — Build MOFCISS atom dictionary (raw per-plugin K anchors, NO SVD).

NEW_THEOREM_TEST.md Phase F11 §5.  Constructs the dictionary
    D = {(K_n, verb(n), domain(n)) : n in plugins}
used by `eval_metatool_subtask4_mofciss.py` at inference time for per-step
sparse (OMP) subtraction with facet-aware step-decay.

Unlike F9/F10 builders this file performs NO SVD, NO group aggregation, and
NO orthogonalization — each atom is literally the last-token K activation of
one MetaTool plugin description, preserving atom identity so OMP can select
individual tools at decode time.

Output: <out-dir>/<tag>/dictionary.pt  (torch.save) containing
    atoms_K         dict[plugin_name -> (L, n_kv, head_dim) float16]
    facet_labels    dict[plugin_name -> (verb_str, domain_str)]
    plugin_order    sorted list[str]  (canonical atom index order)
    atoms_stacked   Tensor (M, L, n_kv, head_dim) float16, pre-stacked for einsum
    model           model id str
    n_layers, n_kv, head_dim
    target_layers   list(range(L))
    construction    "f11_metatool_raw_atoms_no_svd"
    n_atoms         M

Usage:
  source /home/woori/venvs/seka_env/bin/activate
  CUDA_VISIBLE_DEVICES=0 python3 scripts/new_theorem_test/build_f11_dictionary.py \\
    --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 \\
    --tag f11-qwen25-7b-metatool-dictionary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

# Reuse the labelling heuristics + K-capture helper from the F9 builder.
from afod_probe_metatool_toolbench import extract_verb, extract_domain  # noqa: E402
from build_metatool_verb_domain_bont import (  # noqa: E402
    capture_last_token_K,
    METATOOL_INFO,
)


def _build_plugin_index() -> Tuple[list, Dict[str, dict]]:
    """Load MetaTool plugin_info.json and filter to those with descriptions."""
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
    return plugin_names, plugins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="external/SEKA/seka_projections")
    ap.add_argument("--tag", default="f11-qwen25-7b-metatool-dictionary")
    args = ap.parse_args()

    plugin_names, plugins = _build_plugin_index()
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

    # --- Extract raw last-token K per plugin (no SVD, no aggregation). ---
    K_by_plugin: Dict[str, torch.Tensor] = {}
    t0 = time.time()
    for k, n in enumerate(plugin_names):
        K = capture_last_token_K(
            model, tok, plugins[n]["desc"], sel_layers, n_kv, head_dim,
        )
        if K is None:
            continue
        # Store fp16 CPU copy; kept in fp32 during capture for numerical fidelity.
        K_by_plugin[n] = K.half().cpu().contiguous()
        if (k + 1) % 50 == 0 or k == 0:
            elapsed = time.time() - t0
            eta = elapsed / (k + 1) * (len(plugin_names) - k - 1)
            print(
                f"  [{k+1}/{len(plugin_names)}] t={elapsed:.1f}s "
                f"eta={eta:.1f}s cached={len(K_by_plugin)}",
                flush=True,
            )
    print(f"[K cache] {len(K_by_plugin)}")

    assert len(K_by_plugin) > 0, (
        "F11 dictionary build failed: no plugins captured K successfully. "
        "Check model load / hook targets."
    )

    # Free model memory before stacking.
    del model
    torch.cuda.empty_cache()

    # Canonical atom order: sorted plugin names that succeeded.
    atom_order = sorted(K_by_plugin.keys())
    facet_labels = {n: (plugins[n]["verb"], plugins[n]["domain"]) for n in atom_order}

    # Pre-stack for efficient einsum at inference:  (M, L, n_kv, head_dim)
    atoms_stacked = torch.stack([K_by_plugin[n] for n in atom_order], dim=0).contiguous()
    M = atoms_stacked.shape[0]
    print(
        f"[stacked] shape={tuple(atoms_stacked.shape)}  "
        f"M={M} L={L_total} n_kv={n_kv} d={head_dim}"
    )

    out_path = REPO / args.out_dir / args.tag / "dictionary.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "atoms_K": K_by_plugin,
            "facet_labels": facet_labels,
            "plugin_order": atom_order,
            "atoms_stacked": atoms_stacked,
            "model": args.model,
            "n_layers": L_total,
            "n_kv": n_kv,
            "head_dim": head_dim,
            "target_layers": sel_layers,
            "construction": "f11_metatool_raw_atoms_no_svd",
            "n_atoms": M,
        },
        out_path,
    )
    print(f"[saved] {out_path} | M={M} L={L_total} n_kv={n_kv} d={head_dim}")


if __name__ == "__main__":
    main()
