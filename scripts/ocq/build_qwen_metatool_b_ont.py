#!/usr/bin/env python3
"""
Build per-(layer, head) ontology basis B_ont for Qwen2.5-7B using the
MetaTool-derived ontology.

Reuses the extraction + Gram-Schmidt pipeline from
scripts/ontology_facet_basis.py but feeds in the MetaTool ontology
(catalog-derived 4-facet structure) instead of the toy product/manuf
ontology that script ships with by default.

Output format compatible with scripts/ocq/quantizer.py and the hook-mode
eval driver (loads B_ont as an external tensor):
  Tensor shape (n_layers, n_heads, d_head, r_ont)
  saved as a .pt file. r_ont is the per-(layer, head) Gram-Schmidt basis
  width (variable across heads but padded to the global maximum so the
  tensor is rectangular).

IMPORTANT: the hook in this script registers on ``layer.self_attn.k_proj``
(line below), so the collected K vectors — and therefore B_ont — are in
**pre-RoPE** space. Any eval driver that applies this basis must
rewrite the K tensor before ``apply_rotary_pos_emb`` is called, i.e.
via a forward hook on the same ``k_proj`` module. Applying B_ont to
post-RoPE ``past_key_values`` is a space mismatch.

CLI:
  python3 build_qwen_metatool_b_ont.py
    [--model Qwen/Qwen2.5-7B]
    [--target-layers all|last10|...]
    [--ontology-json reports/.../metatool_ontology.json]
    [--out external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse Phase 1.x ontology pipeline functions
REPO = Path("/home/woori/workspace_common/boltzmann-attention")
sys.path.insert(0, str(REPO / "scripts"))

# IMPORTANT: ontology_facet_basis.py sets CUDA_VISIBLE_DEVICES='1' on
# import. We override with the user's --device after import.
import ontology_facet_basis as ofb  # noqa: E402

DTYPE = torch.bfloat16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--target-layers",
        default="all",
        help="'all' / 'last10' / 'last15' / comma list e.g. '14,16,18'",
    )
    parser.add_argument(
        "--ontology-json",
        default=str(REPO / "reports" / "axis2_theoretical_verification" / "metatool_ontology.json"),
    )
    parser.add_argument(
        "--out",
        default=str(REPO / "external" / "SEKA" / "seka_projections" / "ontology-qwen25-7b-metatool" / "B_ont.pt"),
    )
    parser.add_argument("--diag", default=str(
        REPO / "reports" / "axis2_theoretical_verification" / "build_qwen_metatool_b_ont.json"
    ))
    parser.add_argument(
        "--center-facets",
        action="store_true",
        help=(
            "v2c mode: before SVD within each facet, subtract the empirical "
            "grand mean of the category-mean K columns (tool-English). "
            "Design-time only."
        ),
    )
    parser.add_argument(
        "--pad-to-max",
        action="store_true",
        help=(
            "Instead of truncating all heads to min(r_per_head), pad every "
            "head to max(r_per_head) with zero columns. Low-rank heads keep "
            "their natural rank; unused columns are zero and contribute nothing "
            "in the K-bias hook (B @ 0-col = 0). This avoids the min-truncation "
            "bottleneck where a single pathological head forces all 256 heads "
            "to lose ontology columns."
        ),
    )
    parser.add_argument(
        "--center-from-wiki",
        action="store_true",
        help=(
            "Stage 1 improved centering: compute per-(L,h) K-mean across "
            "~50k Wikipedia content tokens (large, unbiased reference for "
            "language-mean direction = W_K·E[x] + b_K) and subtract from "
            "every category-mean BEFORE Gram-Schmidt. More stable than "
            "--center-facets (which uses only 5.4k tool tokens). Mutually "
            "exclusive with --center-facets. Design-time, Mode C safe."
        ),
    )
    parser.add_argument(
        "--wiki-tokens",
        type=int,
        default=50000,
        help="Number of Wikipedia content tokens for wiki-mean calibration.",
    )
    args = parser.parse_args()
    if args.center_facets and args.center_from_wiki:
        parser.error("--center-facets and --center-from-wiki are mutually exclusive")
    return args


def parse_layers(spec: str, n_layers: int) -> List[int]:
    if spec == "all":
        return list(range(n_layers))
    if spec.startswith("last"):
        k = int(spec[4:])
        return list(range(n_layers - k, n_layers))
    if spec.startswith("first"):
        k = int(spec[5:])
        return list(range(k))
    return [int(x) for x in spec.split(",")]


@torch.no_grad()
def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path = Path(args.diag)
    diag_path.parent.mkdir(parents=True, exist_ok=True)

    # Load ontology JSON and inject into the ontology_facet_basis module
    print(f"Loading ontology JSON {args.ontology_json}", flush=True)
    with open(args.ontology_json) as f:
        ont_payload = json.load(f)
    ofb.ONTOLOGY = ont_payload["ontology"]
    ofb.FACET_ORDER = ont_payload["facet_order"]
    print(f"  facets: {ofb.FACET_ORDER}")
    for facet in ofb.FACET_ORDER:
        print(f"    {facet}: {len(ofb.ONTOLOGY[facet])} categories")
    print()

    # Monkey-patch build_facet_basis with a centered variant if requested.
    # Centering removes the per-(L,h) "language-mean" direction that
    # dominates all English-text K vectors (diagnosed 2026-04-10: u_1 has
    # |cos|>0.99 with K of arbitrary English sentences at mid/late layers).
    # Without this, top singular direction is language-mean, Range(B) is
    # contaminated, and ε_q saturates at ~r/d for every query type.
    if args.center_facets:
        print("[centered] monkey-patching ofb.build_facet_basis — "
              "subtracting per-facet grand mean before Gram-Schmidt SVD",
              flush=True)
        orig_build = ofb.build_facet_basis

        def build_facet_basis_centered(
            E_list,
            energy_threshold=ofb.ENERGY_THRESHOLD,
            min_singval_ratio=ofb.MIN_SINGVAL_RATIO,
        ):
            # Mean-subtract each facet's category-mean matrix so the
            # language-common direction is removed before residualization.
            E_list_c = [
                (E - E.mean(axis=1, keepdims=True)) if E.size > 0 else E
                for E in E_list
            ]
            return orig_build(
                E_list_c,
                energy_threshold=energy_threshold,
                min_singval_ratio=min_singval_ratio,
            )

        ofb.build_facet_basis = build_facet_basis_centered

    print(f"Loading model {args.model} on {args.device}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=DTYPE,
        device_map=args.device,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // n_q
    )
    print(f"  loaded in {time.time()-t0:.1f}s; n_layers={n_layers} n_kv={n_kv} head_dim={head_dim}",
          flush=True)

    target_layers = parse_layers(args.target_layers, n_layers)
    print(f"  target_layers: {target_layers}\n")

    # Patch device override on extract_category_K (the existing function
    # uses 'cuda:0'; we forward inputs/outputs on the actual device).
    # We monkey-patch via a small wrapper.
    orig_extract = ofb.extract_category_K
    DEVICE = args.device

    @torch.no_grad()
    def extract_on_device(model, tok, ontology, target_layers, n_kv, head_dim):
        n_layers = model.config.num_hidden_layers
        pl_buf = [None] * n_layers
        handles = []
        for li in target_layers:
            layer = model.model.layers[li]

            def make_hook(li_):
                def h(m, inp, out):
                    pl_buf[li_] = out.detach().cpu().float().numpy()
                return h

            handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook(li)))

        out = {}
        for facet_name, cats in ontology.items():
            for cat_name, sentences in cats.items():
                sentence_K = {(li, h): [] for li in target_layers for h in range(n_kv)}
                for sent in sentences:
                    ids = tok(sent, return_tensors="pt")["input_ids"].to(DEVICE)
                    model(ids, use_cache=False)
                    for li in target_layers:
                        K_li = pl_buf[li]
                        if K_li is None:
                            continue
                        K_li = K_li[0]
                        K_li = K_li.reshape(K_li.shape[0], n_kv, head_dim)
                        if K_li.shape[0] > 1:
                            K_li = K_li[1:]
                        for h in range(n_kv):
                            sentence_K[(li, h)].append(K_li[:, h, :].mean(0))
                for (li, h), v_list in sentence_K.items():
                    if v_list:
                        out[(facet_name, cat_name, li, h)] = np.stack(v_list).mean(0)

        for hd in handles:
            hd.remove()
        return out

    print("[1/2] Extracting per-category K vectors …", flush=True)
    t0 = time.time()
    cat_K = extract_on_device(model, tok, ofb.ONTOLOGY, target_layers, n_kv, head_dim)
    print(f"  got {len(cat_K)} (facet, cat, layer, head) entries in {time.time()-t0:.1f}s",
          flush=True)

    # --- Stage 1 centering: Wikipedia K-mean calibration ---
    wiki_means: Dict[Tuple[int, int], np.ndarray] = {}
    if args.center_from_wiki:
        print(f"\n[1b/2] Wikipedia K-mean calibration ({args.wiki_tokens} tokens) …",
              flush=True)
        t0 = time.time()
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if len(t.strip()) > 100]
        calib_text = "\n\n".join(texts[:500])
        ids = tok(calib_text, return_tensors="pt", truncation=True,
                  max_length=args.wiki_tokens)["input_ids"].to(args.device)

        pl_buf = [None] * n_layers
        handles = []
        for li in target_layers:
            layer = model.model.layers[li]
            def make_hook(li_):
                def h(m, inp, out):
                    pl_buf[li_] = out.detach().cpu().float().numpy()
                return h
            handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook(li)))

        # Run in chunks to avoid OOM on very long contexts
        chunk = 2048
        sums = {(li, h): np.zeros(head_dim, dtype=np.float64) for li in target_layers for h in range(n_kv)}
        counts = {(li, h): 0 for li in target_layers for h in range(n_kv)}
        T = ids.shape[1]
        for start in range(0, T, chunk):
            sub = ids[:, start:start + chunk]
            if sub.shape[1] == 0:
                continue
            model(sub, use_cache=False)
            for li in target_layers:
                K_li = pl_buf[li]
                if K_li is None:
                    continue
                K_li = K_li[0].reshape(K_li.shape[1], n_kv, head_dim)
                if K_li.shape[0] > 1:
                    K_li = K_li[1:]  # skip BOS of the chunk
                for h in range(n_kv):
                    sums[(li, h)] += K_li[:, h, :].sum(axis=0).astype(np.float64)
                    counts[(li, h)] += K_li.shape[0]

        for hd in handles:
            hd.remove()

        for (li, h) in sums:
            c = counts[(li, h)]
            wiki_means[(li, h)] = (sums[(li, h)] / max(c, 1)).astype(np.float32)

        total_tokens = counts[(target_layers[0], 0)]
        print(f"  wiki-mean computed over {total_tokens} content tokens "
              f"in {time.time()-t0:.1f}s", flush=True)

        # Subtract wiki_mean from every category-mean BEFORE Gram-Schmidt.
        # cat_K entries have shape (head_dim,) for each (facet, cat, li, h).
        n_subtracted = 0
        for key in list(cat_K.keys()):
            _, _, li, h = key
            cat_K[key] = (cat_K[key] - wiki_means[(li, h)]).astype(np.float32)
            n_subtracted += 1
        print(f"  subtracted wiki-mean from {n_subtracted} category K vectors",
              flush=True)

    del model
    torch.cuda.empty_cache()

    print("\n[2/2] Building per-(layer, head) facet basis …", flush=True)
    d = head_dim
    # First pass to find max r_ont across all heads (for tensor shape)
    per_pair_basis: Dict[Tuple[int, int], np.ndarray] = {}
    r_per_pair: Dict[Tuple[int, int], List[int]] = {}
    skipped = []
    for li in target_layers:
        for h in range(n_kv):
            E_list = []
            ok = True
            for facet_name in ofb.FACET_ORDER:
                cats = list(ofb.ONTOLOGY[facet_name].keys())
                cols = []
                for c in cats:
                    key = (facet_name, c, li, h)
                    if key not in cat_K:
                        ok = False
                        break
                    cols.append(cat_K[key])
                if not ok:
                    break
                E_f = np.stack(cols, axis=1).astype(np.float64)
                E_list.append(E_f)
            if not ok:
                skipped.append((li, h, "missing cat"))
                continue
            B, r_per_facet = ofb.build_facet_basis(E_list)
            if B.shape[1] == 0:
                skipped.append((li, h, "zero rank"))
                continue
            per_pair_basis[(li, h)] = B.astype(np.float32)
            r_per_pair[(li, h)] = list(r_per_facet)

    if not per_pair_basis:
        raise RuntimeError("No valid (layer, head) basis built — ontology K extraction failed")

    r_min = min(B.shape[1] for B in per_pair_basis.values())
    r_max = max(B.shape[1] for B in per_pair_basis.values())
    r_med = int(np.median([B.shape[1] for B in per_pair_basis.values()]))
    print(f"  per-head r_ont range: min={r_min}  max={r_max}  median={r_med}",
          flush=True)

    if args.pad_to_max:
        r_use = r_max
        print(f"  padding all heads to r_ont={r_max} (pad-to-max mode)",
              flush=True)
    else:
        r_use = r_min
        print(f"  truncating all heads to r_ont={r_min} for uniform tensor",
              flush=True)
    B_tensor = np.zeros((n_layers, n_kv, d, r_use), dtype=np.float32)
    for (li, h), B in per_pair_basis.items():
        r_head = min(B.shape[1], r_use)
        B_tensor[li, h, :, :r_head] = B[:, :r_head]

    B_t = torch.from_numpy(B_tensor)
    payload = {
        "B_ont": B_t,
        "model": args.model,
        "target_layers": target_layers,
        "facet_order": ofb.FACET_ORDER,
        "ontology_summary": {f: list(c.keys()) for f, c in ofb.ONTOLOGY.items()},
        "n_layers": n_layers,
        "n_kv": n_kv,
        "head_dim": d,
        "r_ont": r_use,
        "r_max_unrtruncated": r_max,
        "r_median_unrtruncated": r_med,
        "r_per_pair": {f"L{li}_H{h}": r for (li, h), r in r_per_pair.items()},
        "skipped": skipped,
        "center_mode": (
            "wiki_mean" if args.center_from_wiki
            else ("empirical_grand" if args.center_facets else "none")
        ),
    }
    torch.save(payload, out_path)
    print(f"\nwrote {out_path}  (shape={tuple(B_t.shape)})")

    diag_path.write_text(json.dumps({
        "model": args.model,
        "target_layers": target_layers,
        "facet_order": ofb.FACET_ORDER,
        "n_pairs_built": len(per_pair_basis),
        "r_min": min(B.shape[1] for B in per_pair_basis.values()),
        "r_max": r_max,
        "r_median": int(np.median([B.shape[1] for B in per_pair_basis.values()])),
        "n_skipped": len(skipped),
        "out_path": str(out_path),
    }, indent=2))
    print(f"wrote diagnostic {diag_path}")


if __name__ == "__main__":
    main()
