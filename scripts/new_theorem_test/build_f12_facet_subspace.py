#!/usr/bin/env python3
"""F12 — Build per-(L,h) facet subspace B_fac and anchors from F11 plugdes K cache.

NEW_THEOREM_TEST.md §5 Phase F12. Reuses the F11 plugdes dictionary (199
plugins, last-token K captured on Qwen2.5-7B; no GPU required) to construct:

    facet_members[f] = {n : (verb(n), domain(n)) == f}     F = 75 pairs
    anchor[f, L, h]  = mean_{n in members(f)} K_n[L, h, :]  (F, L, H, d)
    B_fac[L, h]      = SVD_topR( anchors[:, L, h, :].T )    (L, H, d, R)

The resulting (B_fac, anchors) file is loaded by `train_f12_facetrot_qk.py`
for LoRA training of facet-indexed SO(2) rotations (Thm 6.14 Hybrid
commuting-subgroup construction + Lemma 6.14.A soft-gate Lipschitz).

Usage:
  source /home/woori/venvs/seka_env/bin/activate
  python3 scripts/new_theorem_test/build_f12_facet_subspace.py \\
    --f11-dictionary external/SEKA/seka_projections/f11-qwen25-7b-metatool-plugdes/dictionary.pt \\
    --rank 32 \\
    --tag f12-qwen25-7b-metatool-facet-subspace
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def build_subspace(
    atoms_stacked: torch.Tensor,          # (M, L, H, d) fp16
    plugin_order: List[str],
    facet_labels: Dict[str, Tuple[str, str]],
    rank: int,
    min_members: int = 1,
) -> Dict[str, object]:
    M, L, H, d = atoms_stacked.shape
    assert M == len(plugin_order)

    # Group plugins by (verb, domain) facet pair
    facet_members: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, name in enumerate(plugin_order):
        pair = facet_labels.get(name)
        if pair is None:
            continue
        facet_members[pair].append(i)

    # Filter facets with >= min_members, preserve order by descending size then sorted label
    facet_entries = [
        (pair, idxs)
        for pair, idxs in facet_members.items()
        if len(idxs) >= min_members
    ]
    facet_entries.sort(key=lambda x: (-len(x[1]), x[0]))
    facet_labels_ordered = [pair for pair, _ in facet_entries]
    F = len(facet_entries)
    assert F >= 2, f"need at least 2 facets, got {F}"
    assert F >= rank, (
        f"rank ({rank}) cannot exceed facet count ({F}); set --rank <= {F}"
    )
    print(f"[facets] {F} (verb,domain) pairs; min_members={min_members}")

    # Per-(L, h), per-facet centroid in fp32 for SVD stability
    A = atoms_stacked.to(torch.float32)   # (M, L, H, d)
    anchors = torch.zeros(F, L, H, d, dtype=torch.float32)
    for f_idx, (_pair, members) in enumerate(facet_entries):
        mems = torch.tensor(members, dtype=torch.long)
        anchors[f_idx] = A.index_select(0, mems).mean(dim=0)

    # SVD per-(L, h): M_slice = anchors[:, L, h, :].T  shape (d, F)
    B_fac = torch.zeros(L, H, d, rank, dtype=torch.float32)
    sv_dbg: List[float] = []
    for ell in range(L):
        for h in range(H):
            M_slice = anchors[:, ell, h, :].T    # (d, F)
            try:
                U, S, _Vt = torch.linalg.svd(M_slice, full_matrices=False)
            except RuntimeError:
                # Fall back to CPU double precision for stability
                U, S, _Vt = torch.linalg.svd(M_slice.double(), full_matrices=False)
                U = U.float()
                S = S.float()
            r_eff = min(rank, U.shape[1])
            B_fac[ell, h, :, :r_eff] = U[:, :r_eff]
            if ell in {0, L // 2, L - 1} and h == 0:
                sv_dbg.append(float(S[0]))

    # Orthonormality sanity on a sampled head
    sample = B_fac[L // 2, 0]
    ortho_err = (sample.T @ sample - torch.eye(rank)).abs().max().item()
    print(f"[svd] orthonormality max|B^T B - I| = {ortho_err:.2e} @ mid-layer h=0")
    print(f"[svd] top singular value samples: {sv_dbg}")

    blob = {
        "B_fac": B_fac,                          # (L, H, d, R)
        "anchors": anchors,                      # (F, L, H, d)
        "facet_labels": facet_labels_ordered,    # list[tuple[verb, domain]]
        "plugin_order": plugin_order,
        "n_facets": F,
        "n_layers": L,
        "n_kv": H,
        "head_dim": d,
        "rank": rank,
    }
    return blob


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--f11-dictionary",
        default="external/SEKA/seka_projections/f11-qwen25-7b-metatool-plugdes/dictionary.pt",
    )
    ap.add_argument("--rank", type=int, default=32,
                    help="facet subspace dim R (F12 default 32 → rot_pairs=16; F13 4 → rot_pairs=2)")
    ap.add_argument("--min-members", type=int, default=1,
                    help="skip facet pairs with < this many member plugins")
    ap.add_argument("--out-dir", default="external/SEKA/seka_projections")
    ap.add_argument("--tag", default="f12-qwen25-7b-metatool-facet-subspace")
    args = ap.parse_args()

    dict_path = REPO / args.f11_dictionary if not Path(args.f11_dictionary).is_absolute() else Path(args.f11_dictionary)
    assert dict_path.exists(), f"F11 dictionary not found: {dict_path}"
    print(f"[load] {dict_path}", flush=True)
    payload = torch.load(dict_path, map_location="cpu", weights_only=False)
    atoms_stacked = payload["atoms_stacked"]
    plugin_order = payload["plugin_order"]
    facet_labels = payload["facet_labels"]
    model_id = payload.get("model", "?")
    construction_in = payload.get("construction", "?")
    print(
        f"[dict] M={atoms_stacked.shape[0]} L={atoms_stacked.shape[1]} "
        f"H={atoms_stacked.shape[2]} d={atoms_stacked.shape[3]} "
        f"src={construction_in}",
        flush=True,
    )

    blob = build_subspace(
        atoms_stacked=atoms_stacked,
        plugin_order=plugin_order,
        facet_labels=facet_labels,
        rank=args.rank,
        min_members=args.min_members,
    )
    blob["model"] = model_id
    blob["construction"] = "f12_facet_subspace_verb_domain_svd_topR"
    blob["source_dictionary"] = str(dict_path)

    out_path = REPO / args.out_dir / args.tag / "facet_subspace.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, out_path)
    print(
        f"[saved] {out_path}  F={blob['n_facets']} L={blob['n_layers']} "
        f"H={blob['n_kv']} d={blob['head_dim']} R={blob['rank']}",
        flush=True,
    )

    (out_path.parent / "meta.json").write_text(json.dumps({
        "n_facets": blob["n_facets"],
        "n_layers": blob["n_layers"],
        "n_kv": blob["n_kv"],
        "head_dim": blob["head_dim"],
        "rank": blob["rank"],
        "facet_labels_head": blob["facet_labels"][:20],
        "facet_labels_tail": blob["facet_labels"][-10:],
        "source_dictionary": str(dict_path),
        "model": model_id,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
