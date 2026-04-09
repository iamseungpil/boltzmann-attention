#!/usr/bin/env python3
"""
Phase 1.4 — slice an all-layers SEKA projection tensor into layer
sub-ranges, saving each as a standalone .pt file for the eval.

Usage:
  python3 scripts/phase1_slice_projection.py \
    --src external/SEKA/seka_projections/ontology-mistral-7b-v03-rank8-all \
    --short Mistral-7B-v0.3 --n_layers 32

Outputs a directory per slice:
  <src>-slice-<tag>/{short}_{pos,neg}_proj.pt

with slice tag in {last5, last10, last15, mid10, first10, early10, all}.
"""

import argparse
import torch
from pathlib import Path


def slice_one(src_dir: Path, short: str, layers: list[int], out_tag: str):
    """Load src and write a sliced copy."""
    pos = torch.load(src_dir / f'{short}_pos_proj.pt', map_location='cpu')
    neg = torch.load(src_dir / f'{short}_neg_proj.pt', map_location='cpu')
    full_layers = pos['layers']

    # Build an index map from absolute layer index → position in full_layers
    lookup = {l: i for i, l in enumerate(full_layers)}
    missing = [l for l in layers if l not in lookup]
    if missing:
        raise ValueError(f'layers {missing} not in source {full_layers}')
    sel = [lookup[l] for l in layers]

    pos_sliced = {'layers': list(layers), 'proj': pos['proj'][sel].clone()}
    neg_sliced = {'layers': list(layers), 'proj': neg['proj'][sel].clone()}

    out_dir = src_dir.parent / f'{src_dir.name.replace("-all", "")}-slice-{out_tag}'
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(pos_sliced, out_dir / f'{short}_pos_proj.pt')
    torch.save(neg_sliced, out_dir / f'{short}_neg_proj.pt')
    print(f'  {out_tag:10s} → {out_dir}  shape={tuple(pos_sliced["proj"].shape)}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, type=Path)
    p.add_argument('--short', required=True)
    p.add_argument('--n_layers', required=True, type=int)
    args = p.parse_args()

    N = args.n_layers
    slices = {
        'last5':   list(range(N - 5, N)),
        'last10':  list(range(N - 10, N)),
        'last15':  list(range(N - 15, N)),
        'mid10':   list(range((N // 2) - 5, (N // 2) + 5)),
        'first10': list(range(0, 10)),
        'first5':  list(range(0, 5)),
        'all':     list(range(N)),
    }
    for tag, layers in slices.items():
        slice_one(args.src, args.short, layers, tag)


if __name__ == '__main__':
    main()
