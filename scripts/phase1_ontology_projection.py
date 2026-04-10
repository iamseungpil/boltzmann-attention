#!/usr/bin/env python3
"""
Build an ontology-derived SEKA projector payload.

This is the portable Phase 1.2 builder for the ontology substitution study.
It removes machine-specific path and GPU assumptions so the same command can
run locally or on E8.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from ontology_facet_basis import FACET_ORDER, ONTOLOGY, extract_category_K, build_facet_basis  # type: ignore


def default_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def parse_layers(spec):
    return [int(x) for x in spec.split(',') if x.strip()]


def resolve_dtype(name):
    table = {
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
        'float16': torch.float16,
        'fp16': torch.float16,
        'float32': torch.float32,
        'fp32': torch.float32,
    }
    key = name.lower()
    if key not in table:
        raise ValueError(f'Unsupported dtype: {name}')
    return table[key]


def build_parser():
    parser = argparse.ArgumentParser(
        description='Build ontology-derived projector payload in SEKA format.'
    )
    parser.add_argument('--model-id', default='Qwen/Qwen3-4B-Base')
    parser.add_argument('--short-name', default='Qwen3-4B-Base')
    parser.add_argument('--layers', default='26,27,28,29,30,31,32,33,34,35')
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--device', default=default_device())
    parser.add_argument('--repo-root', type=Path, default=REPO_ROOT)
    parser.add_argument('--out-tag', default='ontology-qwen3-4b')
    parser.add_argument('--diag-json', type=Path, default=None)
    parser.add_argument('--expected-n-layers', type=int, default=36)
    parser.add_argument('--expected-n-kv', type=int, default=8)
    parser.add_argument('--expected-head-dim', type=int, default=128)
    parser.add_argument(
        '--self-test',
        action='store_true',
        help='Validate import/path/output wiring without loading a model.',
    )
    parser.add_argument(
        '--require-external-seka',
        action='store_true',
        help='Fail self-test if external/SEKA is missing.',
    )
    return parser


def main():
    args = build_parser().parse_args()
    layers = parse_layers(args.layers)
    dtype = resolve_dtype(args.dtype)
    out_dir = (
        args.repo_root / 'external' / 'SEKA' / 'seka_projections' / args.out_tag
    )
    diag_json = args.diag_json or (
        args.repo_root / 'reports' / 'axis2_theoretical_verification'
        / f'phase1_{args.out_tag.replace("-", "_")}.json'
    )

    if args.self_test:
        seka_root = args.repo_root / 'external' / 'SEKA'
        seka_repo_ok = (
            (seka_root / 'benchmarks').exists() and (seka_root / 'src').exists()
        )
        print('phase1_ontology_projection self-test')
        print(f'  repo_root      : {args.repo_root}')
        print(f'  scripts_dir     : {SCRIPT_DIR}')
        print(f'  model_id        : {args.model_id}')
        print(f'  short_name      : {args.short_name}')
        print(f'  device          : {args.device}')
        print(f'  layers          : {layers}')
        print(f'  out_dir         : {out_dir}')
        print(f'  diag_json       : {diag_json}')
        print(f'  external_seka   : {seka_repo_ok}')
        if args.require_external_seka and not seka_repo_ok:
            raise SystemExit('external/SEKA is missing')
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    diag_json.parent.mkdir(parents=True, exist_ok=True)

    print('=' * 72)
    print('Phase 1.2 ontology projector build')
    print('=' * 72)
    print(f'  model          : {args.model_id}')
    print(f'  short name     : {args.short_name}')
    print(f'  device         : {args.device}')
    print(f'  target layers  : {layers}')
    print(f'  output dir     : {out_dir}')
    print()

    print('Loading model ...', flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map=args.device if args.device != 'cpu' else None,
        attn_implementation='eager',
        low_cpu_mem_usage=True,
    )
    if args.device == 'cpu':
        model.to(args.device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_q = model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, 'head_dim', None) or (
        model.config.hidden_size // n_q
    )
    print(
        f'  loaded in {time.time()-t0:.1f}s: n_layers={n_layers} '
        f'n_kv={n_kv} head_dim={head_dim}',
        flush=True,
    )
    if args.expected_n_layers and n_layers != args.expected_n_layers:
        raise ValueError(f'Unexpected num_hidden_layers: {n_layers}')
    if args.expected_n_kv and n_kv != args.expected_n_kv:
        raise ValueError(f'Unexpected num_key_value_heads: {n_kv}')
    if args.expected_head_dim and head_dim != args.expected_head_dim:
        raise ValueError(f'Unexpected head_dim: {head_dim}')
    if max(layers) >= n_layers:
        raise ValueError(f'Layer selection {layers} exceeds model depth {n_layers}')

    print('\n[1/3] Extracting per-category K vectors ...', flush=True)
    t0 = time.time()
    cat_k = extract_category_K(
        model, tok, ONTOLOGY, layers, n_kv, head_dim, device=args.device
    )
    print(
        f'  extracted {len(cat_k)} (facet, cat, layer, head) entries '
        f'in {time.time()-t0:.1f}s',
        flush=True,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print('\n[2/3] Building per-(layer, head) basis -> P = B B^T ...', flush=True)
    proj_stack = np.zeros((len(layers), n_kv, head_dim, head_dim), dtype=np.float32)
    r_tot_grid = np.zeros((len(layers), n_kv), dtype=np.int32)
    r_per_facet_grid = {}
    skipped = []

    for li_idx, layer_idx in enumerate(layers):
        for head_idx in range(n_kv):
            e_list = []
            ok = True
            for facet_name in FACET_ORDER:
                cols = []
                for category in ONTOLOGY[facet_name].keys():
                    key = (facet_name, category, layer_idx, head_idx)
                    if key not in cat_k:
                        ok = False
                        break
                    cols.append(cat_k[key])
                if not ok:
                    break
                e_list.append(np.stack(cols, axis=1).astype(np.float64))
            if not ok:
                skipped.append((layer_idx, head_idx, 'missing category'))
                continue

            basis, r_per_facet = build_facet_basis(e_list)
            if basis.shape[1] == 0:
                skipped.append((layer_idx, head_idx, 'zero-rank basis'))
                continue

            orth_err = float(
                np.max(np.abs((basis.T @ basis) - np.eye(basis.shape[1])))
            )
            if orth_err >= 1e-6:
                raise ValueError(f'Orthogonality check failed at L{layer_idx} H{head_idx}')

            projector = (basis @ basis.T).astype(np.float32)
            proj_stack[li_idx, head_idx] = projector
            r_tot_grid[li_idx, head_idx] = int(basis.shape[1])
            r_per_facet_grid[f'L{layer_idx}_H{head_idx}'] = [int(x) for x in r_per_facet]

            print(
                f'  L{layer_idx:2d} H{head_idx}: r_per_facet={r_per_facet} '
                f'r_tot={basis.shape[1]:3d} trace(P)={projector.trace():.3f} '
                f'orth={orth_err:.1e}',
                flush=True,
            )

    valid_ranks = r_tot_grid[r_tot_grid > 0]
    if valid_ranks.size == 0:
        raise RuntimeError('No valid ontology projectors were built')

    print(
        f'\n  rank stats over {valid_ranks.size} valid heads: '
        f'min={valid_ranks.min()} median={int(np.median(valid_ranks))} '
        f'max={valid_ranks.max()} mean={valid_ranks.mean():.1f}',
        flush=True,
    )

    print('\n[3/3] Saving SEKA projector payload ...', flush=True)
    payload = {'layers': layers, 'proj': torch.from_numpy(proj_stack)}
    pos_path = out_dir / f'{args.short_name}_pos_proj.pt'
    neg_path = out_dir / f'{args.short_name}_neg_proj.pt'
    torch.save(payload, pos_path)
    torch.save(payload, neg_path)
    print(f'  wrote {pos_path}')
    print(f'  wrote {neg_path}')

    diag = {
        'model': args.model_id,
        'short_name': args.short_name,
        'target_layers': layers,
        'facet_order': FACET_ORDER,
        'n_kv_heads': n_kv,
        'head_dim': head_dim,
        'device': args.device,
        'ontology_categories': {facet: list(cats.keys()) for facet, cats in ONTOLOGY.items()},
        'r_tot_grid': r_tot_grid.tolist(),
        'r_per_facet': r_per_facet_grid,
        'r_tot_stats': {
            'min': int(valid_ranks.min()),
            'max': int(valid_ranks.max()),
            'median': int(np.median(valid_ranks)),
            'mean': float(valid_ranks.mean()),
        },
        'skipped': skipped,
        'output_files': {'pos': str(pos_path), 'neg': str(neg_path)},
    }
    diag_json.write_text(json.dumps(diag, indent=2, default=float))
    print(f'\nwrote diagnostic {diag_json}')


if __name__ == '__main__':
    main()
