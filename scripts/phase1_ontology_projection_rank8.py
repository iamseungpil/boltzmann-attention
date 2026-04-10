#!/usr/bin/env python3
"""
Build a rank-truncated ontology SEKA projector payload.

This is the portable Phase 1.3 builder. It matches the ontology study's
rank-truncation logic while removing machine-specific path and GPU bindings.
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

from ontology_facet_basis import FACET_ORDER, ONTOLOGY, build_facet_basis, compute_per_head_sigma, extract_category_K  # type: ignore


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
        description='Build rank-truncated ontology projector payload in SEKA format.'
    )
    parser.add_argument('--model-id', default='Qwen/Qwen3-4B-Base')
    parser.add_argument('--short-name', default='Qwen3-4B-Base')
    parser.add_argument('--layers', default='26,27,28,29,30,31,32,33,34,35')
    parser.add_argument('--target-rank', type=int, default=8)
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--device', default=default_device())
    parser.add_argument('--repo-root', type=Path, default=REPO_ROOT)
    parser.add_argument('--out-tag', default='ontology-qwen3-4b-rank8')
    parser.add_argument('--diag-json', type=Path, default=None)
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
        print('phase1_ontology_projection_rank8 self-test')
        print(f'  repo_root      : {args.repo_root}')
        print(f'  model_id        : {args.model_id}')
        print(f'  short_name      : {args.short_name}')
        print(f'  device          : {args.device}')
        print(f'  layers          : {layers}')
        print(f'  target_rank     : {args.target_rank}')
        print(f'  out_dir         : {out_dir}')
        print(f'  diag_json       : {diag_json}')
        print(f'  external_seka   : {seka_repo_ok}')
        if args.require_external_seka and not seka_repo_ok:
            raise SystemExit('external/SEKA is missing')
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    diag_json.parent.mkdir(parents=True, exist_ok=True)

    print('=' * 72)
    print(f'Phase 1.3 ontology rank-{args.target_rank} projector build')
    print('=' * 72)
    print(f'  model          : {args.model_id}')
    print(f'  short name     : {args.short_name}')
    print(f'  device         : {args.device}')
    print(f'  target layers  : {layers}')
    print(f'  target rank    : {args.target_rank}')
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
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    head_dim = getattr(model.config, 'head_dim', None) or (
        model.config.hidden_size // n_q
    )
    print(
        f'  loaded in {time.time()-t0:.1f}s: n_kv={n_kv} head_dim={head_dim}',
        flush=True,
    )
    if args.expected_n_kv and n_kv != args.expected_n_kv:
        raise ValueError(f'Unexpected num_key_value_heads: {n_kv}')
    if args.expected_head_dim and head_dim != args.expected_head_dim:
        raise ValueError(f'Unexpected head_dim: {head_dim}')

    print('\n[1/3] Extract ontology K vectors ...', flush=True)
    t0 = time.time()
    cat_k = extract_category_K(
        model, tok, ONTOLOGY, layers, n_kv, head_dim, device=args.device
    )
    print(f'  got {len(cat_k)} entries in {time.time()-t0:.1f}s', flush=True)

    print('\n[2/3] Compute per-head Sigma_K ...', flush=True)
    t0 = time.time()
    sigmas = compute_per_head_sigma(
        model, tok, layers, n_kv, head_dim, device=args.device
    )
    print(f'  got {len(sigmas)} Sigma matrices in {time.time()-t0:.1f}s', flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f'\n[3/3] Build rank-{args.target_rank} projectors ...', flush=True)
    proj_stack = np.zeros((len(layers), n_kv, head_dim, head_dim), dtype=np.float32)
    kept_var_frac = np.zeros((len(layers), n_kv), dtype=np.float64)
    r_orig_grid = np.zeros((len(layers), n_kv), dtype=np.int32)
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

            basis, _ = build_facet_basis(e_list)
            if basis.shape[1] < args.target_rank:
                skipped.append((layer_idx, head_idx, f'r_tot={basis.shape[1]}'))
                continue
            r_orig_grid[li_idx, head_idx] = basis.shape[1]

            sigma = sigmas[(layer_idx, head_idx)]
            sigma = 0.5 * (sigma + sigma.T)
            subspace_cov = basis.T @ sigma @ basis
            subspace_cov = 0.5 * (subspace_cov + subspace_cov.T)
            eigvals, eigvecs = np.linalg.eigh(subspace_cov)
            order = np.argsort(-eigvals)
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            truncated = basis @ eigvecs[:, :args.target_rank]
            orth_err = float(
                np.max(np.abs((truncated.T @ truncated) - np.eye(args.target_rank)))
            )
            if orth_err >= 1e-6:
                raise ValueError(f'Orthogonality check failed at L{layer_idx} H{head_idx}')

            projector = (truncated @ truncated.T).astype(np.float32)
            proj_stack[li_idx, head_idx] = projector

            total_var = float(np.sum(np.maximum(eigvals, 0.0)))
            kept_var = float(np.sum(np.maximum(eigvals[:args.target_rank], 0.0)))
            kept_var_frac[li_idx, head_idx] = (
                kept_var / total_var if total_var > 0 else 0.0
            )

            print(
                f'  L{layer_idx:2d} H{head_idx}: r_orig={basis.shape[1]:2d} '
                f'kept_var={kept_var_frac[li_idx, head_idx]:.4f} '
                f'trace(P)={projector.trace():.3f} orth={orth_err:.1e}',
                flush=True,
            )

    valid = kept_var_frac > 0
    if not valid.any():
        raise RuntimeError('No valid rank-truncated ontology projectors were built')

    print(
        f'\n  kept_var_frac over {int(valid.sum())} valid heads: '
        f'min={kept_var_frac[valid].min():.4f} '
        f'median={np.median(kept_var_frac[valid]):.4f} '
        f'max={kept_var_frac[valid].max():.4f}',
        flush=True,
    )

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
        'target_rank': args.target_rank,
        'device': args.device,
        'n_heads': int(valid.sum()),
        'r_orig_grid': r_orig_grid.tolist(),
        'kept_var_frac_grid': kept_var_frac.tolist(),
        'kept_var_frac_stats': {
            'min': float(kept_var_frac[valid].min()),
            'median': float(np.median(kept_var_frac[valid])),
            'max': float(kept_var_frac[valid].max()),
            'mean': float(kept_var_frac[valid].mean()),
        },
        'skipped': skipped,
        'output_files': {'pos': str(pos_path), 'neg': str(neg_path)},
    }
    diag_json.write_text(json.dumps(diag, indent=2, default=float))
    print(f'\nwrote diagnostic {diag_json}')


if __name__ == '__main__':
    main()
