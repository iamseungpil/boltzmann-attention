#!/usr/bin/env python3
"""
Phase 1.2 — Build an ontology-derived projection tensor in SEKA's format.

Goal: take the per-(layer, head) ontology facet basis B ∈ ℝ^{128×r_tot}
built by scripts/ontology_facet_basis.py machinery, form the symmetric
idempotent projector P = B B^T ∈ ℝ^{128×128}, stack across the last 10
layers and 8 KV heads of Qwen3-4B-Base, and save in the exact .pt format
SEKA's `_load_proj` expects:

    dict(layers=[26, 27, ..., 35], proj=Tensor(10, 8, 128, 128))

Downstream: drop these into SEKA's eval_fact_gen.py via --pos/--neg with
amplify_pos=1.56, amplify_neg=0.0, layers=last10 — identical operator to
the SEKA reproduction from Phase 1.1, substituting ONLY the direction
source (ontology vs contrastive SVD of synthetic QA pairs).

Output: external/SEKA/seka_projections/ontology-qwen3-4b/
          Qwen3-4B-Base_pos_proj.pt  (and identical neg_proj.pt)
"""

import os
import sys
import json
import time
from pathlib import Path

os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
# Phase 1.1 used GPU 1 (cuda:0 physical after CUDA_VISIBLE_DEVICES=1); match.
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Reuse pipeline pieces from the Mistral version.
REPO = Path('/home/woori/workspace_common/boltzmann-attention')
sys.path.insert(0, str(REPO / 'scripts'))
from ontology_facet_basis import (  # type: ignore
    ONTOLOGY,
    FACET_ORDER,
    extract_category_K,
    build_facet_basis,
)

MODEL_ID = 'Qwen/Qwen3-4B-Base'
SHORT = 'Qwen3-4B-Base'
# SEKA Phase 1.1 used layers=last10 → [26..35] on Qwen3-4B-Base (n_layers=36).
TARGET_LAYERS = list(range(26, 36))
DTYPE = torch.bfloat16

OUT_DIR = REPO / 'external' / 'SEKA' / 'seka_projections' / 'ontology-qwen3-4b'
OUT_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR = REPO / 'reports' / 'axis2_theoretical_verification'
DIAG_DIR.mkdir(parents=True, exist_ok=True)
DIAG_JSON = DIAG_DIR / 'phase1_ontology_projection_qwen3_4b.json'


def main():
    print('=' * 72)
    print('Phase 1.2 — ontology projection build for Qwen3-4B-Base')
    print('=' * 72)
    print(f'  model          : {MODEL_ID}')
    print(f'  target layers  : {TARGET_LAYERS}')
    print(f'  output dir     : {OUT_DIR}')
    print()

    print('Loading model …', flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_q = model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, 'head_dim', None) or (
        model.config.hidden_size // n_q
    )
    print(
        f'  loaded in {time.time()-t0:.1f}s: n_layers={n_layers}  '
        f'n_kv={n_kv}  head_dim={head_dim}',
        flush=True,
    )
    assert n_layers == 36 and n_kv == 8 and head_dim == 128, (
        f'Qwen3-4B-Base config mismatch: {n_layers=}, {n_kv=}, {head_dim=}'
    )
    assert max(TARGET_LAYERS) < n_layers

    print('\n[1/3] Extracting per-category K vectors …', flush=True)
    t0 = time.time()
    cat_K = extract_category_K(
        model, tok, ONTOLOGY, TARGET_LAYERS, n_kv, head_dim,
    )
    print(
        f'  extracted {len(cat_K)} (facet, cat, layer, head) entries '
        f'in {time.time()-t0:.1f}s',
        flush=True,
    )

    del model
    torch.cuda.empty_cache()

    print('\n[2/3] Building per-(layer, head) basis → P = B B^T …', flush=True)
    d = head_dim
    proj_stack = np.zeros((len(TARGET_LAYERS), n_kv, d, d), dtype=np.float32)
    r_tot_grid = np.zeros((len(TARGET_LAYERS), n_kv), dtype=np.int32)
    r_per_facet_grid = {}
    skipped = []

    for li_idx, li in enumerate(TARGET_LAYERS):
        for h in range(n_kv):
            E_list = []
            ok = True
            for facet_name in FACET_ORDER:
                cats = list(ONTOLOGY[facet_name].keys())
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
                skipped.append((li, h, 'missing category'))
                continue

            B, r_per_facet = build_facet_basis(E_list)
            if B.shape[1] == 0:
                skipped.append((li, h, 'zero-rank basis'))
                continue

            # orthogonality sanity
            BtB = B.T @ B
            ortho_err = float(np.max(np.abs(BtB - np.eye(B.shape[1]))))
            assert ortho_err < 1e-6, f'L{li}H{h} ortho_err={ortho_err:.2e}'

            P = (B @ B.T).astype(np.float32)
            proj_stack[li_idx, h] = P
            r_tot_grid[li_idx, h] = int(B.shape[1])
            r_per_facet_grid[f'L{li}_H{h}'] = [int(x) for x in r_per_facet]

            print(
                f'  L{li:2d} H{h}: r_per_facet={r_per_facet}  '
                f'r_tot={B.shape[1]:3d}  trace(P)={P.trace():.3f}  '
                f'ortho={ortho_err:.1e}',
                flush=True,
            )

    if skipped:
        print(f'\n  SKIPPED {len(skipped)} pairs: {skipped[:5]} …')

    valid_r = r_tot_grid[r_tot_grid > 0]
    print(
        f'\n  r_tot stats over {valid_r.size} valid heads: '
        f'min={valid_r.min()} median={int(np.median(valid_r))} '
        f'max={valid_r.max()} mean={valid_r.mean():.1f}',
        flush=True,
    )

    print('\n[3/3] Saving projection in SEKA .pt format …', flush=True)
    proj_tensor = torch.from_numpy(proj_stack)  # (10, 8, 128, 128) float32
    payload = {
        'layers': list(TARGET_LAYERS),
        'proj': proj_tensor,
    }

    pos_path = OUT_DIR / f'{SHORT}_pos_proj.pt'
    neg_path = OUT_DIR / f'{SHORT}_neg_proj.pt'
    torch.save(payload, pos_path)
    # Same file for neg — SEKA hook will compute delta = (1.56*pos + 0*neg)/2
    # which halves to 0.78*P·k regardless of what neg is, as long as
    # amplify_neg=0.  We save a copy so the hook loads a valid 4-D tensor.
    torch.save(payload, neg_path)
    print(f'  wrote {pos_path}')
    print(f'  wrote {neg_path}')

    diag = {
        'model': MODEL_ID,
        'short_name': SHORT,
        'target_layers': TARGET_LAYERS,
        'facet_order': FACET_ORDER,
        'n_kv_heads': n_kv,
        'head_dim': d,
        'ontology_categories': {f: list(c.keys()) for f, c in ONTOLOGY.items()},
        'r_tot_grid': r_tot_grid.tolist(),
        'r_per_facet': r_per_facet_grid,
        'r_tot_stats': {
            'min': int(valid_r.min()) if valid_r.size else 0,
            'max': int(valid_r.max()) if valid_r.size else 0,
            'median': int(np.median(valid_r)) if valid_r.size else 0,
            'mean': float(valid_r.mean()) if valid_r.size else 0.0,
        },
        'skipped': skipped,
        'output_files': {
            'pos': str(pos_path),
            'neg': str(neg_path),
        },
    }
    DIAG_JSON.write_text(json.dumps(diag, indent=2, default=float))
    print(f'\nwrote diagnostic {DIAG_JSON}')


if __name__ == '__main__':
    main()
