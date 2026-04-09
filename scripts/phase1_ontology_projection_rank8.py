#!/usr/bin/env python3
"""
Phase 1.3 — Rank-8 truncation of the ontology facet basis.

Motivation: SEKA's contrastive-SVD projector is effectively rank-8 per
head (trace ~8, from top_pct=0.90 energy truncation of singular values).
Our Phase 1.2 ontology projector is rank 12–14. The Phase 1.2 / α-sweep
finding — that ontology beats SEKA at α=3.0 — is interesting but
conflates two factors: (a) direction *content* (facet-derived vs
contrastive-SVD) and (b) projector *rank*. This script isolates (a) by
rank-matching: truncate each per-(layer, head) ontology basis from
rank ~13 down to rank 8 using a Σ_K-weighted criterion.

Procedure per (L, H):

  1. Build B₁₃ ∈ ℝ^{128 × r_tot} using the Phase 1.2 pipeline.
  2. Compute Σ_K from WikiText-2 content tokens (same rule as
     ontology_facet_basis.py, excluding BOS).
  3. Form M = Bᵀ Σ_K B  (shape r_tot × r_tot).
  4. Symmetric eigendecomposition: M = V Λ Vᵀ, sort by −λ.
  5. Keep V[:, :8] — the 8 orthonormal directions **inside the
     ontology subspace** that carry the most K-space variance under
     Σ_K.
  6. B₈ = B @ V[:, :8]   (128 × 8, orthonormal by construction since
     B has orthonormal columns and V is orthonormal).
  7. P₈ = B₈ B₈ᵀ — symmetric idempotent rank-8 projector.

Output: same SEKA .pt dict format as Phase 1.2, saved under
  external/SEKA/seka_projections/ontology-qwen3-4b-rank8/
"""

import os
import sys
import json
import time
from pathlib import Path

os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO = Path('/home/woori/workspace_common/boltzmann-attention')
sys.path.insert(0, str(REPO / 'scripts'))
from ontology_facet_basis import (  # type: ignore
    ONTOLOGY,
    FACET_ORDER,
    extract_category_K,
    build_facet_basis,
    compute_per_head_sigma,
)

MODEL_ID = 'Qwen/Qwen3-4B-Base'
SHORT = 'Qwen3-4B-Base'
TARGET_LAYERS = list(range(26, 36))
TARGET_RANK = 8
DTYPE = torch.bfloat16

OUT_DIR = REPO / 'external' / 'SEKA' / 'seka_projections' / 'ontology-qwen3-4b-rank8'
OUT_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR = REPO / 'reports' / 'axis2_theoretical_verification'
DIAG_JSON = DIAG_DIR / 'phase1_ontology_projection_qwen3_4b_rank8.json'


def main():
    print('=' * 72)
    print(f'Phase 1.3 rank-{TARGET_RANK} truncation of ontology basis')
    print('=' * 72)
    print(f'  model         : {MODEL_ID}')
    print(f'  target layers : {TARGET_LAYERS}')
    print(f'  target rank   : {TARGET_RANK}')
    print()

    print('Loading model …', flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    head_dim = getattr(model.config, 'head_dim', None) or (
        model.config.hidden_size // n_q
    )
    print(f'  n_kv={n_kv} head_dim={head_dim} ({time.time()-t0:.1f}s)',
          flush=True)
    assert n_kv == 8 and head_dim == 128

    print('\n[1/3] Extract ontology K per (facet, cat, layer, head) …',
          flush=True)
    t0 = time.time()
    cat_K = extract_category_K(
        model, tok, ONTOLOGY, TARGET_LAYERS, n_kv, head_dim,
    )
    print(f'  got {len(cat_K)} entries ({time.time()-t0:.1f}s)', flush=True)

    print('\n[2/3] Compute per-head Σ_K on WikiText-2 …', flush=True)
    t0 = time.time()
    sigmas = compute_per_head_sigma(model, tok, TARGET_LAYERS, n_kv, head_dim)
    print(f'  got {len(sigmas)} Σ ({time.time()-t0:.1f}s)', flush=True)

    del model
    torch.cuda.empty_cache()

    print(f'\n[3/3] Build rank-{TARGET_RANK} ontology projectors …',
          flush=True)
    d = head_dim
    proj_stack = np.zeros(
        (len(TARGET_LAYERS), n_kv, d, d), dtype=np.float32
    )
    kept_var_frac = np.zeros(
        (len(TARGET_LAYERS), n_kv), dtype=np.float64
    )
    r_orig_grid = np.zeros((len(TARGET_LAYERS), n_kv), dtype=np.int32)
    skipped = []

    for li_idx, li in enumerate(TARGET_LAYERS):
        for h in range(n_kv):
            # Build the full ontology basis
            E_list = []
            ok = True
            for facet_name in FACET_ORDER:
                cols = []
                for c in ONTOLOGY[facet_name].keys():
                    key = (facet_name, c, li, h)
                    if key not in cat_K:
                        ok = False
                        break
                    cols.append(cat_K[key])
                if not ok:
                    break
                E_list.append(np.stack(cols, axis=1).astype(np.float64))
            if not ok:
                skipped.append((li, h, 'missing cat'))
                continue

            B, _ = build_facet_basis(E_list)
            if B.shape[1] < TARGET_RANK:
                skipped.append((li, h, f'r_tot={B.shape[1]} < {TARGET_RANK}'))
                continue
            r_orig_grid[li_idx, h] = B.shape[1]

            # Σ_K-weighted truncation within the ontology subspace
            Sigma = sigmas[(li, h)]
            Sigma = 0.5 * (Sigma + Sigma.T)
            M = B.T @ Sigma @ B          # (r_tot, r_tot)
            M = 0.5 * (M + M.T)
            eigvals, eigvecs = np.linalg.eigh(M)
            order = np.argsort(-eigvals)
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            V8 = eigvecs[:, :TARGET_RANK]
            B8 = B @ V8                  # (d, TARGET_RANK)

            # orthogonality sanity (should be identity since both
            # orthonormal factors)
            ortho_err = float(
                np.max(np.abs(B8.T @ B8 - np.eye(TARGET_RANK)))
            )
            assert ortho_err < 1e-6, f'L{li}H{h} ortho_err={ortho_err}'

            P8 = (B8 @ B8.T).astype(np.float32)
            proj_stack[li_idx, h] = P8

            total_var = float(np.sum(np.maximum(eigvals, 0.0)))
            kept_var = float(np.sum(np.maximum(eigvals[:TARGET_RANK], 0.0)))
            kept_var_frac[li_idx, h] = (
                kept_var / total_var if total_var > 0 else 0.0
            )

            print(
                f'  L{li:2d} H{h}: r_orig={B.shape[1]:2d} → r8  '
                f'kept_var={kept_var_frac[li_idx, h]:.4f}  '
                f'trace(P8)={P8.trace():.3f}  ortho={ortho_err:.1e}',
                flush=True,
            )

    valid = kept_var_frac > 0
    print(
        f'\n  kept_var_frac over {int(valid.sum())} valid heads: '
        f'min={kept_var_frac[valid].min():.4f}  '
        f'median={np.median(kept_var_frac[valid]):.4f}  '
        f'max={kept_var_frac[valid].max():.4f}',
        flush=True,
    )
    print(
        f'  r_orig: min={r_orig_grid[valid].min()}  '
        f'median={int(np.median(r_orig_grid[valid]))}  '
        f'max={r_orig_grid[valid].max()}',
        flush=True,
    )

    print('\nSaving rank-8 projection .pt …', flush=True)
    proj_tensor = torch.from_numpy(proj_stack)
    payload = {'layers': list(TARGET_LAYERS), 'proj': proj_tensor}
    pos_path = OUT_DIR / f'{SHORT}_pos_proj.pt'
    neg_path = OUT_DIR / f'{SHORT}_neg_proj.pt'
    torch.save(payload, pos_path)
    torch.save(payload, neg_path)
    print(f'  wrote {pos_path}')
    print(f'  wrote {neg_path}')

    diag = {
        'model': MODEL_ID,
        'target_layers': TARGET_LAYERS,
        'target_rank': TARGET_RANK,
        'n_heads': int(valid.sum()),
        'r_orig_grid': r_orig_grid.tolist(),
        'kept_var_frac_grid': kept_var_frac.tolist(),
        'kept_var_frac_stats': {
            'min': float(kept_var_frac[valid].min()) if valid.any() else 0.0,
            'median': float(np.median(kept_var_frac[valid])) if valid.any() else 0.0,
            'max': float(kept_var_frac[valid].max()) if valid.any() else 0.0,
            'mean': float(kept_var_frac[valid].mean()) if valid.any() else 0.0,
        },
        'skipped': skipped,
        'output_files': {'pos': str(pos_path), 'neg': str(neg_path)},
    }
    DIAG_JSON.write_text(json.dumps(diag, indent=2, default=float))
    print(f'\nwrote diagnostic {DIAG_JSON}')


if __name__ == '__main__':
    main()
