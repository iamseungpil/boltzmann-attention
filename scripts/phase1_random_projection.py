#!/usr/bin/env python3
"""
Phase 1.4 — Random orthonormal rank-8 baseline.

Critical ablation for the ontology paper: build a per-(layer, head)
random orthonormal rank-8 projection matched in *every* dimension
(same shape, same rank, same layer set, same SEKA hook) to the
Phase 1.3 rank-8 ontology projection, but with **no ontology content
whatsoever**. Each (layer, head) gets an independent random 128×8
orthonormal basis drawn via QR of a Gaussian matrix, then
P = B B^T.

If CounterFact/BiasBios performance is comparable to the ontology
variant, then the ontology content was incidental and the Path A
paper must pivot. If the random baseline is meaningfully worse, the
ontology direction content is load-bearing.

Seed is fixed (42) for reproducibility. Output in SEKA .pt dict
format, stored alongside the ontology projections.
"""

import os
import json
from pathlib import Path

os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

import numpy as np
import torch

REPO = Path('/home/woori/workspace_common/boltzmann-attention')
SHORT = 'Qwen3-4B-Base'
TARGET_LAYERS = list(range(26, 36))
N_KV = 8
HEAD_DIM = 128
RANK = 8
SEED = 42

OUT_DIR = REPO / 'external' / 'SEKA' / 'seka_projections' / 'random-rank8-qwen3-4b'
OUT_DIR.mkdir(parents=True, exist_ok=True)
DIAG = REPO / 'reports' / 'axis2_theoretical_verification' / 'phase1_random_projection_qwen3_4b.json'


def main():
    print('=' * 72)
    print(f'Phase 1.4 random orthonormal rank-{RANK} baseline')
    print('=' * 72)
    print(f'  shape        : ({len(TARGET_LAYERS)}, {N_KV}, {HEAD_DIM}, {HEAD_DIM})')
    print(f'  per-head rank: {RANK}')
    print(f'  seed         : {SEED}')
    print()

    rng = np.random.default_rng(SEED)
    proj = np.zeros(
        (len(TARGET_LAYERS), N_KV, HEAD_DIM, HEAD_DIM), dtype=np.float32
    )
    max_ortho_err = 0.0
    max_trace_err = 0.0

    for li_idx in range(len(TARGET_LAYERS)):
        for h in range(N_KV):
            # Draw a random 128×8 Gaussian, QR to get orthonormal columns
            G = rng.standard_normal((HEAD_DIM, RANK)).astype(np.float64)
            Q, _ = np.linalg.qr(G)      # Q: (128, 8), orthonormal
            P = Q @ Q.T                  # (128, 128), symmetric, rank-8
            proj[li_idx, h] = P.astype(np.float32)

            ortho_err = float(np.max(np.abs(Q.T @ Q - np.eye(RANK))))
            trace_err = abs(float(P.trace()) - RANK)
            max_ortho_err = max(max_ortho_err, ortho_err)
            max_trace_err = max(max_trace_err, trace_err)

    print(f'  built 80 per-head random projections')
    print(f'  max ortho err : {max_ortho_err:.2e}')
    print(f'  max trace err : {max_trace_err:.2e}')
    assert max_ortho_err < 1e-10
    assert max_trace_err < 1e-4

    proj_tensor = torch.from_numpy(proj)
    payload = {'layers': list(TARGET_LAYERS), 'proj': proj_tensor}
    pos_path = OUT_DIR / f'{SHORT}_pos_proj.pt'
    neg_path = OUT_DIR / f'{SHORT}_neg_proj.pt'
    torch.save(payload, pos_path)
    torch.save(payload, neg_path)
    print(f'\n  wrote {pos_path}')
    print(f'  wrote {neg_path}')

    DIAG.write_text(json.dumps({
        'target_layers': TARGET_LAYERS,
        'n_kv': N_KV,
        'head_dim': HEAD_DIM,
        'rank': RANK,
        'seed': SEED,
        'max_ortho_err': max_ortho_err,
        'max_trace_err': max_trace_err,
        'output_files': {'pos': str(pos_path), 'neg': str(neg_path)},
    }, indent=2))
    print(f'  wrote diagnostic {DIAG}')


if __name__ == '__main__':
    main()
