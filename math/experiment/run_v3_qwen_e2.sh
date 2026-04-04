#!/bin/bash
# FOKVQ Exp 4-2 v3: Qwen2.5-7B — E2 only (main contribution)
# E2: K-only PCA + per-axis Lloyd-Max adaptive codebook
# E1 excluded: GQA Q-head averaging collapse
# E3 excluded: mathematically equivalent to E2 (constant 1D scaling is no-op for Lloyd-Max)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=1

echo "========================================"
echo "FOKVQ v3: Qwen2.5-7B (E2 only)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================"

python3 exp4_2_v3_full_quant_ppl.py \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --model-key qwen2.5-7b \
    --device cuda:0 \
    --dtype bfloat16 \
    --context-len 2048 \
    --protocol post_rope \
    --methods fokvq_e2 \
    --bits 2 3 4 \
    --gamma 0.3 \
    --attn-implementation eager \
    --output-dir "$SCRIPT_DIR/results/v3" \
    2>&1 | tee "$SCRIPT_DIR/run_v3_qwen_e2.log"
