#!/bin/bash
# Qwen rotation-mechanistic smoke/full launcher
#
# Intent:
#   Test whether structured bases beat agnostic controls in the same harness.
# Hypothesis:
#   identity < random < fokvq / structured Lie candidates in the stress regime.
# Verification:
#   rotation_mechanistic preset on WikiText-2 PPL with explicit axis controls.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PYTHON_BIN="${PYTHON_BIN:-/home/v-seungplee/miniconda3/envs/llm-addiction/bin/python}"

echo "========================================"
echo "FOKVQ v3: Qwen2.5-7B (Rotation Mechanistic)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================"

"$PYTHON_BIN" exp4_2_v3_full_quant_ppl.py \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --model-key qwen2.5-7b-rotation \
    --device cuda:0 \
    --dtype bfloat16 \
    --context-len 2048 \
    --protocol post_rope \
    --benchmark-preset rotation_mechanistic \
    --methods fp16 identity random fokvq fokvq_e2 kivi_residual turboquant_rand complex_unitary_residual banded_complex_unitary_residual \
    --bits 2 3 4 \
    --gamma 0.3 \
    --attn-implementation eager \
    --output-dir "$REPO_ROOT/results/v3" \
    "$@" \
    2>&1 | tee "$REPO_ROOT/reports/run_v3_qwen_rotation.log"
