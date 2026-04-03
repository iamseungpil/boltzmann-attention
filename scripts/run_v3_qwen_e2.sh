#!/bin/bash
# Qwen practical comparison focused on the E2 family
#
# Intent:
#   Test whether the E2 quantizer improves practical quality in the same harness.
# Hypothesis:
#   fokvq_e2 should improve over plain fokvq, but must be compared against the
#   stronger recent-tail practical controls.
# Verification:
#   ppl_quality-style run with explicit E2-centered method list.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PYTHON_BIN="${PYTHON_BIN:-/home/v-seungplee/miniconda3/envs/llm-addiction/bin/python}"

echo "========================================"
echo "FOKVQ v3: Qwen2.5-7B (E2 only)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================"

"$PYTHON_BIN" exp4_2_v3_full_quant_ppl.py \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --model-key qwen2.5-7b-e2 \
    --device cuda:0 \
    --dtype bfloat16 \
    --context-len 2048 \
    --protocol post_rope \
    --benchmark-preset ppl_quality \
    --methods fp16 kivi_residual turboquant_rand fokvq fokvq_e2 fokvq_e2_residual \
    --bits 2 3 4 \
    --gamma 0.3 \
    --attn-implementation eager \
    --output-dir "$REPO_ROOT/results/v3" \
    "$@" \
    2>&1 | tee "$REPO_ROOT/reports/run_v3_qwen_e2.log"
