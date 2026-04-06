#!/bin/bash
# Axis2/Axis3 Diagnosis: Mistral-7B-v0.3 (E1 + E4 + E2)
# Estimated runtime: ~2.5 hours on A6000/A100
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/v-seungplee/miniconda3/envs/llm-addiction/bin/python}"

echo "========================================"
echo "Axis2/Axis3 Diagnosis: Mistral-7B-v0.3"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Experiments: E1 (MK) + E4 (Integer WF) + E2 (Sink)"
echo "========================================"

"$PYTHON_BIN" exp_axis2_axis3_diagnosis.py \
    --experiment ALL \
    --model-name mistralai/Mistral-7B-v0.3 \
    --model-key mistral-7b-v0.3 \
    --device cuda:0 \
    --dtype bfloat16 \
    --context-len 2048 \
    --bits 2 3 \
    --gamma 0.3 \
    --attn-implementation eager \
    --mk-calib-tokens 2048 \
    --sink-counts 0 4 16 64 \
    --wf-variants floor1 floor2 no_one uniform \
    --output-dir "$SCRIPT_DIR/../results/diagnosis" \
    "$@" \
    2>&1 | tee "$SCRIPT_DIR/../results/diagnosis/run_diagnosis_mistral.log"
