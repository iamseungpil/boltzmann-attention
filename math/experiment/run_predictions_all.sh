#!/bin/bash
# 4개 모델에 대해 6-벤치마크 예측 실행
# Usage: source set.env && bash docs/architecture/paper/FOKVQ/experiment/4-2_ex/run_predictions_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/predict_benchmarks.py"
DEVICE="${1:-cuda:1}"
OUTPUT_DIR="${SCRIPT_DIR}/results/predictions"

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "6-벤치마크 예측: 4개 모델"
echo "Device: ${DEVICE}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

# KVTC 논문에 대응하는 4개 모델 (모두 GQA + RoPE)
MODELS=(
    "Qwen/Qwen2.5-7B"
    "meta-llama/Llama-3.1-8B"
    "mistralai/Mistral-7B-v0.3"
    "Qwen/Qwen2.5-1.5B"
)

for model in "${MODELS[@]}"; do
    echo ""
    echo "============================================"
    echo "Processing: ${model}"
    echo "============================================"

    python "${SCRIPT}" \
        --model-name "${model}" \
        --device "${DEVICE}" \
        --n-calib-tokens 2000 \
        --bits 2 3 4 \
        --output-dir "${OUTPUT_DIR}" \
        2>&1 | tee "${OUTPUT_DIR}/log_$(echo ${model} | tr '/' '_').txt"

    echo "Done: ${model}"
    echo ""
done

echo "============================================"
echo "모든 모델 예측 완료"
echo "결과: ${OUTPUT_DIR}/"
echo "============================================"
