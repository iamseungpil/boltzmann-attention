#!/bin/bash
# FOKVQ Exp 4-2 v3: Qwen2.5-7B with Full K Quantization
# Protocol: post_rope (quantize K after RoPE, inside attention)
#
# Key difference vs v2:
#   v2: sliding window, only prefix K quantized (~50%)
#   v3: non-overlapping chunks, ALL K quantized via hooks (100%)
#
# Expected runtime: ~180-240 min on A6000

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=1

echo "========================================"
echo "FOKVQ v3: Qwen2.5-7B (Full K Quant)"
echo "========================================"

python3 exp4_2_v3_full_quant_ppl.py \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --model-key qwen2.5-7b \
    --device cuda:0 \
    --dtype bfloat16 \
    --context-len 2048 \
    --protocol post_rope \
    --methods fp16 uniform kivi fokvq \
    --bits 2 3 4 \
    --gamma 0.3 \
    --attn-implementation eager \
    --output-dir "$SCRIPT_DIR/results/v3" \
    2>&1 | tee "$SCRIPT_DIR/run_v3_qwen.log"
