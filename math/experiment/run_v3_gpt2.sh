#!/bin/bash
# FOKVQ Exp 4-2 v3: GPT-2 Medium with Full K Quantization
# Protocol: post_rope (quantize K after attention computation, inside attention)
#
# Key difference vs v2:
#   v2: sliding window, only prefix K quantized (~50%)
#   v3: non-overlapping chunks, ALL K quantized via hooks (100%)
#
# Expected runtime: ~60-90 min on A6000 (similar to v2, no overlap)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=1

echo "========================================"
echo "FOKVQ v3: GPT-2 Medium (Full K Quant)"
echo "========================================"

python3 exp4_2_v3_full_quant_ppl.py \
    --model-name gpt2-medium \
    --model-key gpt2-medium \
    --device cuda:0 \
    --dtype float16 \
    --context-len 1024 \
    --protocol post_rope \
    --methods fp16 uniform kivi fokvq \
    --bits 2 3 4 \
    --gamma 0.3 \
    --attn-implementation eager \
    --output-dir "$SCRIPT_DIR/results/v3" \
    2>&1 | tee "$SCRIPT_DIR/run_v3_gpt2.log"
