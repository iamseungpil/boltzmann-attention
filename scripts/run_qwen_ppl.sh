#!/bin/bash
source /home/woori/workspace_common/CDP/poc/set.env
cd /home/woori/workspace_common/CDP/poc/docs/architecture/paper/FOKVQ/experiment/4-2_ex

echo "=== Start: $(date) ==="
python3 exp4_2_standard_ppl_benchmark_v2.py \
  --model-name Qwen/Qwen2.5-7B \
  --model-key qwen2.5-7b \
  --device cuda:0 \
  --dtype bfloat16 \
  --context-len 2048 \
  --stride 1024 \
  --methods fp16 uniform kivi fokvq \
  --bits 2 3 4 \
  --gamma 0.3 \
  --output-dir ./results
echo "=== End: $(date) ==="
