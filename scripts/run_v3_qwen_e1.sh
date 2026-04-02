#!/bin/bash
source /home/woori/workspace_common/CDP/poc/set.env
cd /home/woori/workspace_common/CDP/poc/docs/architecture/paper/FOKVQ/experiment/4-2_ex

echo "=== Qwen v3+E1 Start: $(date) ==="
python3 exp4_2_v3_full_quant_ppl.py \
  --model-name Qwen/Qwen2.5-7B \
  --model-key qwen2.5-7b \
  --device cuda:0 \
  --dtype bfloat16 \
  --context-len 2048 \
  --methods fp16 uniform kivi fokvq fokvq_qw \
  --bits 2 3 4 \
  --gamma 0.3 \
  --protocol post_rope \
  --attn-implementation eager \
  --output-dir ./results/v3
echo "=== Qwen v3+E1 End: $(date) ==="
