#!/bin/bash
source /home/woori/workspace_common/CDP/poc/set.env
cd /home/woori/workspace_common/CDP/poc/docs/architecture/paper/FOKVQ/experiment/4-2_ex

echo "=== Qwen v3 FOKVQ-first Start: $(date) ==="
python3 exp4_2_v3_full_quant_ppl.py \
  --model-name Qwen/Qwen2.5-7B \
  --model-key qwen2.5-7b-fokvq \
  --device cuda:0 \
  --dtype bfloat16 \
  --context-len 2048 \
  --methods fokvq fokvq_qw fokvq_e2 fokvq_e3 fokvq_full \
  --bits 2 3 4 \
  --gamma 0.3 \
  --protocol post_rope \
  --attn-implementation eager \
  --output-dir ./results/v3_all
echo "=== Qwen v3 FOKVQ-first End: $(date) ==="

echo "=== Qwen v3 SOTA-remain Start: $(date) ==="
python3 exp4_2_v3_full_quant_ppl.py \
  --model-name Qwen/Qwen2.5-7B \
  --model-key qwen2.5-7b-sota \
  --device cuda:0 \
  --dtype bfloat16 \
  --context-len 2048 \
  --methods kvquant gear zipcache turboquant \
  --bits 2 3 4 \
  --gamma 0.3 \
  --protocol post_rope \
  --attn-implementation eager \
  --output-dir ./results/v3_all
echo "=== Qwen v3 SOTA-remain End: $(date) ==="
