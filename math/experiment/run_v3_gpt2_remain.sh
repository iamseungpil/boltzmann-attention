#!/bin/bash
source /home/woori/workspace_common/CDP/poc/set.env
cd /home/woori/workspace_common/CDP/poc/docs/architecture/paper/FOKVQ/experiment/4-2_ex

echo "=== GPT-2 v3 REMAINING methods Start: $(date) ==="
python3 exp4_2_v3_full_quant_ppl.py \
  --model-name gpt2-medium \
  --model-key gpt2-medium-remain \
  --device cuda:1 \
  --dtype float16 \
  --context-len 1024 \
  --methods quip kvquant gear zipcache turboquant fokvq_e2 fokvq_e3 fokvq_full \
  --bits 2 3 4 \
  --gamma 0.3 \
  --protocol post_rope \
  --attn-implementation eager \
  --output-dir ./results/v3_all
echo "=== GPT-2 v3 REMAINING methods End: $(date) ==="
