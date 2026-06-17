#!/bin/bash
# Overnight OpenRouter width ladder (no GPU): broad scale/family S*(width) curve + decomposition
# sufficiency on the SMALL models. Sequential (paces cost + rate limits), robust (each model isolated,
# failure -> next). Decomp arm only on small/cheap models (where width-offload matters; ~3x calls).
#   Per model: width_eval widths 1-5, n=60, gloss=1. Results -> depth/c8/width/width_<tag>.json
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
PY=/home/woori/venvs/seka_env/bin/python; MA=$REPO/scripts/distill/ma
OUT=/home/woori/scratch/depth/c8/width; KEY=/home/woori/.openrouter_key
OR=https://openrouter.ai/api/v1
mkdir -p $OUT; LOG=$OUT/overnight_or.log; exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --ff-only 2>&1 | tail -1

# model : tag : arms   (Decomp = per-attr offload; only on small/cheap models)
SPECS=(
  "meta-llama/llama-3.1-8b-instruct:llama8b:A,B,Decomp"
  "openai/gpt-4o-mini:gpt4omini:A,B,Decomp"
  "qwen/qwen-2.5-7b-instruct:qwen7b_or:A,B,Decomp"
  "mistralai/mistral-7b-instruct:mistral7b:A,B,Decomp"
  "openai/gpt-4.1-mini:gpt41mini:A,B"
  "meta-llama/llama-3.3-70b-instruct:llama70b:A,B"
  "qwen/qwen-2.5-72b-instruct:qwen72b:A,B"
)
for S in "${SPECS[@]}"; do
  M="${S%%:*}"; rest="${S#*:}"; T="${rest%%:*}"; ARMS="${rest#*:}"
  echo "##### OR $T ($M) arms=$ARMS #####"; date
  $PY $MA/width_eval.py --base $OR --model "$M" --api_key_file $KEY --tag $T \
    --widths 1,2,3,4,5 --n 60 --arms "$ARMS" --gloss 1 --out $OUT/width_$T.json || echo "FAIL $T"
done
echo "WIDTH_OVERNIGHT_OR_DONE"; date