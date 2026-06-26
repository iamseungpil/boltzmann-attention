#!/bin/bash
# facet (3) eval 재실행 (학습 완료·adapter 존재·grep 버그로 eval만 실패 → eval만 재실행).
# base floor + trained native op-naming on synth held-out (252).
# Usage: facet3_eval_rerun.sh <gpu> <port>
set -u
GPU=${1:?gpu}; PORT=${2:?port}
R=/home/woori/workspace_common/boltzmann-attention-pi
MA=$R/scripts/distill/ma
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
BASE=Qwen/Qwen2.5-7B-Instruct
S=/home/woori/scratch
OUT=$S/depth/c8/facet3
TAG=facet3_native_ep1
HO=$OUT/heldout_native.jsonl
LOG=$OUT/logs/eval_rerun_g${GPU}.log
exec > $LOG 2>&1; set -x; date

serve(){ # $1=served-id $2=lora(empty=base)
  for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
  if [ -n "$2" ]; then
    CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT --max-model-len 8192 \
      --enable-auto-tool-choice --tool-call-parser hermes --enable-lora --max-lora-rank 64 \
      --lora-modules ${1}=${2} > $OUT/logs/serve_${1}.log 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT --max-model-len 8192 \
      --enable-auto-tool-choice --tool-call-parser hermes > $OUT/logs/serve_base2.log 2>&1 &
  fi
  for i in $(seq 1 70); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -qE "${1}|Qwen2.5-7B" && return 0; sleep 10; done
  echo "SERVE_FAIL $1"; tail -20 $OUT/logs/serve_${1}.log; return 1
}

# base floor
if serve base ""; then
  $PY $MA/synth_native_eval.py --data $HO --base http://localhost:$PORT/v1 --model "$BASE" \
    --out $OUT/results/base_heldout.json 2>&1 | sed 's/^/[base] /' || echo BASE_EVAL_FAIL
fi
# trained
if serve $TAG "$S/adapters/$TAG"; then
  $PY $MA/synth_native_eval.py --data $HO --base http://localhost:$PORT/v1 --model "$TAG" \
    --out $OUT/results/trained_heldout.json 2>&1 | sed 's/^/[trained] /' || echo TRAINED_EVAL_FAIL
fi

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "=== FACET3 EVAL RESULT (native op-naming·held-out 252·§21 op-IR 1.00 재현?) ==="
for f in base_heldout trained_heldout; do echo "--- $f ---"; $PY -c "import json;d=json.load(open('$OUT/results/$f.json'));print('overall',d['overall']);print('by_op',{k:v['recognition'] for k,v in d['by_op'].items()})" 2>/dev/null || echo NA; done
date; echo FACET3_EVAL_DONE
