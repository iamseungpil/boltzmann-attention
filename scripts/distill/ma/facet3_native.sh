#!/bin/bash
# facet (3) keystone phase-1 — synth content-op routing을 NATIVE 포맷서 학습·전이 확정.
# §23E 다리: op-IR(MD_route)이 native agent 깨뜨림 → resolve_selection native tool_call로 학습.
# 측정: synth held-out(새 어휘) op-명명 = §21(op-IR 1.00)을 native로 재현하나. base floor 포함.
# (retail/airline τ² 전이 eval = phase-2 별도.)
# Usage: facet3_native.sh <gpu> <port> [n_per_op]
set -u
GPU=${1:?gpu}; PORT=${2:?port}; NPO=${3:-860}
R=/home/woori/workspace_common/boltzmann-attention-pi
MA=$R/scripts/distill/ma
DIST=$R/scripts/distill
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
BASE=Qwen/Qwen2.5-7B-Instruct
S=/home/woori/scratch
OUT=$S/depth/c8/facet3; mkdir -p $OUT/logs $OUT/results $S/adapters
TAG=facet3_native_ep1
LOG=$OUT/logs/run_g${GPU}.log
exec > $LOG 2>&1; set -x; date

serve(){ # $1=served-id $2=lora(empty=base)
  for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
  if [ -n "$2" ]; then
    CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT --max-model-len 8192 \
      --enable-auto-tool-choice --tool-call-parser hermes --enable-lora --max-lora-rank 64 \
      --lora-modules ${1}=${2} > $OUT/logs/serve_${1}.log 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT --max-model-len 8192 \
      --enable-auto-tool-choice --tool-call-parser hermes > $OUT/logs/serve_base.log 2>&1 &
  fi
  # base는 모델명이 Qwen2.5-7B-Instruct(서빙id "base" 아님) → 둘 다 매치
  for i in $(seq 1 70); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -qE "${1}|Qwen2.5-7B" && return 0; sleep 10; done
  echo "SERVE_FAIL $1"; tail -20 $OUT/logs/serve_${1}.log; return 1
}

# 1) data: native train(7-op diverse) + held-out(다른 seed=새 어휘)
$PY $MA/synth_to_nativefc.py --out $OUT/route_native.jsonl    --n_per_op $NPO --N 5,10,20 --diverse --seed 0
$PY $MA/synth_to_nativefc.py --out $OUT/heldout_native.jsonl  --n_per_op 36  --N 5,10,20 --diverse --seed 91237
NTR=$(wc -l < $OUT/route_native.jsonl); NHO=$(wc -l < $OUT/heldout_native.jsonl)
echo "DATA train=$NTR heldout=$NHO"

# 2) base floor (native op-naming on held-out)
if serve base ""; then
  $PY $MA/synth_native_eval.py --data $OUT/heldout_native.jsonl --base http://localhost:$PORT/v1 \
    --model "$BASE" --out $OUT/results/base_heldout.json 2>&1 | sed 's/^/[base] /' || true
fi

# 3) train native LoRA
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU $PY $DIST/lora_train_chat_toolcall.py \
  --base-model $BASE --train-jsonl $OUT/route_native.jsonl --out-dir $S/adapters/$TAG --device cuda:0 \
  --epochs 1 --lr 2e-4 --max-seq-len 2048 --lora-r 64 --lora-alpha 128 --grad-accum 16 --max-examples $NTR \
  2>&1 | tail -40
[ -d $S/adapters/$TAG ] || { echo TRAIN_FAIL; exit 1; }

# 4) trained eval (native op-naming on held-out)
if serve $TAG "$S/adapters/$TAG"; then
  $PY $MA/synth_native_eval.py --data $OUT/heldout_native.jsonl --base http://localhost:$PORT/v1 \
    --model "$TAG" --out $OUT/results/trained_heldout.json 2>&1 | sed 's/^/[trained] /' || true
fi

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "=== FACET3 PHASE1 DONE — base vs trained native op-naming (held-out) ==="
for f in base_heldout trained_heldout; do echo "--- $f ---"; $PY -c "import json;d=json.load(open('$OUT/results/$f.json'));print(d['overall']);print('by_op',d['by_op'])" 2>/dev/null || echo NA; done
date; echo FACET3_DONE
