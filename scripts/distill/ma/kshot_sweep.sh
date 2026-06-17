#!/bin/bash
# K-sweep: expression-diversity -> transfer curve (EXPRESSION_DIVERSITY_TRANSFER_DESIGN §5).
# For each (K, method): build K-controlled train data (record D), LoRA-train, eval tau2 (surface-
# collapse rate + accuracy) + held-out FULL-diverse (OOD expression). One gpu, sequential, robust.
#   Usage: bash kshot_sweep.sh <GPU> <PORT> <METHOD:random|kcenter> "<K list>"
set -u
GPU=${1:?}; PORT=${2:?}; METHOD=${3:?}; KS=${4:-"1 2 4 8 16 32"}
REPO=/home/woori/workspace_common/boltzmann-attention-pi
MA=$REPO/scripts/distill/ma; DIST=$REPO/scripts/distill
S=/home/woori/scratch/depth/c8; KS_DIR=$S/ksweep
PY=/home/woori/venvs/seka_env/bin/python; VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
BASE=Qwen/Qwen2.5-7B-Instruct
mkdir -p $KS_DIR $S/adapters $S/logs
exec > $S/logs/ksweep_${METHOD}_g${GPU}.log 2>&1; set -x; date; echo "METHOD=$METHOD KS=$KS"
cd $REPO && git pull --ff-only 2>&1 | tail -1 || true
kill_gpu(){ for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 5; }
serve(){ kill_gpu; CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT --max-model-len 16384 \
  --enable-lora --max-lora-rank 64 --lora-modules $1=$2 > $S/logs/ksweep_serve_$1.log 2>&1 &
  for i in $(seq 1 90); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q '"id"' && return 0; sleep 6; done
  echo "SERVE_FAIL $1"; return 1; }

for K in $KS; do
  TAG=K${K}_${METHOD}
  echo "##### CELL $TAG #####"; date
  # 1. K-controlled train data (records <out>.D.json)
  $PY $MA/c8_build_sft.py --mode train --diverse 1 --K $K --method $METHOD \
      --n 3000 --out $KS_DIR/train_$TAG.jsonl || { echo "BUILD_FAIL $TAG"; continue; }
  # 2. train
  kill_gpu
  CUDA_VISIBLE_DEVICES=$GPU $PY $DIST/lora_train_chat_toolcall.py --base-model $BASE \
    --train-jsonl $KS_DIR/train_$TAG.jsonl --out-dir $S/adapters/$TAG --device cuda:0 \
    --epochs 2 --lr 2e-4 --max-seq-len 2048 --lora-r 64 --lora-alpha 128 --grad-accum 16 \
    --max-examples 3000 || { echo "TRAIN_FAIL $TAG"; continue; }
  [ -d $S/adapters/$TAG ] || { echo "NO_ADAPTER $TAG"; continue; }
  # 3. serve + eval (tau2 transfer + held-out full-diverse OOD)
  serve $TAG $S/adapters/$TAG || continue
  $PY $MA/tau2_op_eval.py --model $TAG --gloss 0 --base http://localhost:$PORT/v1 \
      --out $KS_DIR/tau2_$TAG.json || true
  $PY $MA/depth_eval.py --data $S/c8_eval_heldout_div.jsonl --base http://localhost:$PORT/v1 \
      --model $TAG --arms B --gloss 0 --out $KS_DIR/heldout_$TAG.json || true
  kill_gpu
  echo "##### CELL $TAG DONE #####"; date
done
echo "KSWEEP_${METHOD}_DONE"; date
