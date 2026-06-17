#!/bin/bash
# C8 transfer batch — train several LoRA variants sequentially on ONE gpu, eval each gloss-free on
# held-out (S2, new vocab/schema = transfer) + in-dist (S3, convergence). Robust: every variant is
# independent (train/serve failure -> log + continue to next). Designed for unattended ~4h run.
# Launch ONE per gpu (A on gpu0, B on gpu1). Use setsid nohup so it survives ssh disconnect.
#   Usage: bash c8_batch.sh <GPU> <PORT> <LINE:A|B>
set -u
GPU=${1:?gpu}; PORT=${2:?port}; LINE=${3:?A|B}
REPO=/home/woori/workspace_common/boltzmann-attention-pi
MA=$REPO/scripts/distill/ma; DIST=$REPO/scripts/distill
S=/home/woori/scratch/depth/c8
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
BASE=Qwen/Qwen2.5-7B-Instruct
mkdir -p $S/adapters $S/results $S/logs
exec > $S/logs/batch_${LINE}_g${GPU}.log 2>&1; set -x; date; echo "LINE=$LINE GPU=$GPU PORT=$PORT"

kill_gpu(){ for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 5; }

serve(){ # $1=served-id  $2=lora_path(empty=base)
  kill_gpu
  if [ -n "${2:-}" ]; then
    CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT --max-model-len 16384 \
      --enable-lora --max-lora-rank 64 --lora-modules ${1}=${2} > $S/logs/serve_${1}_g${GPU}.log 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT --max-model-len 16384 \
      > $S/logs/serve_base_g${GPU}.log 2>&1 &
  fi
  for i in $(seq 1 100); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$1" && return 0; sleep 6; done
  echo "SERVE_FAIL $1"; tail -20 $S/logs/serve_${1}_g${GPU}.log; return 1
}

evalm(){ # $1=served-id  -> heldout(S2) + indist(S3), gloss-free
  $PY $MA/depth_eval.py --data $S/c8_eval_heldout.jsonl --base http://localhost:$PORT/v1 --model "$1" --arms B --gloss 0 --out $S/results/${1}__heldout.json || true
  $PY $MA/depth_eval.py --data $S/c8_eval_indist.jsonl  --base http://localhost:$PORT/v1 --model "$1" --arms B --gloss 0 --out $S/results/${1}__indist.json  || true
}

train(){ # $1=tag $2=trainjsonl $3=ep $4=lr
  CUDA_VISIBLE_DEVICES=$GPU $PY $DIST/lora_train_chat_toolcall.py \
    --base-model $BASE --train-jsonl "$2" --out-dir $S/adapters/$1 --device cuda:0 \
    --epochs "$3" --lr "$4" --max-seq-len 2048 --lora-r 64 --lora-alpha 128 --grad-accum 16 \
    || { echo "TRAIN_FAIL $1"; return 1; }
}

variant(){ # $1=tag $2=trainjsonl $3=ep $4=lr
  echo "##### VARIANT $1 ep=$3 lr=$4 data=$2 #####"; date
  train "$1" "$2" "$3" "$4" || return 0
  [ -d $S/adapters/$1 ] || { echo "NO_ADAPTER $1"; return 0; }
  serve "$1" "$S/adapters/$1" || return 0
  evalm "$1"
  kill_gpu
  echo "##### VARIANT $1 DONE #####"; date
}

cd $REPO && git pull --ff-only 2>&1 | tail -2 || true

if [ "$LINE" = "A" ]; then
  # base reference: S0 (gloss off) + S1 (gloss on), held-out + in-dist — once
  if serve base ""; then
    $PY $MA/depth_eval.py --data $S/c8_eval_heldout.jsonl --base http://localhost:$PORT/v1 --model $BASE --arms B --gloss 0 --out $S/results/BASE_S0__heldout.json || true
    $PY $MA/depth_eval.py --data $S/c8_eval_heldout.jsonl --base http://localhost:$PORT/v1 --model $BASE --arms B --gloss 1 --out $S/results/BASE_S1__heldout.json || true
    $PY $MA/depth_eval.py --data $S/c8_eval_indist.jsonl  --base http://localhost:$PORT/v1 --model $BASE --arms B --gloss 0 --out $S/results/BASE_S0__indist.json || true
    kill_gpu
  fi
  # learning-curve line (full 5-op, gloss-free)
  variant A1_ep1 $S/c8_train_sft.jsonl 1 2e-4
  variant A2_ep3 $S/c8_train_sft.jsonl 3 2e-4
  variant A3_ep6 $S/c8_train_sft.jsonl 6 2e-4
else
  # ablation line
  variant B1_cmpheavy $S/c8_train_cmpheavy.jsonl 3 2e-4   # comparative 3x data (hardest op)
  variant B2_glossin  $S/c8_train_glossin.jsonl  3 2e-4   # gloss IN sft (does explicit def help internalise?)
  variant B3_big      $S/c8_train_big.jsonl      2 2e-4   # 8k data (is it data-starved?)
  variant B4_lr3      $S/c8_train_sft.jsonl      3 3e-4   # higher LR
fi
echo "C8_BATCH_${LINE}_DONE"; date
