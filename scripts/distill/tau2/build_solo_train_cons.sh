#!/bin/bash
# 아이디어2 arm: consistency 데이터(no_alias 실명)로 전층 학습. light(alias 데이터·전층)와 1-변수 차이.
# 가설: 실명=base named-tool과 일관→additive→망각(operand bleed·캐논ID 환각) 회피.
# 동일 lr5e-5·r16·ckpt-at → eval_solo_ckpts로 light/mid와 망각곡선 직접 비교.
# Usage: build_solo_train_cons.sh <gpu> [lr] [r] [alpha] [ckpt_at] [seq]
set -u
GPU=${1:?gpu}; LR=${2:-5e-5}; R=${3:-16}; ALPHA=${4:-32}
CKPT_AT=${5:-5,10,20,40,80,160,320,600}; SEQ=${6:-4096}
REPO=/home/woori/workspace_common/boltzmann-attention-pi
DIST=$REPO/scripts/distill
PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; FC=$S/fc_build
TAG=qwen7b_solo_cons
LOG=$S/build_solo_train_cons.log
exec > $LOG 2>&1; set -x; date
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[ -f $FC/sft_solo_cons.jsonl ] || { echo "CONS_DATA_MISSING — run build_solo_data_cons.sh first"; exit 1; }
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
rm -rf $S/adapters/$TAG
CUDA_VISIBLE_DEVICES=$GPU $PY -u $DIST/lora_train_chat_toolcall.py \
  --base-model Qwen/Qwen2.5-7B-Instruct --train-jsonl $FC/sft_solo_cons.jsonl \
  --out-dir $S/adapters/$TAG --device cuda:0 \
  --epochs 1 --lr $LR --lora-r $R --lora-alpha $ALPHA --grad-accum 16 --max-seq-len $SEQ \
  --log-every 10 --save-every 50 --ckpt-at $CKPT_AT --resume \
  --skip-overlong --attn flash_attention_2
[ -d $S/adapters/$TAG ] && echo SOLO_CONS_DONE || echo TRAIN_FAIL
echo "=== snapshots ==="; ls -d $S/adapters/$TAG/step* 2>/dev/null | sort -V
date
