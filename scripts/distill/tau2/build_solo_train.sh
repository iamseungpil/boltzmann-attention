#!/bin/bash
# 단독 통합 LoRA 학습 (SOP+TaskBench+Synth content-op·CFB 제외). build_solo_data.sh 후.
# 1차 빠른 검증: 1ep·r64·seq8192. = qwen7b_solo_sts.
# Usage: build_solo_train.sh <gpu>
set -u
GPU=${1:?gpu}
R=/home/woori/workspace_common/boltzmann-attention-pi
DIST=$R/scripts/distill
PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; FC=$S/fc_build
TAG=qwen7b_solo_sts
LOG=$S/build_solo_train.log
exec > $LOG 2>&1; set -x; date
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU $PY $DIST/lora_train_chat_toolcall.py \
  --base-model Qwen/Qwen2.5-7B-Instruct --train-jsonl $FC/sft_solo_sts.jsonl \
  --out-dir $S/adapters/$TAG --device cuda:0 \
  --epochs 1 --lr 2e-4 --lora-r 64 --lora-alpha 128 --grad-accum 16 --max-seq-len 8192 \
  --skip-overlong --attn flash_attention_2 2>&1 | tail -60
[ -d $S/adapters/$TAG ] && echo SOLO_TRAIN_DONE || echo TRAIN_FAIL
date
