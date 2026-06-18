#!/bin/bash
# 단독 통합 LoRA 학습 (SOP+TaskBench+Synth content-op·CFB 제외). build_solo_data.sh 후.
# 1차 빠른 검증: 1ep·r64·seq4096. = qwen7b_solo_sts.
# ★진행률 가시 의무(30-remote-env): python -u + 직접 LOG(tail 파이프 금지) + --log-every(step/total/ETA) + --save-every(resume).
# 폴링: tail -f $LOG. 중단되면 같은 명령 재실행 → --resume이 ckpt서 이어감.
# Usage: build_solo_train.sh <gpu> [seq] [save_every]
set -u
GPU=${1:?gpu}; SEQ=${2:-4096}; SAVE=${3:-50}
R=/home/woori/workspace_common/boltzmann-attention-pi
DIST=$R/scripts/distill
PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; FC=$S/fc_build
TAG=qwen7b_solo_sts
LOG=$S/build_solo_train.log
exec > $LOG 2>&1; set -x; date
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
# -u=unbuffered·직접 LOG(tail 금지)=실시간 진행률. --log-every 10=step/total/loss/ETA. --save-every=ckpt+resume.
CUDA_VISIBLE_DEVICES=$GPU $PY -u $DIST/lora_train_chat_toolcall.py \
  --base-model Qwen/Qwen2.5-7B-Instruct --train-jsonl $FC/sft_solo_sts.jsonl \
  --out-dir $S/adapters/$TAG --device cuda:0 \
  --epochs 1 --lr 2e-4 --lora-r 64 --lora-alpha 128 --grad-accum 16 --max-seq-len $SEQ \
  --log-every 10 --save-every $SAVE --resume \
  --skip-overlong --attn flash_attention_2
[ -d $S/adapters/$TAG ] && echo SOLO_TRAIN_DONE || echo TRAIN_FAIL
date
