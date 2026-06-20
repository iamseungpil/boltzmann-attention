#!/bin/bash
# ReST S1 학습 — small-rank LoRA + replay로 retail discipline 내재화 (forgetting 통제).
# 설계=REST_INTERNALIZE_DESIGN §8 S1. ★진행률 가시 의무(30-remote-env): -u + 직접 LOG + --log-every + --save-every.
# 폴링: tail -f $LOG. 중단=같은명령 재실행(--resume).
# Usage: rest_s1_train.sh <gpu> [lora_r] [epochs] [seq]
set -u
GPU=${1:?gpu}; R=${2:-16}; EP=${3:-3}; SEQ=${4:-12288}
ALPHA=$((R*2))
REPO=/home/woori/workspace_common/boltzmann-attention-pi
DIST=$REPO/scripts/distill; T2=$DIST/tau2
PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; FC=$S/fc_build
MIX=$FC/sft_rest_s0_mix.jsonl
TAG=qwen7b_rest_s0_r${R}
LOG=$S/rest_s1_train_r${R}.log
exec > $LOG 2>&1; set -x; date
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# build replay mix (retail 318 + general 1:1)
$PY $T2/rest_s0_mix_replay.py $FC/sft_rest_s0_retail.jsonl $MIX 1.0 42

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU $PY -u $DIST/lora_train_chat_toolcall.py \
  --base-model Qwen/Qwen2.5-7B-Instruct --train-jsonl $MIX \
  --out-dir $S/adapters/$TAG --device cuda:0 \
  --epochs $EP --lr 1e-4 --lora-r $R --lora-alpha $ALPHA --grad-accum 8 --max-seq-len $SEQ \
  --val-frac 0.05 --log-every 5 --save-every 50 --resume \
  --skip-overlong --attn flash_attention_2
[ -d $S/adapters/$TAG ] && echo "REST_S1_TRAIN_DONE $TAG" || echo TRAIN_FAIL
date
