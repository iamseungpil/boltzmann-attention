#!/bin/bash
# 구조적 arm (아이디어1): 중간층만 LoRA. Qwen2.5-7B=28층 → 중간밴드 8-19(초기=표면·후기=출력 freeze).
# 가설: 라우팅 추상화는 중간층 → 거기만 건드리면 출력-레벨 망각(operand bleed·캐논ID 환각) 회피.
# light arm(전층 r16)과 동일 lr5e-5·r16·ckpt-at → eval_solo_ckpts로 망각곡선 직접 비교(유일 변화=층 subset).
# Usage: build_solo_train_mid.sh <gpu> [layers] [lr] [r] [alpha] [ckpt_at] [seq]
set -u
GPU=${1:?gpu}; LAYERS=${2:-8-19}; LR=${3:-5e-5}; R=${4:-16}; ALPHA=${5:-32}
CKPT_AT=${6:-5,10,20,40,80,160,320,600}; SEQ=${7:-4096}
REPO=/home/woori/workspace_common/boltzmann-attention-pi
DIST=$REPO/scripts/distill
PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; FC=$S/fc_build
TAG=qwen7b_solo_mid
LOG=$S/build_solo_train_mid.log
exec > $LOG 2>&1; set -x; date
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
rm -rf $S/adapters/$TAG
CUDA_VISIBLE_DEVICES=$GPU $PY -u $DIST/lora_train_chat_toolcall.py \
  --base-model Qwen/Qwen2.5-7B-Instruct --train-jsonl $FC/sft_solo_sts.jsonl \
  --out-dir $S/adapters/$TAG --device cuda:0 \
  --epochs 1 --lr $LR --lora-r $R --lora-alpha $ALPHA --lora-layers $LAYERS \
  --grad-accum 16 --max-seq-len $SEQ \
  --log-every 10 --save-every 50 --ckpt-at $CKPT_AT --resume \
  --skip-overlong --attn flash_attention_2
[ -d $S/adapters/$TAG ] && echo SOLO_MID_DONE || echo TRAIN_FAIL
echo "=== snapshots ==="; ls -d $S/adapters/$TAG/step* 2>/dev/null | sort -V
date
