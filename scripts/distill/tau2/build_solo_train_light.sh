#!/bin/bash
# 가벼운 단독 통합 LoRA 재학습 — qwen7b_solo_sts(lr2e-4·r64·loss→0 과적합→retail e2e 0/80)의 교정판.
# 가설: lr↓·rank↓·early(중간 ckpt)면 base 범용 named-tool 호출 보존하며 resolve_selection만 얹힘.
# 익명툴(func_NNNN)만 본 LoRA가 loss→0까지 가면 base tool-use를 덮어쓰고 operand 스키마가 bleed → 0/80.
# ★진행률 가시(30-remote-env): python -u + 직접 LOG + --log-every + --ckpt-steps(중간 평가가능 snapshot).
# loss가 lr↓r↓에도 opt~25에 near-zero 붕괴(데이터-구동 자명성) → pre-collapse 스냅샷을 명시 포착.
# Usage: build_solo_train_light.sh <gpu> [lr] [r] [alpha] [ckpt_at] [seq]
set -u
GPU=${1:?gpu}; LR=${2:-5e-5}; R=${3:-16}; ALPHA=${4:-32}
CKPT_AT=${5:-5,10,20,40,80,160,320,600}; SEQ=${6:-4096}
REPO=/home/woori/workspace_common/boltzmann-attention-pi
DIST=$REPO/scripts/distill
PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; FC=$S/fc_build
TAG=qwen7b_solo_lite
LOG=$S/build_solo_train_light.log
exec > $LOG 2>&1; set -x; date
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
rm -rf $S/adapters/$TAG   # 깨끗한 fresh run (이전 collapse 상태/거친 스냅샷 제거·--resume no-op)
CUDA_VISIBLE_DEVICES=$GPU $PY -u $DIST/lora_train_chat_toolcall.py \
  --base-model Qwen/Qwen2.5-7B-Instruct --train-jsonl $FC/sft_solo_sts.jsonl \
  --out-dir $S/adapters/$TAG --device cuda:0 \
  --epochs 1 --lr $LR --lora-r $R --lora-alpha $ALPHA --grad-accum 16 --max-seq-len $SEQ \
  --log-every 10 --save-every 50 --ckpt-at $CKPT_AT --resume \
  --skip-overlong --attn flash_attention_2
[ -d $S/adapters/$TAG ] && echo SOLO_LITE_DONE || echo TRAIN_FAIL
echo "=== numbered snapshots (eval-able) ==="; ls -d $S/adapters/$TAG/step* 2>/dev/null
date
