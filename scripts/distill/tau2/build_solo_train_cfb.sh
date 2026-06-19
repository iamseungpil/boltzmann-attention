#!/bin/bash
# solo+cfbsynth 학습. solo_lite(synth+soptb)에 cfbsynth(P2b/P4 추상) 추가가 retail 연속홉 fetch를 살리나.
# 동일 lr5e-5·r16·ckpt-at → eval로 order_id 날조율·pass를 solo_lite와 비교(1-변수=cfbsynth 추가).
# Usage: build_solo_train_cfb.sh <gpu> [layers] [lr] [r] [alpha] [ckpt_at] [seq]
set -u
GPU=${1:?gpu}; LAYERS=${2:-}; LR=${3:-5e-5}; R=${4:-16}; ALPHA=${5:-32}
CKPT_AT=${6:-5,10,20,40,80,160,320,600}; SEQ=${7:-4096}
REPO=/home/woori/workspace_common/boltzmann-attention-pi
DIST=$REPO/scripts/distill
PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; FC=$S/fc_build
TAG=qwen7b_solo_cfb
LOG=$S/build_solo_train_cfb.log
exec > $LOG 2>&1; set -x; date
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[ -f $FC/sft_solo_cfb.jsonl ] || { echo "CFB_DATA_MISSING — run build_solo_data_cfb.sh first"; exit 1; }
LAYERARG=""; [ -n "$LAYERS" ] && LAYERARG="--lora-layers $LAYERS"
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
rm -rf $S/adapters/$TAG
CUDA_VISIBLE_DEVICES=$GPU $PY -u $DIST/lora_train_chat_toolcall.py \
  --base-model Qwen/Qwen2.5-7B-Instruct --train-jsonl $FC/sft_solo_cfb.jsonl \
  --out-dir $S/adapters/$TAG --device cuda:0 \
  --epochs 1 --lr $LR --lora-r $R --lora-alpha $ALPHA $LAYERARG --grad-accum 16 --max-seq-len $SEQ \
  --log-every 10 --save-every 50 --ckpt-at $CKPT_AT --resume \
  --skip-overlong --attn flash_attention_2
[ -d $S/adapters/$TAG ] && echo SOLO_CFB_DONE || echo TRAIN_FAIL
echo "=== snapshots ==="; ls -d $S/adapters/$TAG/step* 2>/dev/null | sort -V
date
