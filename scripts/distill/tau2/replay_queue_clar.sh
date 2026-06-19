#!/bin/bash
# #6 instruction-clarity arm — KEY_MISMATCH 지배 크기(7B·32B-int8)에 --clarify replay. baseline과 비교.
# Usage: replay_queue_clar.sh <gpu> <port>
set -u
GPU=${1:-0}; PORT=${2:-8351}
T2=/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2
S=/home/woori/scratch
exec > $S/replay_queue_clar.log 2>&1; date
export REPLAY_FLAGS=--clarify
for spec in \
  "base7b_clar:Qwen/Qwen2.5-7B-Instruct" \
  "base32bint8_clar:Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8" ; do
  TAG=${spec%%:*}; MODEL=${spec#*:}
  echo "######## $TAG ($MODEL) --clarify ########"; date
  bash $T2/replay_scale.sh $GPU $PORT "$MODEL" "$TAG"
  grep -E "=====|P\(resolve|P\(ground|autopsy" $S/replay_scale_${TAG}.log
done
echo "=== REPLAY_QUEUE_CLAR DONE ==="; date; echo REPLAY_QUEUE_CLAR_DONE
