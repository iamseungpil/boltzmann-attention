#!/bin/bash
# 선택-formalize base scale 곡선 큐 — 한 GPU서 크기들 순차 serve+replay(비용0·로컬). woori ≤32B int8.
# Usage: replay_queue.sh <gpu> <port>
set -u
GPU=${1:-0}; PORT=${2:-8351}
T2=/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2
S=/home/woori/scratch
QLOG=$S/replay_queue.log
exec > $QLOG 2>&1; date
# tag:hf_model  (32B는 int8=우리 상한)
for spec in \
  "base1p5b:Qwen/Qwen2.5-1.5B-Instruct" \
  "base7b:Qwen/Qwen2.5-7B-Instruct" \
  "base14b:Qwen/Qwen2.5-14B-Instruct" \
  "base32bint8:Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8" ; do
  TAG=${spec%%:*}; MODEL=${spec#*:}
  echo "######## $TAG ($MODEL) ########"; date
  bash $T2/replay_scale.sh $GPU $PORT "$MODEL" "$TAG"
  grep -A4 "=====" $S/replay_scale_${TAG}.log | grep -E "=====|P\(resolve|P\(ground|autopsy"
done
echo "=== REPLAY_QUEUE DONE ==="; date; echo REPLAY_QUEUE_DONE
