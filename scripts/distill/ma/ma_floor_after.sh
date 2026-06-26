#!/bin/bash
# Watcher: wait until BOTH in-flight sweeps (p8013 7B/14B, p8014 32B) finish, then run the
# decisive FLOOR + Bfair sweep (with cost instrumentation) on GPU0 — sequentially over all
# 3 models. No concurrency (avoids GPU/port collisions). Fire-and-forget.
set -u
S=/home/woori/scratch
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=$S/ma_floor_after.log
exec > $LOG 2>&1; set -x; date

# wait for both prior sweeps (cap ~3h)
for i in $(seq 1 540); do
  d13=$(grep -c MA_SCALE_DONE $S/ma_eval_scale_p8013.log 2>/dev/null || echo 0)
  d14=$(grep -c MA_SCALE_DONE $S/ma_eval_scale_p8014.log 2>/dev/null || echo 0)
  [ "$d13" -ge 1 ] && [ "$d14" -ge 1 ] && { echo "BOTH_DONE after ${i} ticks"; break; }
  sleep 20
done
date
# free GPU0 (kill leftover vllm on GPU0 only)
for p in $(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
cd $REPO && git pull --ff-only
# decisive arms: A (concrete baseline) + Bfair (fair selector, gate) + floor ladder L0-L3
bash scripts/distill/ma/ma_eval_scale.sh \
  "Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8" \
  "A,Bfair,L0,L1,L2a,L2b,L3" "_floor" 0 8013
echo MA_FLOOR_AFTER_DONE; date
