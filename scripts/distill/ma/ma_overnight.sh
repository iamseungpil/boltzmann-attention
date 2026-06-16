#!/bin/bash
# ===== overnight factorial orchestrator (sole owner of ONE gpu) =========================
# Runs a queue of (seed x arm) factorial jobs back-to-back on a single GPU so it never idles.
# Resilient: each ma_factorial_batch.sh self-skips if its split exists (resumable) and a failing
# arm does NOT stop the queue (continue-on-fail). Optional wait-for-marker = deterministic
# handoff (e.g. GPU0 waits for exp0's EXP0_DONE before claiming the GPU) -> no race, no clobber.
#
# Usage: bash ma_overnight.sh <GPU> <PORT> "<SEEDS>" "<ARMS>" [WAIT_FILE] [WAIT_PAT]
set -u
GPU="${1:?gpu}"; PORT="${2:?port}"; SEEDS="${3:?seeds e.g. '0 1 2'}"; ARMS="${4:?arms}"
WAIT_FILE="${5:-}"; WAIT_PAT="${6:-}"
MA=/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/ma
LOG=/home/woori/scratch/ma_overnight_g${GPU}.log
exec > $LOG 2>&1; set -x; date
echo "OVERNIGHT gpu=$GPU port=$PORT seeds=[$SEEDS] arms=[$ARMS] wait=${WAIT_FILE}:${WAIT_PAT}"

# ---- deterministic handoff: wait for marker (e.g. exp0 done) before owning the GPU ----
if [ -n "$WAIT_FILE" ] && [ -n "$WAIT_PAT" ]; then
  echo "waiting for '$WAIT_PAT' in $WAIT_FILE ..."
  for i in $(seq 1 720); do                       # up to 720*30s = 6h safety cap
    grep -q "$WAIT_PAT" "$WAIT_FILE" 2>/dev/null && { echo "marker seen (i=$i)"; break; }
    sleep 30
  done
  sleep 20   # let the other job's final GPU free settle
fi

# ---- queue: seed-major so every arm gets seed0 first, then replications ----
for SEED in $SEEDS; do
  for ARM in $ARMS; do
    echo "===== [g$GPU] launch arm=$ARM seed=$SEED $(date) ====="
    bash $MA/ma_factorial_batch.sh "$ARM" "$GPU" "$PORT" "$SEED" || echo "ARM_FAILED arm=$ARM seed=$SEED (continuing)"
  done
done
echo "OVERNIGHT_g${GPU}_DONE"; date
