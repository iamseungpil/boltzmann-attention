#!/bin/bash
# Phase 2a — Steering inference server for Hermes-3-Llama-3.1-8B on GPU1:9201
set -euo pipefail
cd /home/woori/workspace_common/boltzmann-attention-pi

PHASE2="reports/facet_rft_2026/phase2_steering"
mkdir -p "$PHASE2"
LOG="$PHASE2/steerserv_hermes3_$(date +%Y%m%d_%H%M%S).log"

VEC="$PHASE2/steering_vectors_hermes3.pt"
if [ ! -f "$VEC" ]; then
  echo "[phase2a-steerserv-hermes3] ERROR vectors not found: $VEC" >&2
  exit 1
fi

python -c "import fastapi, uvicorn" 2>/dev/null || pip install -q fastapi uvicorn

nohup python _steering_server.py \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --steering-vectors "$VEC" \
  --device cuda:1 \
  --dtype bfloat16 \
  --port 9201 \
  --tool-parser hermes \
  --served-model-name hermes3-steer \
  >> "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PHASE2/steerserv_hermes3.pid"
echo "[phase2a-steerserv-hermes3] pid=$PID log=$LOG"

for i in $(seq 1 90); do
  if curl -sf http://127.0.0.1:9201/health >/dev/null; then
    echo "[phase2a-steerserv-hermes3] READY after ${i}s"
    exit 0
  fi
  sleep 2
done
echo "[phase2a-steerserv-hermes3] WARN: not ready after 180s"
