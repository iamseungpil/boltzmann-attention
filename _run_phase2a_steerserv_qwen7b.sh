#!/bin/bash
# Phase 2a — Steering inference server for Qwen2.5-7B-Instruct on GPU0:9200
set -euo pipefail
cd /home/woori/workspace_common/boltzmann-attention-pi

PHASE2="reports/facet_rft_2026/phase2_steering"
mkdir -p "$PHASE2"
LOG="$PHASE2/steerserv_qwen7b_$(date +%Y%m%d_%H%M%S).log"

VEC="$PHASE2/steering_vectors_qwen7b.pt"
if [ ! -f "$VEC" ]; then
  echo "[phase2a-steerserv-qwen7b] ERROR vectors not found: $VEC" >&2
  exit 1
fi

# pip deps (idempotent)
python -c "import fastapi, uvicorn" 2>/dev/null || pip install -q fastapi uvicorn

nohup python _steering_server.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --steering-vectors "$VEC" \
  --device cuda:0 \
  --dtype bfloat16 \
  --port 9200 \
  --tool-parser hermes \
  --served-model-name qwen7b-steer \
  >> "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PHASE2/steerserv_qwen7b.pid"
echo "[phase2a-steerserv-qwen7b] pid=$PID log=$LOG"

# wait until /health responds
for i in $(seq 1 90); do
  if curl -sf http://127.0.0.1:9200/health >/dev/null; then
    echo "[phase2a-steerserv-qwen7b] READY after ${i}s"
    exit 0
  fi
  sleep 2
done
echo "[phase2a-steerserv-qwen7b] WARN: not ready after 180s"
