#!/bin/bash
# Phase 2a — T1 steering vector extraction for Hermes-3-Llama-3.1-8B
set -euo pipefail

cd /home/woori/workspace_common/boltzmann-attention-pi

REPO="$(pwd)"
PHASE0="$REPO/reports/facet_rft_2026/phase0_probing"
PHASE2="$REPO/reports/facet_rft_2026/phase2_steering"
mkdir -p "$PHASE2"

LOG="$PHASE2/extract_hermes3_$(date +%Y%m%d_%H%M%S).log"

echo "[phase2a-extract-hermes3] start $(date)" | tee "$LOG"
DEVICE="${PHASE2A_EXTRACT_DEVICE:-cuda:0}"
nohup python _extract_steering_vectors.py \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --pairs "$PHASE0/contrast_pairs_v3.json" \
  --out "$PHASE2/steering_vectors_hermes3.pt" \
  --device "$DEVICE" \
  --dtype bfloat16 \
  --layers all \
  --batch-size 4 \
  --max-length 2048 \
  >> "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PHASE2/extract_hermes3.pid"
echo "[phase2a-extract-hermes3] pid=$PID log=$LOG"
