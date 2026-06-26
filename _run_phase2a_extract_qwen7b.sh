#!/bin/bash
# Phase 2a — T1 steering vector extraction for Qwen2.5-7B-Instruct
# Run after Qwen baseline (Phase 1) completes.
set -euo pipefail

cd /home/woori/workspace_common/boltzmann-attention-pi

REPO="$(pwd)"
PHASE0="$REPO/reports/facet_rft_2026/phase0_probing"
PHASE2="$REPO/reports/facet_rft_2026/phase2_steering"
mkdir -p "$PHASE2"

LOG="$PHASE2/extract_qwen7b_$(date +%Y%m%d_%H%M%S).log"

echo "[phase2a-extract-qwen7b] start $(date)" | tee "$LOG"
nohup python _extract_steering_vectors.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --pairs "$PHASE0/contrast_pairs_v3.json" \
  --out "$PHASE2/steering_vectors_qwen7b.pt" \
  --device cuda:0 \
  --dtype bfloat16 \
  --layers all \
  --batch-size 4 \
  --max-length 2048 \
  >> "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PHASE2/extract_qwen7b.pid"
echo "[phase2a-extract-qwen7b] pid=$PID log=$LOG"
