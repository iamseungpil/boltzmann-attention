#!/usr/bin/env bash
# Auto-chain: wait for current full-995 sum/mean runs to finish, then launch
# control bundle (random + featshuffle) on both GPUs in parallel.
#
# GPU0: random control × {sum, mean} scorer (sequentially within the job)
# GPU1: featshuffle control × {sum, mean} scorer (sequentially within the job)
set -u

REPO=/home/woori/workspace_common/boltzmann-attention
cd "$REPO"
source /home/woori/workspace_common/CDP/poc/set.env

LOG_DIR=logs/codex_verify_2026_04_14
OUT_DIR=reports/codex_verify_2026_04_14
mkdir -p "$LOG_DIR" "$OUT_DIR"

# --- Wait for existing runs to finish ---
PIDS="327202 328228"
for p in $PIDS; do
  while kill -0 "$p" 2>/dev/null; do
    sleep 30
  done
  echo "[chain] PID $p exited" >> "$LOG_DIR/chain.log"
done
echo "[chain] all prior runs exited, launching control bundle at $(date)" >> "$LOG_DIR/chain.log"

# --- GPU0: random control, both scorers ---
(
  for NORM in sum mean; do
    python scripts/ocq/eval_metatool_subtask1.py \
      --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 \
      --scorer label_logprob --lp-normalize "$NORM" \
      --methods no_steer ocq_bias_a0.3 \
      --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-metatool-random/B_ont.pt \
      --max-samples 0 \
      --per-sample-dump "$OUT_DIR/full995_${NORM}_random_persample.jsonl" \
      --out "$OUT_DIR/full995_${NORM}_random.json" \
      > "$LOG_DIR/full995_${NORM}_random.log" 2>&1
  done
) &
GPU0_PID=$!

# --- GPU1: featshuffle control, both scorers ---
(
  for NORM in sum mean; do
    python scripts/ocq/eval_metatool_subtask1.py \
      --model Qwen/Qwen2.5-7B-Instruct --device cuda:1 \
      --scorer label_logprob --lp-normalize "$NORM" \
      --methods no_steer ocq_bias_a0.3 \
      --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-metatool-featshuffle/B_ont.pt \
      --max-samples 0 \
      --per-sample-dump "$OUT_DIR/full995_${NORM}_featshuffle_persample.jsonl" \
      --out "$OUT_DIR/full995_${NORM}_featshuffle.json" \
      > "$LOG_DIR/full995_${NORM}_featshuffle.log" 2>&1
  done
) &
GPU1_PID=$!

echo "[chain] launched GPU0=$GPU0_PID GPU1=$GPU1_PID at $(date)" >> "$LOG_DIR/chain.log"
wait $GPU0_PID $GPU1_PID
echo "[chain] control bundle complete at $(date)" >> "$LOG_DIR/chain.log"
