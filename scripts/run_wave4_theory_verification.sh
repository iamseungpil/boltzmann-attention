#!/usr/bin/env bash
# Wave 4 chain: theory-verification experiments.
# Launches after Wave 3 supervisor (PID 356966) exits.
#
# Wave 4a (GPU0+GPU1, parallel):
#   GPU0: Thm 6.1 empirical verification on Qwen L=13, N=100.
#   GPU1: Thm 6.1 empirical verification on Llama L=15, N=100.
#
# Wave 4b is deferred — (R)-necessity MMLU grid and Cor 6.9 nrank measurement
# require additional tooling (hard-gate implementation toggle + SVD pipeline).
# They are scripted here but not auto-launched; operator kicks them off
# manually after verifying Wave 4a output.
set -u

REPO=/home/woori/workspace_common/boltzmann-attention
cd "$REPO"
source /home/woori/workspace_common/CDP/poc/set.env

LOG_DIR=logs/theory_verify_2026_04_14
OUT_DIR=reports/theory_verify_2026_04_14
mkdir -p "$LOG_DIR" "$OUT_DIR"

WAVE3_PID=356966
echo "[wave4] waiting for Wave-3 supervisor PID $WAVE3_PID at $(date)" >> "$LOG_DIR/wave4.log"
while kill -0 "$WAVE3_PID" 2>/dev/null; do
  sleep 60
done
echo "[wave4] Wave-3 exited, launching Wave-4a at $(date)" >> "$LOG_DIR/wave4.log"

# --- Wave 4a: Thm 6.1 per-head verification ---
(
  python scripts/ocq/measure_theorem_6_1.py \
    --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 \
    --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt \
    --alpha 0.3 --layer 13 --max-samples 100 \
    --out "$OUT_DIR/thm61_qwen_L13_a0.3_N100.json" \
    > "$LOG_DIR/thm61_qwen.log" 2>&1
) &
GPU0_PID=$!
(
  python scripts/ocq/measure_theorem_6_1.py \
    --model meta-llama/Llama-3.1-8B --device cuda:1 \
    --b-ont external/SEKA/seka_projections/ontology-llama31-8b-metatool/B_ont.pt \
    --alpha 0.3 --layer 15 --max-samples 100 \
    --out "$OUT_DIR/thm61_llama_L15_a0.3_N100.json" \
    > "$LOG_DIR/thm61_llama.log" 2>&1
) &
GPU1_PID=$!
wait $GPU0_PID $GPU1_PID
echo "[wave4] Wave-4a complete at $(date)" >> "$LOG_DIR/wave4.log"
