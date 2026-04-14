#!/usr/bin/env bash
# Wave 3 chain: launches after the Wave-1+Wave-2 (Qwen real + controls) supervisor
# (PID 343600) exits. Runs Llama + Mistral label_logprob verification in parallel
# to keep GPU0+GPU1 busy until the cross-model story under the new scorer is
# fully resolved.
#
# Wave 3a (GPU0/GPU1, parallel):
#   GPU0: Llama-3.1-8B original B_ont × {no_steer, a0.3} × {sum, mean} scorer
#   GPU1: Mistral-7B-v0.3 skipL0+padmax B_ont × {no_steer, a0.3} × {sum, mean}
# Wave 3b (GPU0/GPU1, parallel):
#   GPU0: Llama random + featshuffle × {no_steer, a0.3} × {sum, mean}
#   GPU1: Mistral-7B-Instruct-v0.3 skipL0+padmax × {no_steer, a0.3} × {sum, mean} (H2)
set -u

REPO=/home/woori/workspace_common/boltzmann-attention
cd "$REPO"
source /home/woori/workspace_common/CDP/poc/set.env

LOG_DIR=logs/codex_verify_2026_04_14
OUT_DIR=reports/codex_verify_2026_04_14
mkdir -p "$LOG_DIR" "$OUT_DIR"

WAVE2_PID=343600
echo "[wave3] waiting for Wave-2 supervisor PID $WAVE2_PID at $(date)" >> "$LOG_DIR/wave3.log"
while kill -0 "$WAVE2_PID" 2>/dev/null; do
  sleep 60
done
echo "[wave3] Wave-2 exited, launching Wave-3a at $(date)" >> "$LOG_DIR/wave3.log"

run_eval() {
  local TAG=$1 MODEL=$2 DEVICE=$3 BONT=$4
  for NORM in sum mean; do
    python scripts/ocq/eval_metatool_subtask1.py \
      --model "$MODEL" --device "$DEVICE" \
      --scorer label_logprob --lp-normalize "$NORM" \
      --methods no_steer ocq_bias_a0.3 \
      --b-ont "$BONT" \
      --max-samples 0 \
      --per-sample-dump "$OUT_DIR/full995_${NORM}_${TAG}_persample.jsonl" \
      --out "$OUT_DIR/full995_${NORM}_${TAG}.json" \
      > "$LOG_DIR/full995_${NORM}_${TAG}.log" 2>&1
  done
}

# --- Wave 3a: Llama real (GPU0) + Mistral skipL0+padmax (GPU1) ---
(
  run_eval "llama31_real" "meta-llama/Llama-3.1-8B" "cuda:0" \
    "external/SEKA/seka_projections/ontology-llama31-8b-metatool/B_ont.pt"
) &
WAVE3A_GPU0=$!
(
  run_eval "mistral_skipL0padmax" "mistralai/Mistral-7B-v0.3" "cuda:1" \
    "external/SEKA/seka_projections/ontology-mistral-7b-v03-metatool-skipL0-padmax/B_ont.pt"
) &
WAVE3A_GPU1=$!
wait $WAVE3A_GPU0 $WAVE3A_GPU1
echo "[wave3] Wave-3a complete at $(date)" >> "$LOG_DIR/wave3.log"

# --- Wave 3b: Llama controls (GPU0) + Mistral-Instruct H2 (GPU1) ---
(
  run_eval "llama31_random" "meta-llama/Llama-3.1-8B" "cuda:0" \
    "external/SEKA/seka_projections/ontology-llama31-8b-metatool-random/B_ont.pt"
  run_eval "llama31_featshuffle" "meta-llama/Llama-3.1-8B" "cuda:0" \
    "external/SEKA/seka_projections/ontology-llama31-8b-metatool-featshuffle/B_ont.pt"
) &
WAVE3B_GPU0=$!
(
  run_eval "mistral_instruct_skipL0padmax" "mistralai/Mistral-7B-Instruct-v0.3" "cuda:1" \
    "external/SEKA/seka_projections/ontology-mistral-7b-v03-metatool-skipL0-padmax/B_ont.pt"
) &
WAVE3B_GPU1=$!
wait $WAVE3B_GPU0 $WAVE3B_GPU1
echo "[wave3] Wave-3b complete at $(date)" >> "$LOG_DIR/wave3.log"
