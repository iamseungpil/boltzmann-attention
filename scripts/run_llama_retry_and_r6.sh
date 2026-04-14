#!/usr/bin/env bash
# Two-track chain on GPU0 (GPU1 is busy running Mistral skipL0+padmax from Wave 3a):
#   Track A: Llama-3.1-8B label_logprob retry using NousResearch mirror (un-gated)
#            — both sum and mean scorers × full 995 on original + random + featshuffle
#   Track B (after Track A): R6 MMLU gate-mode grid (12 configs × N=1000)
#
# Estimated total: Track A ~2.5h, Track B ~4h. Total ~6.5 GPU-hours on GPU0.
set -u

REPO=/home/woori/workspace_common/boltzmann-attention
cd "$REPO"
source /home/woori/workspace_common/CDP/poc/set.env

LOG_DIR=logs/codex_verify_2026_04_14
OUT_DIR=reports/codex_verify_2026_04_14
R6_LOG=logs/theory_verify_2026_04_14/r6
R6_OUT=reports/theory_verify_2026_04_14/r6
mkdir -p "$LOG_DIR" "$OUT_DIR" "$R6_LOG" "$R6_OUT"

LLAMA_MODEL="NousResearch/Meta-Llama-3.1-8B"
QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"
B_ONT_QWEN="external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt"

run_metatool() {
  local MODEL=$1 TAG=$2 BONT=$3 NORM=$4
  python scripts/ocq/eval_metatool_subtask1.py \
    --model "$MODEL" --device cuda:0 \
    --scorer label_logprob --lp-normalize "$NORM" \
    --methods no_steer ocq_bias_a0.3 \
    --b-ont "$BONT" \
    --max-samples 0 \
    --per-sample-dump "$OUT_DIR/full995_${NORM}_${TAG}_persample.jsonl" \
    --out "$OUT_DIR/full995_${NORM}_${TAG}.json" \
    > "$LOG_DIR/full995_${NORM}_${TAG}.log" 2>&1
}

run_mmlu() {
  local TAG=$1 GATE_MODE=$2; shift 2
  local METHODS=$@
  python scripts/ocq/eval_mmlu_subset.py \
    --model "$QWEN_MODEL" --device cuda:0 \
    --n-samples 1000 --n-shot 5 --seed 42 \
    --methods $METHODS \
    --b-ont "$B_ONT_QWEN" \
    --gate-mode "$GATE_MODE" \
    --out "$R6_OUT/r6_${TAG}.json" \
    > "$R6_LOG/r6_${TAG}.log" 2>&1
}

# ============ Track A: Llama label_logprob retry ============
echo "[chain] Track A start (Llama retry) at $(date)" >> "$LOG_DIR/llama_retry.log"

for NORM in sum mean; do
  run_metatool "$LLAMA_MODEL" "llama31_real" \
    "external/SEKA/seka_projections/ontology-llama31-8b-metatool/B_ont.pt" "$NORM"
  run_metatool "$LLAMA_MODEL" "llama31_random" \
    "external/SEKA/seka_projections/ontology-llama31-8b-metatool-random/B_ont.pt" "$NORM"
  run_metatool "$LLAMA_MODEL" "llama31_featshuffle" \
    "external/SEKA/seka_projections/ontology-llama31-8b-metatool-featshuffle/B_ont.pt" "$NORM"
done

echo "[chain] Track A done at $(date)" >> "$LOG_DIR/llama_retry.log"

# ============ Track B: R6 MMLU gate-mode grid ============
echo "[chain] Track B start (R6 MMLU grid) at $(date)" >> "$R6_LOG/r6_master.log"

# Baseline
run_mmlu "baseline" "soft" no_steer
# Flat K-bias α sweep
for A in 0.1 0.2 0.3 0.5 1.0; do
  run_mmlu "flat_a${A}" "soft" "ocq_bias_a${A}"
done
# Soft facet-gated
for A in 0.3 1.0; do
  run_mmlu "soft_a${A}" "soft" "ocq_facet_gated_a${A}"
done
# Hard-threshold (R-violation variant 1)
for A in 0.3 1.0; do
  run_mmlu "hard_thresh_a${A}" "hard_thresh" "ocq_facet_gated_a${A}"
done
# Hard-argmax (R-violation variant 2)
for A in 0.3 1.0; do
  run_mmlu "hard_argmax_a${A}" "hard_argmax" "ocq_facet_gated_a${A}"
done

echo "[chain] Track B done at $(date)" >> "$R6_LOG/r6_master.log"
