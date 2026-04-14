#!/usr/bin/env bash
# R6: MMLU gate-mode grid for §3.4.1 figure (Remark 6.14.A.3 verification).
# Tests: no_steer, ocq_bias_a<α> (flat), ocq_facet_gated_a<α> with
# gate_mode ∈ {soft, hard_thresh, hard_argmax} on Qwen2.5-7B MMLU N=1000.
#
# Predicted pattern (Remark 6.14.A.3):
#   no_steer ≈ baseline
#   soft flat bias ≈ baseline ± small (noise floor)
#   soft facet_gated (Lipschitz) ≈ baseline (H-R satisfied)
#   hard_thresh / hard_argmax → monotone degradation in α
#
# Runtime: ~4 GPU-hours total on one A6000 (all on GPU0 since Wave3 Mistral
# is still running on GPU1).
set -u

REPO=/home/woori/workspace_common/boltzmann-attention
cd "$REPO"
source /home/woori/workspace_common/CDP/poc/set.env

LOG_DIR=logs/theory_verify_2026_04_14/r6
OUT_DIR=reports/theory_verify_2026_04_14/r6
mkdir -p "$LOG_DIR" "$OUT_DIR"

MODEL="Qwen/Qwen2.5-7B-Instruct"
B_ONT="external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt"

run_eval() {
  local TAG=$1 GATE_MODE=$2 shift 2
  local METHODS=$@
  python scripts/ocq/eval_mmlu_subset.py \
    --model "$MODEL" --device cuda:0 \
    --n-samples 1000 --n-shot 5 --seed 42 \
    --methods $METHODS \
    --b-ont "$B_ONT" \
    --gate-mode "$GATE_MODE" \
    --out "$OUT_DIR/r6_${TAG}.json" \
    > "$LOG_DIR/r6_${TAG}.log" 2>&1
}

# Baseline (no_steer only, cached for all comparisons)
run_eval "baseline" "soft" no_steer

# Flat K-bias (no gate, all α)
run_eval "flat_a0.1" "soft" ocq_bias_a0.1
run_eval "flat_a0.2" "soft" ocq_bias_a0.2
run_eval "flat_a0.3" "soft" ocq_bias_a0.3
run_eval "flat_a0.5" "soft" ocq_bias_a0.5
run_eval "flat_a1.0" "soft" ocq_bias_a1.0

# Soft facet-gated (Hypothesis R satisfied, expected ≈ baseline)
run_eval "soft_a0.3" "soft" ocq_facet_gated_a0.3
run_eval "soft_a1.0" "soft" ocq_facet_gated_a1.0

# Hard-threshold (R-violation variant 1, expected: monotone degradation)
run_eval "hard_thresh_a0.3" "hard_thresh" ocq_facet_gated_a0.3
run_eval "hard_thresh_a1.0" "hard_thresh" ocq_facet_gated_a1.0

# Hard-argmax (R-violation variant 2, expected: stronger degradation)
run_eval "hard_argmax_a0.3" "hard_argmax" ocq_facet_gated_a0.3
run_eval "hard_argmax_a1.0" "hard_argmax" ocq_facet_gated_a1.0

echo "[r6] all runs complete at $(date)" >> "$LOG_DIR/r6_master.log"
