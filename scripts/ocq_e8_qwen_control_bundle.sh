#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/scratch/boltzmann/ba-ocq-develop}"
VENV_DIR="${2:-/scratch/boltzmann/venvs/ocq}"
GPU_QWEN="${GPU_QWEN:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12}"
HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
DRY_RUN="${DRY_RUN:-0}"
SHUFFLE_SEED="${SHUFFLE_SEED:-13}"

if [ "$DRY_RUN" = "1" ]; then
  if [ "$MAX_SAMPLES" = "0" ]; then
    MAX_SAMPLES="20"
  fi
  RUN_SUFFIX="smoke"
else
  if [ "$MAX_SAMPLES" = "0" ]; then
    RUN_SUFFIX="full"
  else
    RUN_SUFFIX="$MAX_SAMPLES"
  fi
fi

source "$VENV_DIR/bin/activate"
cd "$ROOT_DIR"
export HF_HOME
export TRANSFORMERS_CACHE="$HF_HOME"

mkdir -p results/ocq/cross_model

python scripts/ocq/build_metatool_ontology_v2.py \
  --out reports/axis2_theoretical_verification/metatool_ontology_v2.json

CUDA_VISIBLE_DEVICES="$GPU_QWEN" \
python scripts/ocq/build_qwen_metatool_b_ont.py \
  --model Qwen/Qwen2.5-7B \
  --device cuda:0 \
  --target-layers first1 \
  --ontology-json reports/axis2_theoretical_verification/metatool_ontology_v2.json \
  --out results/ocq/cross_model/qwen25_7b_B_ont_first1.pt \
  --diag reports/axis2_theoretical_verification/qwen25_7b_build_first1.json

python scripts/ocq/make_control_b_ont.py \
  --src results/ocq/cross_model/qwen25_7b_B_ont_first1.pt \
  --out results/ocq/cross_model/qwen25_7b_B_random_first1.pt \
  --mode random_orthonormal \
  --seed 17

python scripts/ocq/make_control_b_ont.py \
  --src results/ocq/cross_model/qwen25_7b_B_ont_first1.pt \
  --out results/ocq/cross_model/qwen25_7b_B_featshuffle_first1.pt \
  --mode feature_shuffle \
  --seed 23

run_eval() {
  local tag="$1"
  local b_ont="$2"
  local tool_name_mode="$3"
  CUDA_VISIBLE_DEVICES="$GPU_QWEN" \
  python scripts/ocq/eval_metatool_subtask1.py \
    --model Qwen/Qwen2.5-7B \
    --device cuda:0 \
    --dataset external/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json \
    --methods no_steer ocq_bias_a0.2 ocq_bias_a0.3 \
    --b-ont "$b_ont" \
    --max-samples "$MAX_SAMPLES" \
    --shuffle-seed "$SHUFFLE_SEED" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --scoring-mode first_line \
    --tool-name-mode "$tool_name_mode" \
    --dump-failures \
    --out "results/ocq/cross_model/${tag}_${RUN_SUFFIX}.json" \
    > "results/ocq/cross_model/${tag}_${RUN_SUFFIX}.log" 2>&1
}

run_eval "qwen25_7b_parser_safe_original" "results/ocq/cross_model/qwen25_7b_B_ont_first1.pt" "original"
run_eval "qwen25_7b_parser_safe_opaque" "results/ocq/cross_model/qwen25_7b_B_ont_first1.pt" "opaque_local"
run_eval "qwen25_7b_parser_safe_random" "results/ocq/cross_model/qwen25_7b_B_random_first1.pt" "original"
run_eval "qwen25_7b_parser_safe_random_opaque" "results/ocq/cross_model/qwen25_7b_B_random_first1.pt" "opaque_local"
run_eval "qwen25_7b_parser_safe_featshuffle" "results/ocq/cross_model/qwen25_7b_B_featshuffle_first1.pt" "original"
run_eval "qwen25_7b_parser_safe_featshuffle_opaque" "results/ocq/cross_model/qwen25_7b_B_featshuffle_first1.pt" "opaque_local"

echo "[ocq_e8_qwen_control_bundle] done"
