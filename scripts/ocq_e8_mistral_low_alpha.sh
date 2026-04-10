#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/scratch/boltzmann/ba-ocq-develop}"
VENV_DIR="${2:-/scratch/boltzmann/venvs/ocq}"
GPU_MISTRAL="${GPU_MISTRAL:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-995}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12}"
HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
DRY_RUN="${DRY_RUN:-0}"

run_cmd() {
  echo "+ $*"
  if [ "$DRY_RUN" != "1" ]; then
    "$@"
  fi
}

if [ "$DRY_RUN" != "1" ]; then
  source "$VENV_DIR/bin/activate"
fi
if [ -d "$ROOT_DIR" ]; then
  cd "$ROOT_DIR"
elif [ "$DRY_RUN" = "1" ]; then
  echo "[DRY_RUN] missing ROOT_DIR: $ROOT_DIR"
else
  echo "missing ROOT_DIR: $ROOT_DIR" >&2
  exit 1
fi
export HF_HOME
export TRANSFORMERS_CACHE="$HF_HOME"

mkdir -p results/ocq/cross_model reports/axis2_theoretical_verification

if [ ! -f results/ocq/cross_model/mistral_7b_v03_B_ont_first1.pt ]; then
  run_cmd python scripts/ocq/build_metatool_ontology_v2.py \
    --out reports/axis2_theoretical_verification/metatool_ontology_v2.json
  run_cmd env CUDA_VISIBLE_DEVICES="$GPU_MISTRAL" python scripts/ocq/build_qwen_metatool_b_ont.py \
    --model mistralai/Mistral-7B-v0.3 \
    --device cuda:0 \
    --target-layers first1 \
    --ontology-json reports/axis2_theoretical_verification/metatool_ontology_v2.json \
    --out results/ocq/cross_model/mistral_7b_v03_B_ont_first1.pt \
    --diag reports/axis2_theoretical_verification/mistral_7b_v03_build_first1.json
fi

echo "+ CUDA_VISIBLE_DEVICES=$GPU_MISTRAL python scripts/ocq/eval_metatool_subtask1.py --model mistralai/Mistral-7B-v0.3 --device cuda:0 --dataset external/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json --methods no_steer ocq_bias_a0.05 ocq_bias_a0.10 ocq_bias_a0.15 ocq_bias_a0.20 --b-ont results/ocq/cross_model/mistral_7b_v03_B_ont_first1.pt --max-samples $MAX_SAMPLES --max-new-tokens $MAX_NEW_TOKENS --out results/ocq/cross_model/mistral_7b_v03_metatool_low_alpha_${MAX_SAMPLES}.json > results/ocq/cross_model/mistral_7b_v03_metatool_low_alpha_${MAX_SAMPLES}.log"
if [ "$DRY_RUN" != "1" ]; then
  CUDA_VISIBLE_DEVICES="$GPU_MISTRAL" python scripts/ocq/eval_metatool_subtask1.py \
    --model mistralai/Mistral-7B-v0.3 \
    --device cuda:0 \
    --dataset external/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json \
    --methods no_steer ocq_bias_a0.05 ocq_bias_a0.10 ocq_bias_a0.15 ocq_bias_a0.20 \
    --b-ont results/ocq/cross_model/mistral_7b_v03_B_ont_first1.pt \
    --dump-failures \
    --max-samples "$MAX_SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --out "results/ocq/cross_model/mistral_7b_v03_metatool_low_alpha_${MAX_SAMPLES}.json" \
    > "results/ocq/cross_model/mistral_7b_v03_metatool_low_alpha_${MAX_SAMPLES}.log" 2>&1
fi
