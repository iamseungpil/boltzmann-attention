#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/scratch/boltzmann/ba-ocq-develop}"
VENV_DIR="${2:-/scratch/boltzmann/venvs/ocq}"
GPU_QWEN="${GPU_QWEN:-2}"
N_SAMPLES="${N_SAMPLES:-1000}"
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

if [ ! -f results/ocq/cross_model/qwen25_7b_B_ont_first1.pt ]; then
  run_cmd python scripts/ocq/build_metatool_ontology_v2.py \
    --out reports/axis2_theoretical_verification/metatool_ontology_v2.json
  run_cmd env CUDA_VISIBLE_DEVICES="$GPU_QWEN" python scripts/ocq/build_qwen_metatool_b_ont.py \
    --model Qwen/Qwen2.5-7B \
    --device cuda:0 \
    --target-layers first1 \
    --ontology-json reports/axis2_theoretical_verification/metatool_ontology_v2.json \
    --out results/ocq/cross_model/qwen25_7b_B_ont_first1.pt \
    --diag reports/axis2_theoretical_verification/qwen25_7b_build_first1.json
fi

echo "+ CUDA_VISIBLE_DEVICES=$GPU_QWEN python scripts/ocq/eval_mmlu_subset.py --model Qwen/Qwen2.5-7B --device cuda:0 --methods no_steer ocq_bias_a0.2 ocq_bias_a0.3 --b-ont results/ocq/cross_model/qwen25_7b_B_ont_first1.pt --n-samples $N_SAMPLES --n-shot 5 --seed 42 --out results/ocq/cross_model/qwen25_7b_mmlu_safety_${N_SAMPLES}.json > results/ocq/cross_model/qwen25_7b_mmlu_safety_${N_SAMPLES}.log"
if [ "$DRY_RUN" != "1" ]; then
  CUDA_VISIBLE_DEVICES="$GPU_QWEN" python scripts/ocq/eval_mmlu_subset.py \
    --model Qwen/Qwen2.5-7B \
    --device cuda:0 \
    --methods no_steer ocq_bias_a0.2 ocq_bias_a0.3 \
    --b-ont results/ocq/cross_model/qwen25_7b_B_ont_first1.pt \
    --n-samples "$N_SAMPLES" \
    --n-shot 5 \
    --seed 42 \
    --out "results/ocq/cross_model/qwen25_7b_mmlu_safety_${N_SAMPLES}.json" \
    > "results/ocq/cross_model/qwen25_7b_mmlu_safety_${N_SAMPLES}.log" 2>&1
fi
