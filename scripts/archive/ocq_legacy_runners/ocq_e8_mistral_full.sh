#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/scratch/boltzmann/ba-ocq-develop}"
VENV_DIR="${2:-/scratch/boltzmann/venvs/ocq}"
GPU_MISTRAL="${GPU_MISTRAL:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-995}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12}"
HF_HOME="${HF_HOME:-/root/.cache/huggingface}"

source "$VENV_DIR/bin/activate"
cd "$ROOT_DIR"
export HF_HOME
export TRANSFORMERS_CACHE="$HF_HOME"

mkdir -p results/ocq/cross_model

python scripts/ocq/build_metatool_ontology_v2.py \
  --out reports/axis2_theoretical_verification/metatool_ontology_v2.json

CUDA_VISIBLE_DEVICES="$GPU_MISTRAL" \
python scripts/ocq/build_qwen_metatool_b_ont.py \
  --model mistralai/Mistral-7B-v0.3 \
  --device cuda:0 \
  --target-layers first1 \
  --ontology-json reports/axis2_theoretical_verification/metatool_ontology_v2.json \
  --out results/ocq/cross_model/mistral_7b_v03_B_ont_first1.pt \
  --diag reports/axis2_theoretical_verification/mistral_7b_v03_build_first1.json

CUDA_VISIBLE_DEVICES="$GPU_MISTRAL" \
python scripts/ocq/eval_metatool_subtask1.py \
  --model mistralai/Mistral-7B-v0.3 \
  --device cuda:0 \
  --dataset external/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json \
  --methods no_steer ocq_bias_a0.2 ocq_bias_a0.25 ocq_bias_a0.3 ocq_bias_a0.35 ocq_bias_a0.4 \
  --b-ont results/ocq/cross_model/mistral_7b_v03_B_ont_first1.pt \
  --max-samples "$MAX_SAMPLES" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --out results/ocq/cross_model/mistral_7b_v03_metatool_alpha_sweep_${MAX_SAMPLES}.json \
  > results/ocq/cross_model/mistral_7b_v03_metatool_alpha_sweep_${MAX_SAMPLES}.log 2>&1
