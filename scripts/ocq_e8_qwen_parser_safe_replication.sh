#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/scratch/boltzmann/ba-ocq-develop}"
VENV_DIR="${2:-/scratch/boltzmann/venvs/ocq}"
GPU_QWEN="${GPU_QWEN:-3}"
MAX_SAMPLES="${MAX_SAMPLES:-995}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12}"
SCORING_MODE="${SCORING_MODE:-first_line}"
LOGPROB_NORMALIZE="${LOGPROB_NORMALIZE:-mean}"
HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
SHUFFLE_SEED="${SHUFFLE_SEED:-13}"

source "$VENV_DIR/bin/activate"
cd "$ROOT_DIR"
export HF_HOME
export TRANSFORMERS_CACHE="$HF_HOME"

mkdir -p results/ocq/cross_model reports/axis2_theoretical_verification

if [ ! -f results/ocq/cross_model/qwen25_7b_B_ont_first1.pt ]; then
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
fi

CUDA_VISIBLE_DEVICES="$GPU_QWEN" \
python scripts/ocq/eval_metatool_subtask1.py \
  --model Qwen/Qwen2.5-7B \
  --device cuda:0 \
  --dataset external/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json \
  --methods no_steer ocq_bias_a0.2 ocq_bias_a0.3 \
  --b-ont results/ocq/cross_model/qwen25_7b_B_ont_first1.pt \
  --max-samples "$MAX_SAMPLES" \
  --shuffle-seed "$SHUFFLE_SEED" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --scoring-mode "$SCORING_MODE" \
  --logprob-normalize "$LOGPROB_NORMALIZE" \
  --tool-name-mode original \
  --dump-failures \
  --out "results/ocq/cross_model/qwen25_7b_parser_safe_replication_${MAX_SAMPLES}.json" \
  > "results/ocq/cross_model/qwen25_7b_parser_safe_replication_${MAX_SAMPLES}.log" 2>&1

echo "[ocq_e8_qwen_parser_safe_replication] done"
