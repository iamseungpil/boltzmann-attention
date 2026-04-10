#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/scratch/boltzmann/ba-ocq-develop}"
VENV_DIR="${2:-/scratch/boltzmann/venvs/ocq}"
GPU_META="${GPU_META:-0}"
GPU_MMLU="${GPU_MMLU:-1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B}"

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export METATOOL_DIR="${METATOOL_DIR:-$ROOT_DIR/external/MetaTool/dataset}"

mkdir -p "$ROOT_DIR" "$(dirname "$VENV_DIR")"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install transformers datasets accelerate sentencepiece safetensors scipy pandas

mkdir -p "$ROOT_DIR/external"
if [ ! -d "$ROOT_DIR/external/MetaTool/dataset" ]; then
  rm -rf "$ROOT_DIR/external/MetaTool"
  git clone https://github.com/HowieHwong/MetaTool.git "$ROOT_DIR/external/MetaTool"
fi

cd "$ROOT_DIR"

python scripts/ocq/build_metatool_ontology_v2.py \
  --out reports/axis2_theoretical_verification/metatool_ontology_v2.json

CUDA_VISIBLE_DEVICES="$GPU_META" \
python scripts/ocq/build_qwen_metatool_b_ont.py \
  --model "$MODEL_ID" \
  --device cuda:0 \
  --target-layers first1 \
  --ontology-json reports/axis2_theoretical_verification/metatool_ontology_v2.json \
  --out results/ocq/first1/B_ont_first1_full.pt \
  --diag reports/axis2_theoretical_verification/build_qwen_metatool_b_ont_first1_full.json

CUDA_VISIBLE_DEVICES="$GPU_META" \
python scripts/ocq/eval_metatool_subtask1.py \
  --model "$MODEL_ID" \
  --device cuda:0 \
  --dataset external/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json \
  --methods no_steer ocq_bias_a0.3 ocq_facet_gated_a0.3 ocq_facet_gated_a1.0 \
  --b-ont results/ocq/first1/B_ont_first1_full.pt \
  --max-samples 200 \
  --max-new-tokens 12 \
  --out results/ocq/first1/metatool_eval_200_e8.json \
  > results/ocq/first1/metatool_eval_200_e8.log 2>&1 &
PID_META=$!

CUDA_VISIBLE_DEVICES="$GPU_MMLU" \
python scripts/ocq/eval_mmlu_subset.py \
  --model "$MODEL_ID" \
  --device cuda:0 \
  --methods no_steer ocq_bias_a0.3 ocq_facet_gated_a0.3 \
  --b-ont results/ocq/first1/B_ont_first1_full.pt \
  --n-samples 500 \
  --n-shot 1 \
  --out results/ocq/first1/mmlu_eval_500_e8.json \
  > results/ocq/first1/mmlu_eval_500_e8.log 2>&1 &
PID_MMLU=$!

wait "$PID_META"
wait "$PID_MMLU"

echo "[ocq_e8_wave] done"
echo "  MetaTool: $ROOT_DIR/results/ocq/first1/metatool_eval_200_e8.json"
echo "  MMLU:     $ROOT_DIR/results/ocq/first1/mmlu_eval_500_e8.json"
