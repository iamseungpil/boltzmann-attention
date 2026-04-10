#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/scratch/boltzmann/ba-ocq-develop}"
VENV_DIR="${2:-/scratch/boltzmann/venvs/ocq}"
GPU_MISTRAL="${GPU_MISTRAL:-1}"
GPU_QWEN="${GPU_QWEN:-2}"

if [ ! -d "$ROOT_DIR" ]; then
  echo "missing ROOT_DIR: $ROOT_DIR" >&2
  exit 1
fi

cd "$ROOT_DIR"
mkdir -p results/ocq/cross_model

if pgrep -af "mistral_7b_v03_metatool_low_alpha_995|ocq_e8_mistral_low_alpha.sh" >/dev/null; then
  echo "mistral low-alpha run already active" >&2
  exit 1
fi
if pgrep -af "qwen25_7b_mmlu_safety_1000|ocq_e8_qwen_mmlu_safety.sh" >/dev/null; then
  echo "qwen MMLU safety run already active" >&2
  exit 1
fi

nohup env GPU_MISTRAL="$GPU_MISTRAL" MAX_SAMPLES=995 \
  bash scripts/ocq_e8_mistral_low_alpha.sh "$ROOT_DIR" "$VENV_DIR" \
  > results/ocq/cross_model/mistral_7b_v03_low_alpha_launcher.out 2>&1 < /dev/null &
PID_MISTRAL=$!

nohup env GPU_QWEN="$GPU_QWEN" N_SAMPLES=1000 \
  bash scripts/ocq_e8_qwen_mmlu_safety.sh "$ROOT_DIR" "$VENV_DIR" \
  > results/ocq/cross_model/qwen25_7b_mmlu_safety_launcher.out 2>&1 < /dev/null &
PID_QWEN=$!

echo "[ocq_e8_followup_wave] launched"
echo "  mistral_low_alpha pid=$PID_MISTRAL"
echo "  qwen_mmlu_safety pid=$PID_QWEN"
