#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/scratch/boltzmann/ba-ocq-develop}"
VENV_DIR="${2:-/scratch/boltzmann/venvs/ocq}"
GPU_LLAMA="${GPU_LLAMA:-0}"
GPU_MISTRAL="${GPU_MISTRAL:-1}"
HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
METATOOL_DIR="${METATOOL_DIR:-$ROOT_DIR/external/MetaTool/dataset}"
MAX_SAMPLES="${MAX_SAMPLES:-995}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12}"

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
export HF_HOME
export TRANSFORMERS_CACHE="$HF_HOME"
export METATOOL_DIR

python scripts/ocq/build_metatool_ontology_v2.py \
  --out reports/axis2_theoretical_verification/metatool_ontology_v2.json

mkdir -p results/ocq/cross_model

run_one() {
  local gpu="$1"
  local model_id="$2"
  local out_tag="$3"
  local sample_tag="$4"
  CUDA_VISIBLE_DEVICES="$gpu" \
  python scripts/ocq/build_qwen_metatool_b_ont.py \
    --model "$model_id" \
    --device cuda:0 \
    --target-layers first1 \
    --ontology-json reports/axis2_theoretical_verification/metatool_ontology_v2.json \
    --out "results/ocq/cross_model/${out_tag}_B_ont_first1.pt" \
    --diag "reports/axis2_theoretical_verification/${out_tag}_build_first1.json"

  CUDA_VISIBLE_DEVICES="$gpu" \
  python scripts/ocq/eval_metatool_subtask1.py \
    --model "$model_id" \
    --device cuda:0 \
    --dataset external/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json \
    --methods no_steer ocq_bias_a0.2 ocq_bias_a0.25 ocq_bias_a0.3 ocq_bias_a0.35 ocq_bias_a0.4 \
    --b-ont "results/ocq/cross_model/${out_tag}_B_ont_first1.pt" \
    --max-samples "$MAX_SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --out "results/ocq/cross_model/${out_tag}_metatool_alpha_sweep_${sample_tag}.json" \
    > "results/ocq/cross_model/${out_tag}_metatool_alpha_sweep_${sample_tag}.log" 2>&1
}

write_status() {
  local out_tag="$1"
  local model_id="$2"
  local status="$3"
  local note="$4"
  cat > "results/ocq/cross_model/${out_tag}_status.json" <<EOF
{"model":"$model_id","status":"$status","note":"$note","max_samples":$MAX_SAMPLES}
EOF
}

launch_one() {
  local gpu="$1"
  local model_id="$2"
  local out_tag="$3"
  local sample_tag="$4"
  (
    if run_one "$gpu" "$model_id" "$out_tag" "$sample_tag"; then
      write_status "$out_tag" "$model_id" "ok" "completed"
    else
      write_status "$out_tag" "$model_id" "failed" "see corresponding log/output for details"
    fi
  ) &
  echo $!
}

if [ "$MAX_SAMPLES" = "995" ]; then
  SAMPLE_TAG="995"
else
  SAMPLE_TAG="$MAX_SAMPLES"
fi

PID_LLAMA=$(launch_one "$GPU_LLAMA" "meta-llama/Meta-Llama-3.1-8B" "llama31_8b" "$SAMPLE_TAG")
PID_MISTRAL=$(launch_one "$GPU_MISTRAL" "mistralai/Mistral-7B-v0.3" "mistral_7b_v03" "$SAMPLE_TAG")

wait "$PID_LLAMA" || true
wait "$PID_MISTRAL" || true

echo "[ocq_e8_r1_cross_model] done"
echo "  Llama  -> results/ocq/cross_model/llama31_8b_metatool_alpha_sweep_${SAMPLE_TAG}.json"
echo "  Mistral-> results/ocq/cross_model/mistral_7b_v03_metatool_alpha_sweep_${SAMPLE_TAG}.json"
