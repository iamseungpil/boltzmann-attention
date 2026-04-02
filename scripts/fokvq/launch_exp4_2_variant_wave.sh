#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <bits> <output_root> [repo_root]" >&2
  exit 1
fi

BITS="$1"
OUTPUT_ROOT="$2"
REPO_ROOT="${3:-$(cd "$(dirname "$0")/../.." && pwd)}"

SCRIPT_PATH="$REPO_ROOT/scripts/fokvq/exp4_2_standard_ppl_benchmark.py"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

COMMON_ARGS=(
  --model-name openai-community/gpt2-medium
  --model-key "gpt2-medium-${BITS}bit"
  --calibration-samples 8
  --calibration-max-len 512
  --max-eval-tokens 4096
  --attn-implementation eager
  --seed 42
  --bits "$BITS"
  --fokvq-topk-frac 0.5
  --fokvq-adaptive-energy-frac 0.9
  --fokvq-clip-quantile 0.995
)

mkdir -p \
  "$OUTPUT_ROOT/main" \
  "$OUTPUT_ROOT/axis_ablation" \
  "$OUTPUT_ROOT/longctx512" \
  "$OUTPUT_ROOT/longctx1024"

launch_job() {
  local gpu="$1"
  local out_dir="$2"
  shift 2
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python "$SCRIPT_PATH" "${COMMON_ARGS[@]}" "$@" --output-dir "$out_dir"
  ) >"$out_dir/run.log" 2>&1 &
  local pid=$!
  echo "$pid" >"$out_dir/run.pid"
  echo "launched gpu=$gpu pid=$pid out_dir=$out_dir"
}

launch_job 0 "$OUTPUT_ROOT/main" \
  --device cuda:0 \
  --context-len 256 \
  --stride 128 \
  --methods fp16 uniform kivi turboquant fokvq fokvq_lloyd fokvq_adaptive fokvq_clip

launch_job 1 "$OUTPUT_ROOT/axis_ablation" \
  --device cuda:0 \
  --context-len 256 \
  --stride 128 \
  --methods fp16 identity random fokvq fokvq_lloyd fokvq_adaptive fokvq_clip

launch_job 2 "$OUTPUT_ROOT/longctx512" \
  --device cuda:0 \
  --context-len 512 \
  --stride 256 \
  --methods fp16 uniform kivi turboquant fokvq fokvq_lloyd fokvq_adaptive fokvq_clip

launch_job 3 "$OUTPUT_ROOT/longctx1024" \
  --device cuda:0 \
  --context-len 1024 \
  --stride 512 \
  --methods fp16 uniform kivi turboquant fokvq fokvq_lloyd fokvq_adaptive fokvq_clip

wait
