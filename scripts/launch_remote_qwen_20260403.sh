#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/boltzmann-attention-v3-repro"
PY="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/logs"
OUT_DIR="$ROOT/results/v3"

mkdir -p "$LOG_DIR" "$OUT_DIR" /scratch/hf_cache

export HF_HOME=/scratch/hf_cache
export TRANSFORMERS_CACHE=/scratch/hf_cache
export HUGGINGFACE_HUB_CACHE=/scratch/hf_cache

nohup env CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/scripts/exp4_2_v3_full_quant_ppl.py" \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --model-key qwen2.5-7b-ham-bounded-20260403 \
  --device cuda:0 --protocol post_rope --context-len 256 --max-eval-tokens 512 \
  --bits 2 3 --benchmark-preset hamiltonian_descriptive \
  --output-dir "$OUT_DIR" \
  > "$LOG_DIR/qwen_ham_bounded_20260403.log" 2>&1 &
PID0=$!

nohup env CUDA_VISIBLE_DEVICES=1 "$PY" "$ROOT/scripts/exp4_2_v3_full_quant_ppl.py" \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --model-key qwen2.5-7b-promo-1024-20260403 \
  --device cuda:0 --protocol post_rope --context-len 1024 --max-eval-tokens 4096 \
  --methods fp16 kivi_residual turboquant_rand fokvq fokvq_e2 \
  --bits 2 3 --benchmark-preset ppl_quality \
  --output-dir "$OUT_DIR" \
  > "$LOG_DIR/qwen_promo_1024_20260403.log" 2>&1 &
PID1=$!

nohup env CUDA_VISIBLE_DEVICES=2 "$PY" "$ROOT/scripts/exp4_2_v3_full_quant_ppl.py" \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --model-key qwen2.5-7b-promo-2048-20260403 \
  --device cuda:0 --protocol post_rope --context-len 2048 --max-eval-tokens 8192 \
  --methods fp16 kivi_residual turboquant_rand fokvq fokvq_e2 \
  --bits 2 3 --benchmark-preset ppl_quality \
  --output-dir "$OUT_DIR" \
  > "$LOG_DIR/qwen_promo_2048_20260403.log" 2>&1 &
PID2=$!

nohup env CUDA_VISIBLE_DEVICES=3 "$PY" "$ROOT/scripts/exp4_2_v3_full_quant_ppl.py" \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --model-key qwen2.5-7b-retrieval-20260403 \
  --device cuda:0 --protocol post_rope --mode niah --context-len 2048 \
  --methods fp16 kivi_residual turboquant_rand fokvq_e2 \
  --bits 2 3 --benchmark-preset custom \
  --niah-context-lens 4096 8192 --niah-depths 0.1 0.5 0.9 --niah-repeats 1 \
  --output-dir "$OUT_DIR" \
  > "$LOG_DIR/qwen_retrieval_20260403.log" 2>&1 &
PID3=$!

sleep 5
printf 'PID0=%s\nPID1=%s\nPID2=%s\nPID3=%s\n' "$PID0" "$PID1" "$PID2" "$PID3"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
