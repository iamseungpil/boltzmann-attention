#!/usr/bin/env bash
# GPU당 독립 엔진 (TP=1 · 로컬과 수치 동일). $1=GPU index $2=port
export HF_HOME=/workspace/.hf_home
export CUDA_VISIBLE_DEVICES=$1
exec ~/venv_vllm/bin/vllm serve Qwen/Qwen3.8-27B-FP8   --port $2 --enable-auto-tool-choice   --tool-call-parser qwen3_coder --reasoning-parser qwen3   --max-model-len 131072 --gpu-memory-utilization 0.9   --enable-prefix-caching --max-num-seqs 128
