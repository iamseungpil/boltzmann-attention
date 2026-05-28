#!/bin/bash
# Launch vLLM serving Qwen2.5-7B-Instruct on GPU1, port 9000, max-model-len 32768
# v2 settings:
#   - GPU1 (GPU0 occupied by mais user's vllm on port 8010 since 2026-05-27 08:41)
#   - port 9000 (avoid conflict with mais's 8010 and old 8000)
#   - max-model-len 32768 (v1 had 16384 -> 18 ContextWindowExceededError in B0)
set -e

LOG=/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/phase1_baseline/vllm_v2_32k.log
PIDFILE=/tmp/vllm_qwen.pid
mkdir -p "$(dirname "$LOG")"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "vLLM already running pid=$(cat $PIDFILE)"
  exit 0
fi

export CUDA_VISIBLE_DEVICES=1
nohup /home/woori/venvs/tau2_vllm_env/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --served-model-name Qwen2.5-7B-Instruct \
  --port 9000 \
  --host 127.0.0.1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "vLLM launched pid=$(cat $PIDFILE) log=$LOG"
