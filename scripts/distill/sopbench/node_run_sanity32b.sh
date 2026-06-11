#!/bin/bash
# node_run_sanity32b.sh — COWORKER plan v1.42 #0: 32B vanilla leaderboard-reproduction sanity.
# Serves Qwen2.5-32B on GPU0 and runs, in parallel against the same endpoint:
#   - arm1react / full / bank  (leaderboard anchor: README 40.30% must reproduce +-2pp)
#   - arm1 fc   / full / bank  (our FC base anchor for the 4-column table)
# Optionally extend: DOMAINS="bank dmv healthcare hotel library online_market university"
set -x
export HF_HUB_CACHE=/scratch/hf_cache
PY=/scratch/venvs/sop_env/bin/python
VLLM=/scratch/venvs/sop_env/bin/vllm
SW=/scratch/boltzmann-attention/scripts/distill/sopbench/run_arm3_sweep.sh
DOMAINS="${DOMAINS:-bank}"
PORT=9100

cd /scratch/SOPBench

# serve (skip if already up)
if ! curl -s http://localhost:$PORT/v1/models | grep -q qwen2.5-32b-instruct; then
  # TP2 (GPU0+3): 32B bf16 65.5GB + 32k-ctx KV won't fit one 80GB H100
  # both served names: the sweep uses the short tag but SOPBench's llm handler
  # canonicalizes to the full HF id (single name -> 404 on every task -> NA)
  CUDA_VISIBLE_DEVICES=${SANITY_GPUS:-0,3} nohup $VLLM serve Qwen/Qwen2.5-32B-Instruct \
    --served-model-name qwen2.5-32b-instruct Qwen/Qwen2.5-32B-Instruct \
    --port $PORT --tensor-parallel-size 2 \
    --dtype bfloat16 --gpu-memory-utilization 0.90 --max-model-len 32000 \
    --enable-auto-tool-choice --tool-call-parser hermes --trust-remote-code \
    > /scratch/logs/serve32b.log 2>&1 &
  for i in $(seq 1 120); do
    curl -s http://localhost:$PORT/v1/models | grep -q qwen2.5-32b-instruct && { echo SERVER_READY; break; }
    sleep 15
  done
fi

MODEL=qwen2.5-32b-instruct SOPBENCH_VLLM_BASE_URL=http://localhost:$PORT/v1 PY=$PY \
  ARMS="arm1react" DOMAINS="$DOMAINS" OUT=/scratch/sopbench_runs/sanity_react \
  bash $SW > /scratch/logs/sanity_react.log 2>&1 &
R1=$!
MODEL=qwen2.5-32b-instruct SOPBENCH_VLLM_BASE_URL=http://localhost:$PORT/v1 PY=$PY \
  ARMS="arm1" MODE=fc DOMAINS="$DOMAINS" OUT=/scratch/sopbench_runs/sanity_fc \
  bash $SW > /scratch/logs/sanity_fc.log 2>&1 &
R2=$!
wait $R1 $R2
echo "SANITY_DONE"
tail -5 /scratch/sopbench_runs/sanity_react/summary_*.tsv /scratch/sopbench_runs/sanity_fc/summary_*.tsv
