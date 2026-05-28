#!/bin/bash
# Phase 2a sanity check — α=0 only (steering disabled) on both Qwen and Hermes.
# Verifies that vLLM steering server == vanilla vLLM baseline on stratified subset.
#
# Expected:
#   Qwen   α=0 pass^1 ≈ 0.183 (subset baseline from Qwen+OR B0)
#   Hermes α=0 pass^1 ≈ 0.125 (subset baseline from Hermes+OR B0)
#
# Both run in parallel (Qwen GPU0, Hermes GPU1).
set -uo pipefail
cd /home/woori/workspace_common/boltzmann-attention-pi

PHASE2="reports/facet_rft_2026/phase2_steering"
mkdir -p "$PHASE2"
LOG="$PHASE2/_sanity_a0_$(date +%Y%m%d_%H%M%S).log"
exec >>"$LOG" 2>&1

echo "[sanity-a0] start $(date)"

VLLM_PY=/home/woori/venvs/tau2_vllm_env/bin/python  # has vllm
TAU2_PY=/home/woori/venvs/seka_env/bin/python        # has tau2
SERVER_PY=_steering_vllm_server.py
RUNNER=scripts/phase1_runner.py
TASK_IDS="$PHASE2/stratified_task_ids.json"

if [ -f /home/woori/.openrouter_key ]; then
  set -a; . /home/woori/.openrouter_key; set +a
fi
OR_KEY="$OPENROUTER_API_KEY"

run_one() {
  # $1: tag  $2: hf model  $3: served  $4: device  $5: port  $6: master_port
  local tag="$1" hf_model="$2" served="$3" device="$4" port="$5" mport="${6:-8010}"
  local ts=$(date +%Y%m%d_%H%M%S)
  local out_dir="$PHASE2/sanity_a0_${tag}_${ts}"
  local srv_log="$PHASE2/sanity_a0_${tag}_${ts}_server.log"
  mkdir -p "$out_dir"
  local gpu_idx="${device#cuda:}"

  echo "[$tag] === α=0 sanity start $(date) ==="
  CUDA_VISIBLE_DEVICES="$gpu_idx" MASTER_PORT="$mport" VLLM_PORT="$mport" nohup "$VLLM_PY" "$SERVER_PY" \
    --alpha 0.0 -- \
    --model "$hf_model" --served-model-name "$served" \
    --port "$port" --host 127.0.0.1 \
    --dtype bfloat16 --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --tool-call-parser hermes --enable-auto-tool-choice \
    > "$srv_log" 2>&1 &
  local SRV=$!
  echo "[$tag] vllm pid=$SRV log=$srv_log"

  local ready=0
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[$tag] :$port READY after ${i}s"; ready=1; break
    fi
    sleep 2
  done
  if [ "$ready" -ne 1 ]; then
    echo "[$tag] :$port FAILED to be ready"
    kill -KILL $SRV 2>/dev/null || true
    return 1
  fi

  "$TAU2_PY" "$RUNNER" \
    --variants B0 --task-split base \
    --base-url "http://127.0.0.1:${port}/v1" \
    --agent-llm "openai/${served}" \
    --user-llm openrouter/openai/gpt-4o-mini \
    --user-base-url https://openrouter.ai/api/v1 \
    --user-api-key "$OR_KEY" \
    --domain telecom --num-trials 4 \
    --max-steps 200 --max-concurrency 3 \
    --timeout 600 --auto-resume \
    --task-ids-file "$TASK_IDS" \
    --out-dir "$out_dir" \
    >> "$out_dir/run.log" 2>&1
  local RC=$?
  echo "[$tag] sims_done rc=$RC out=$out_dir"

  kill -TERM $SRV 2>/dev/null || true
  sleep 5
  kill -KILL $SRV 2>/dev/null || true
  echo "[$tag] α=0 DONE $(date)"
}

( run_one qwen7b  Qwen/Qwen2.5-7B-Instruct        qwen7b-steer  cuda:0 9200 8010 ) &
QWEN=$!
( run_one hermes3 NousResearch/Hermes-3-Llama-3.1-8B hermes3-steer cuda:1 9201 8011 ) &
HERMES=$!

wait $QWEN; QRC=$?
wait $HERMES; HRC=$?
echo "[sanity-a0] qwen rc=$QRC hermes rc=$HRC"

date > "$PHASE2/sanity_a0_done.txt"
echo "[sanity-a0] DONE $(date)"
