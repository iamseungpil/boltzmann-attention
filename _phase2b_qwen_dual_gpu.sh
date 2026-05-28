#!/bin/bash
# Phase 2b — Qwen-7B dual-GPU: relation sweep + layer sweep in parallel.
#
# CONDITIONS IDENTICAL to baseline B0 (same as Phase 2a orchestrator):
#   - served-model-name = Qwen2.5-7B-Instruct
#   - port = 9000 / 9001 (one server per GPU)
#   - agent-llm = openai/Qwen2.5-7B-Instruct
#   - user-llm = openai/openai/gpt-4o-mini
#   - num-trials = 4 (matching baseline)
#   - max-steps = 200, max-concurrency = 3, timeout = 600s
#   - Same 30 stratified tasks
#
# Strategy:
#   GPU0:9000 — relation sweep (5 relations × α=0.5 × layers=[12,13,14])
#               relations: contradicts, enables, requires, step_realizes_tool, subtask_of
#   GPU1:9001 — layer    sweep (5 layer configs × α=0.5 × relation=validates)
#               layers:    [2,3,4], [7,8,9], [12,13,14], [17,18,19], [22,23,24]
#
# Each sweep restarts the vLLM server per config (steering baked at startup).
#
# Override env vars:
#   ALPHA=0.3                  override fixed alpha
#   RELATIONS_LIST="..."       override 5 relations
#   LAYERS_LIST="..."          override 5 layer configs
#   TRIALS=2                   speed up
#
# Prerequisite: Phase 2a complete (steering_vectors_qwen7b.pt exists, vLLM servers stopped)
set -uo pipefail
cd /home/woori/workspace_common/boltzmann-attention-pi

PHASE2="reports/facet_rft_2026/phase2_steering"
mkdir -p "$PHASE2"
LOG="$PHASE2/_phase2b_qwen_$(date +%Y%m%d_%H%M%S).log"
exec >>"$LOG" 2>&1
echo "[phase2b] start $(date)"

VLLM_PY=/home/woori/venvs/tau2_vllm_env/bin/python
TAU2_PY=/home/woori/venvs/seka_env/bin/python
SERVER_PY=_steering_vllm_server.py
RUNNER=scripts/phase1_runner.py
TASK_IDS="$PHASE2/stratified_task_ids.json"

if [ ! -f "$PHASE2/steering_vectors_qwen7b.pt" ]; then
  echo "[phase2b] ERROR steering_vectors_qwen7b.pt missing" >&2; exit 1
fi
if [ ! -f "$TASK_IDS" ]; then
  echo "[phase2b] ERROR task ids missing"; exit 1
fi

if [ -f /home/woori/.openrouter_key ]; then
  set -a; . /home/woori/.openrouter_key; set +a
fi
OR_KEY="$OPENROUTER_API_KEY"

ALPHA=${ALPHA:-0.5}
TRIALS=${TRIALS:-4}
MAX_STEPS=200
CONC=3
TIMEOUT=600

# Default sweeps; override via env (space-separated)
RELATIONS_LIST=${RELATIONS_LIST:-"contradicts enables requires step_realizes_tool subtask_of"}
LAYERS_LIST=${LAYERS_LIST:-"2,3,4 7,8,9 12,13,14 17,18,19 22,23,24"}

# ============================================================
# Helper: launch one server + sims for one config, then kill
# ============================================================
run_config() {
  # $1: tag  $2: relation  $3: layers  $4: port  $5: gpu_idx  $6: master_port
  local tag="$1" rel="$2" layers="$3" port="$4" gpu_idx="$5" mport="$6"
  local ts=$(date +%Y%m%d_%H%M%S)
  local out_dir="$PHASE2/phase2b_${tag}_${ts}"
  local srv_log="$PHASE2/phase2b_${tag}_${ts}_server.log"
  mkdir -p "$out_dir"
  echo "{\"tag\":\"$tag\",\"relation\":\"$rel\",\"alpha\":$ALPHA,\"layers\":[$layers]}" > "$out_dir/steering.json"

  echo "[$tag] start rel=$rel layers=$layers α=$ALPHA port=$port gpu=$gpu_idx mport=$mport $(date)"

  CUDA_VISIBLE_DEVICES="$gpu_idx" MASTER_PORT="$mport" VLLM_PORT="$mport" \
    nohup "$VLLM_PY" "$SERVER_PY" \
    --steering-vectors "$PHASE2/steering_vectors_qwen7b.pt" \
    --relation "$rel" --alpha "$ALPHA" --layers "$layers" \
    -- \
    --model Qwen/Qwen2.5-7B-Instruct --served-model-name Qwen2.5-7B-Instruct \
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
    echo "[$tag] :$port FAILED"; kill -KILL $SRV 2>/dev/null || true; return 1
  fi

  # phase1_runner with BASELINE-EQUIVALENT flags
  "$TAU2_PY" "$RUNNER" \
    --variants B0 --task-split base \
    --base-url "http://127.0.0.1:${port}/v1" \
    --agent-llm openai/Qwen2.5-7B-Instruct \
    --user-llm openai/openai/gpt-4o-mini \
    --user-base-url https://openrouter.ai/api/v1 \
    --user-api-key "$OR_KEY" \
    --domain telecom --num-trials $TRIALS \
    --max-steps $MAX_STEPS --max-concurrency $CONC \
    --timeout $TIMEOUT --auto-resume \
    --task-ids-file "$TASK_IDS" \
    --out-dir "$out_dir" \
    >> "$out_dir/run.log" 2>&1
  local rc=$?
  echo "[$tag] sims_done rc=$rc out=$out_dir"

  kill -TERM $SRV 2>/dev/null || true
  sleep 5
  kill -KILL $SRV 2>/dev/null || true
  sleep 5
}

# Relation sweep on GPU0:9000 (sequential within GPU0)
(
  for rel in $RELATIONS_LIST; do
    run_config "relsweep_${rel}" "$rel" "12,13,14" 9000 0 8010
  done
) &
PIDA=$!

# Layer sweep on GPU1:9001 (sequential within GPU1)
(
  for lyrs in $LAYERS_LIST; do
    safe="$(echo $lyrs | tr ',' '-')"
    run_config "layersweep_L${safe}" "validates" "$lyrs" 9001 1 8011
  done
) &
PIDB=$!

echo "[phase2b] dispatched relsweep_pid=$PIDA layersweep_pid=$PIDB"
wait $PIDA; RA=$?
wait $PIDB; RB=$?
echo "[phase2b] relsweep rc=$RA layersweep rc=$RB"

date > "$PHASE2/phase2b_done.txt"
echo "[phase2b] DONE $(date)"
