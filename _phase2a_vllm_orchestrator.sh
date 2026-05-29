#!/bin/bash
# Phase 2a VLLM-based parallel orchestrator.
#
# Strategy: for each model (Qwen GPU0, Hermes-3 GPU1) sequentially scan α
# values. For each α value, start a fresh vLLM server with steering baked in,
# wait for /health, run 120 sims (30 tasks × 4 trials) via phase1_runner.
#
# CONDITIONS IDENTICAL to baseline B0 (so α=0 ≈ baseline subset):
#   - served-model-name = Qwen2.5-7B-Instruct / Hermes-3-Llama-3.1-8B
#   - port = 9000 / 9001 (baseline ports)
#   - agent-llm = openai/{served-model-name}
#   - user-llm = openai/openai/gpt-4o-mini (LiteLLM openai/ provider, not openrouter/)
#   - num-trials = 4 (matching baseline)
#   - max-steps = 200, max-concurrency = 3, timeout = 600s
#   - Same 30 stratified tasks (subset of phase1 B0)
#
# Override env vars:
#   ALPHAS="0.0 0.1 0.3"   to test fewer alphas
#   TRIALS=2               to speed up at cost of N
#
# Both models run in PARALLEL (independent loops on GPU0 vs GPU1).
set -uo pipefail
cd /home/woori/workspace_common/boltzmann-attention-pi

PHASE2="reports/facet_rft_2026/phase2_steering"
mkdir -p "$PHASE2"
LOG="$PHASE2/_orch_vllm_$(date +%Y%m%d_%H%M%S).log"
exec >>"$LOG" 2>&1

echo "[orch-vllm] start $(date)"

VLLM_PY=/home/woori/venvs/tau2_vllm_env/bin/python  # has vllm
TAU2_PY=/home/woori/venvs/seka_env/bin/python        # has tau2
SERVER_PY=_steering_vllm_server.py
RUNNER=scripts/phase1_runner.py
TASK_IDS="$PHASE2/stratified_task_ids.json"
ALPHAS=(${ALPHAS:-0.0 0.1 0.3 0.5 1.0 2.0})  # override via env
N_TASKS=30  # via --task-ids-file (30 stratified)
TRIALS=${TRIALS:-4}  # MATCH BASELINE B0
MAX_STEPS=200
CONC=3
SIM_TIMEOUT=600

if [ ! -f "$TASK_IDS" ]; then
  echo "[orch-vllm] ERROR task ids file missing: $TASK_IDS"; exit 1
fi

# OpenRouter key
if [ -f /home/woori/.openrouter_key ]; then
  set -a; . /home/woori/.openrouter_key; set +a
fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "[orch-vllm] ERROR OPENROUTER_API_KEY missing"; exit 1
fi
OR_KEY="$OPENROUTER_API_KEY"

# ---- single model loop: starts vllm for each alpha, runs sims, kills ----
run_model_loop() {
  # $1: tag (qwen7b|hermes3)
  # $2: hf model id
  # $3: served-model-name
  # $4: device (cuda:0 / cuda:1)
  # $5: vllm port
  # $6: tool-call-parser (hermes)
  # $7: relation (e.g. validates)
  # $8: layers (e.g. "12,13,14")
  # $9: master_port (for vLLM distributed init, must differ per instance)
  local tag="$1" hf_model="$2" served="$3" device="$4" port="$5" parser="$6" rel="$7" layers="$8" mport="${9:-8010}"
  local vec="$PHASE2/steering_vectors_${tag}.pt"
  if [ ! -f "$vec" ]; then echo "[$tag] missing vec $vec"; return 1; fi

  for alpha in "${ALPHAS[@]}"; do
    local ts=$(date +%Y%m%d_%H%M%S)
    local out_dir="$PHASE2/vllm_grid_${tag}_a${alpha}_${ts}"
    local srv_log="$PHASE2/vllm_grid_${tag}_a${alpha}_${ts}_server.log"
    mkdir -p "$out_dir"
    echo "{\"tag\":\"$tag\",\"alpha\":$alpha,\"relation\":\"$rel\",\"layers\":[$layers]}" > "$out_dir/steering.json"

    echo "[$tag] === alpha=$alpha start $(date) ==="

    # set CUDA_VISIBLE_DEVICES to a single index for this server
    local gpu_idx="${device#cuda:}"

    # RACE FIX: wait until target port is free so /health cannot pass against a
    # stale server being torn down on the same port (Phase 2a ext bug 2026-05-29).
    for i in $(seq 1 60); do
      if ss -ltn 2>/dev/null | grep -q ":${port} "; then
        echo "[$tag] :$port still bound, waiting for teardown (${i})"; sleep 3
      else break; fi
    done
    # start vllm server with steering baked in
    CUDA_VISIBLE_DEVICES="$gpu_idx" MASTER_PORT="$mport" VLLM_PORT="$mport" nohup "$VLLM_PY" "$SERVER_PY" \
      --steering-vectors "$vec" \
      --relation "$rel" --alpha "$alpha" --layers "$layers" \
      -- \
      --model "$hf_model" --served-model-name "$served" \
      --port "$port" --host 127.0.0.1 \
      --dtype bfloat16 --max-model-len 32768 \
      --gpu-memory-utilization 0.85 \
      --tool-call-parser "$parser" --enable-auto-tool-choice \
      > "$srv_log" 2>&1 &
    local SRV=$!
    echo "[$tag] vllm pid=$SRV log=$srv_log"

    # wait for /health
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
      continue
    fi

    # run sims
    "$TAU2_PY" "$RUNNER" \
      --variants B0 --task-split base \
      --base-url "http://127.0.0.1:${port}/v1" \
      --agent-llm "openai/${served}" \
      --user-llm openai/openai/gpt-4o-mini \
      --user-base-url https://openrouter.ai/api/v1 \
      --user-api-key "$OR_KEY" \
      --domain telecom --num-trials $TRIALS \
      --max-steps $MAX_STEPS --max-concurrency $CONC \
      --timeout $SIM_TIMEOUT --auto-resume \
      --task-ids-file "$TASK_IDS" \
      --out-dir "$out_dir" \
      >> "$out_dir/run.log" 2>&1
    local RC=$?
    echo "[$tag] alpha=$alpha sims_done rc=$RC out=$out_dir"

    # tear down server
    kill -TERM $SRV 2>/dev/null || true
    sleep 5
    kill -KILL $SRV 2>/dev/null || true
    # wait until GPU memory freed
    sleep 5
  done
}

# MODEL env var controls which model loop(s) to run:
#   MODEL=qwen7b    — only Qwen on GPU0
#   MODEL=hermes3   — only Hermes on GPU1
#   MODEL=both      — both in parallel (default)
MODEL="${MODEL:-both}"
QWEN_PID=""
HERMES_PID=""
if [ "$MODEL" = "qwen7b" ] || [ "$MODEL" = "both" ]; then
  ( run_model_loop qwen7b  Qwen/Qwen2.5-7B-Instruct        Qwen2.5-7B-Instruct   cuda:0 9000 hermes validates "12,13,14" 8010 ) &
  QWEN_PID=$!
fi
if [ "$MODEL" = "hermes3" ] || [ "$MODEL" = "both" ]; then
  ( run_model_loop hermes3 NousResearch/Hermes-3-Llama-3.1-8B Hermes-3-Llama-3.1-8B cuda:1 9001 hermes validates "14,15,16" 8011 ) &
  HERMES_PID=$!
fi

echo "[orch-vllm] MODEL=$MODEL qwen_loop=$QWEN_PID hermes_loop=$HERMES_PID"
QRC=0; HRC=0
if [ -n "$QWEN_PID" ]; then wait $QWEN_PID; QRC=$?; fi
if [ -n "$HERMES_PID" ]; then wait $HERMES_PID; HRC=$?; fi
echo "[orch-vllm] qwen rc=$QRC hermes rc=$HRC"

# Mark done — per-model marker so a parallel run on the OTHER GPU can
# write its own marker independently.
case "$MODEL" in
  qwen7b)  date > "$PHASE2/phase2a_vllm_done_qwen7b.txt" ;;
  hermes3) date > "$PHASE2/phase2a_vllm_done_hermes3.txt" ;;
  both)    date > "$PHASE2/phase2a_vllm_done.txt" ;;
esac
echo "[orch-vllm] DONE $(date)"
