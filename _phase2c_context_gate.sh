#!/bin/bash
# Phase 2c: H4 context-gating comparison (Qwen, GPU0, alpha fixed).
# Compares gate modes against raw-add (none) at matched N, using the SAFE
# gated steering server. Baseline-equivalent vLLM config (port 9000, cuda:0).
#
# Env: MODES="none decay orth"  ALPHA=0.5  LAYERS=12,13,14  TRIALS=2  DECAY_P0=8000
set -uo pipefail
cd /home/woori/workspace_common/boltzmann-attention-pi
PHASE2="reports/facet_rft_2026/phase2_steering"
mkdir -p "$PHASE2"
LOG="$PHASE2/_phase2c_gpu${1:-0}_$(date +%Y%m%d_%H%M%S).log"
exec >>"$LOG" 2>&1
echo "[2c] start $(date)"

VLLM_PY=/home/woori/venvs/tau2_vllm_env/bin/python
TAU2_PY=/home/woori/venvs/seka_env/bin/python
SERVER_PY=_steering_vllm_server_gated.py
RUNNER=scripts/phase1_runner.py
TASK_IDS="$PHASE2/stratified_task_ids.json"
VEC="$PHASE2/steering_vectors_qwen7b.pt"

# GPU index is a POSITIONAL arg ($1); ports are DERIVED from it.
# (Do NOT read $PORT from env — PORT is a common env var and collides, e.g. a
#  node app on :8100 leaked PORT=8100 into the steering server, 2026-05-29.)
GPU=${1:-0}
PORT=$((9000 + GPU)); MPORT=$((8010 + GPU))
MODES=(${MODES:-none decay orth})
ALPHA=${ALPHA:-0.5}
LAYERS=${LAYERS:-12,13,14}
DECAY_P0=${DECAY_P0:-8000}
TRIALS=${TRIALS:-2}
SERVED=Qwen2.5-7B-Instruct; HF=Qwen/Qwen2.5-7B-Instruct

if [ -f /home/woori/.openrouter_key ]; then set -a; . /home/woori/.openrouter_key; set +a; fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then echo "[2c] ERROR no OPENROUTER_API_KEY"; exit 1; fi
OR_KEY="$OPENROUTER_API_KEY"

for mode in "${MODES[@]}"; do
  ts=$(date +%Y%m%d_%H%M%S)
  out_dir="$PHASE2/vllm_grid_qwen7b_a${ALPHA}_gate-${mode}_${ts}"
  srv_log="${out_dir}_server.log"
  mkdir -p "$out_dir"
  echo "{\"tag\":\"qwen7b\",\"alpha\":$ALPHA,\"gate\":\"$mode\",\"layers\":[$LAYERS],\"decay_p0\":$DECAY_P0}" > "$out_dir/steering.json"
  echo "[2c] === gate=$mode start $(date) ==="

  # wait until port free (avoid stale-server race)
  for i in $(seq 1 60); do
    if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then echo "[2c] :$PORT bound, wait ($i)"; sleep 3; else break; fi
  done

  CUDA_VISIBLE_DEVICES=$GPU MASTER_PORT=$MPORT VLLM_PORT=$MPORT nohup "$VLLM_PY" "$SERVER_PY" \
    --steering-vectors "$VEC" --relation validates --alpha "$ALPHA" --layers "$LAYERS" \
    --gate-mode "$mode" --gate-decay-pos "$DECAY_P0" \
    -- \
    --model "$HF" --served-model-name "$SERVED" --port "$PORT" --host 127.0.0.1 \
    --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.85 \
    --tool-call-parser hermes --enable-auto-tool-choice \
    > "$srv_log" 2>&1 &
  SRV=$!
  echo "[2c] vllm pid=$SRV log=$srv_log"

  ready=0
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then ready=1; echo "[2c] :$PORT READY ${i}s"; break; fi
    sleep 2
  done
  if [ "$ready" -ne 1 ]; then echo "[2c] gate=$mode FAILED to be ready"; kill -KILL $SRV 2>/dev/null || true; continue; fi

  "$TAU2_PY" "$RUNNER" --variants B0 --task-split base \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --agent-llm "openai/${SERVED}" \
    --user-llm openai/openai/gpt-4o-mini --user-base-url https://openrouter.ai/api/v1 \
    --user-api-key "$OR_KEY" \
    --domain telecom --num-trials $TRIALS --max-steps 200 --max-concurrency 3 \
    --timeout 600 --auto-resume --task-ids-file "$TASK_IDS" --out-dir "$out_dir" \
    >> "$out_dir/run.log" 2>&1
  echo "[2c] gate=$mode sims_done rc=$? out=$out_dir"

  kill -TERM $SRV 2>/dev/null || true; sleep 5; kill -KILL $SRV 2>/dev/null || true; sleep 5
done
date > "$PHASE2/phase2c_done_gpu${GPU}.txt"
echo "[2c] DONE gpu=$GPU modes=${MODES[*]} $(date)"
