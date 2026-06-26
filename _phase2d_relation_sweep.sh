#!/bin/bash
# Phase 2d: causal verification of COMPLEMENTARY relations (AXIS-1).
# Inject different relation vectors at fixed alpha (gate=none) and measure
# behavioral change (esp. transfer_to_human_agents firing rate). Tests whether
# representation-space complementarity (error_fallback <-> retry_after_fail/validates)
# is CAUSAL for escalation behavior.
#
# GPU index = positional $1; ports derived (avoid PORT env collision).
# Env: RELATIONS="error_fallback"  ALPHA=0.5  LAYERS=12,13,14  TRIALS=2
set -uo pipefail
cd /home/woori/workspace_common/boltzmann-attention-pi
PHASE2="reports/facet_rft_2026/phase2_steering"
GPU=${1:-0}
PORT=$((9000 + GPU)); MPORT=$((8010 + GPU))
LOG="$PHASE2/_phase2d_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"
exec >>"$LOG" 2>&1
echo "[2d] start gpu=$GPU $(date)"

VLLM_PY=/home/woori/venvs/tau2_vllm_env/bin/python
TAU2_PY=/home/woori/venvs/seka_env/bin/python
SERVER_PY=_steering_vllm_server_gated.py
RUNNER=scripts/phase1_runner.py
TASK_IDS="$PHASE2/stratified_task_ids.json"
VEC="$PHASE2/steering_vectors_qwen7b.pt"

RELATIONS=(${RELATIONS:-validates})
ALPHA=${ALPHA:-0.5}
LAYERS=${LAYERS:-12,13,14}
TRIALS=${TRIALS:-2}
SERVED=Qwen2.5-7B-Instruct; HF=Qwen/Qwen2.5-7B-Instruct

if [ -f /home/woori/.openrouter_key ]; then set -a; . /home/woori/.openrouter_key; set +a; fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then echo "[2d] ERROR no key"; exit 1; fi
OR_KEY="$OPENROUTER_API_KEY"

for rel in "${RELATIONS[@]}"; do
  ts=$(date +%Y%m%d_%H%M%S)
  out_dir="$PHASE2/vllm_grid_qwen7b_a${ALPHA}_rel-${rel}_${ts}"
  srv_log="${out_dir}_server.log"
  mkdir -p "$out_dir"
  echo "{\"tag\":\"qwen7b\",\"alpha\":$ALPHA,\"relation\":\"$rel\",\"gate\":\"none\",\"layers\":[$LAYERS]}" > "$out_dir/steering.json"
  echo "[2d] === relation=$rel start $(date) ==="

  for i in $(seq 1 60); do
    if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then echo "[2d] :$PORT bound, wait ($i)"; sleep 3; else break; fi
  done

  CUDA_VISIBLE_DEVICES=$GPU MASTER_PORT=$MPORT VLLM_PORT=$MPORT nohup "$VLLM_PY" "$SERVER_PY" \
    --steering-vectors "$VEC" --relation "$rel" --alpha "$ALPHA" --layers "$LAYERS" \
    --gate-mode none \
    -- \
    --model "$HF" --served-model-name "$SERVED" --port "$PORT" --host 127.0.0.1 \
    --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.85 \
    --tool-call-parser hermes --enable-auto-tool-choice \
    > "$srv_log" 2>&1 &
  SRV=$!
  echo "[2d] vllm pid=$SRV log=$srv_log"

  ready=0
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then ready=1; echo "[2d] :$PORT READY ${i}s"; break; fi
    sleep 2
  done
  if [ "$ready" -ne 1 ]; then echo "[2d] relation=$rel FAILED ready"; kill -KILL $SRV 2>/dev/null || true; continue; fi

  "$TAU2_PY" "$RUNNER" --variants B0 --task-split base \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --agent-llm "openai/${SERVED}" \
    --user-llm openai/openai/gpt-4o-mini --user-base-url https://openrouter.ai/api/v1 \
    --user-api-key "$OR_KEY" \
    --domain telecom --num-trials $TRIALS --max-steps 200 --max-concurrency 3 \
    --timeout 600 --auto-resume --task-ids-file "$TASK_IDS" --out-dir "$out_dir" \
    >> "$out_dir/run.log" 2>&1
  echo "[2d] relation=$rel sims_done rc=$? out=$out_dir"

  kill -TERM $SRV 2>/dev/null || true; sleep 5; kill -KILL $SRV 2>/dev/null || true; sleep 5
done
date > "$PHASE2/phase2d_done_gpu${GPU}.txt"
echo "[2d] DONE gpu=$GPU rels=${RELATIONS[*]} $(date)"
