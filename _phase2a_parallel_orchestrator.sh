#!/bin/bash
# Phase 2a PARALLEL orchestrator — Qwen on GPU0, Hermes-3 on GPU1, fully concurrent.
#
# Phase A (extract, parallel):
#    Qwen-7B    on GPU0  -> steering_vectors_qwen7b.pt
#    Hermes-3   on GPU1  -> steering_vectors_hermes3.pt
#
# Phase B (server + α grid, parallel):
#    Qwen server   :8200 (GPU0) + α grid (validates × 6 alphas × N=20)
#    Hermes server :8201 (GPU1) + α grid (validates × 6 alphas × N=20)
#
# Both α grids hit OpenRouter user_sim concurrently. Total concurrency=6 (3+3).
#
# Skips already-done steps: if the .pt file exists, the extract is skipped.
#
# Outputs Phase 2a done marker once BOTH α grids finish.
set -uo pipefail
cd /home/woori/workspace_common/boltzmann-attention-pi

PHASE2="reports/facet_rft_2026/phase2_steering"
mkdir -p "$PHASE2"
LOG="$PHASE2/_orchestrator_parallel_$(date +%Y%m%d_%H%M%S).log"
exec >>"$LOG" 2>&1

echo "[orch-parallel] start $(date)"

# ============================================================
# Stratified task subset: 10 tasks/domain × 3 domains = 30 total.
# Use the SAME subset across all alphas + both models for fair
# within-subject comparison.
# ============================================================
TASK_IDS_FILE="$PHASE2/stratified_task_ids.json"
BASELINE_SRC="reports/facet_rft_2026/phase1_baseline/base_n114_qwen_openrouter_mini/B0_telecom_base.json/results.json"
if [ ! -f "$TASK_IDS_FILE" ]; then
  echo "[orch-parallel] generating stratified 30-task subset"
  python _phase2a_stratified_task_ids.py \
    --baseline-json "$BASELINE_SRC" \
    --per-domain 10 --seed 42 \
    --out "$TASK_IDS_FILE"
fi
echo "[orch-parallel] task subset: $TASK_IDS_FILE ($(python3 -c "import json; print(len(json.load(open('$TASK_IDS_FILE'))))") tasks)"

# load OpenRouter key
if [ -z "${OPENROUTER_KEY:-}" ]; then
  if [ -f /home/woori/.openrouter_key ]; then
    set -a; . /home/woori/.openrouter_key; set +a
    OPENROUTER_KEY="${OPENROUTER_API_KEY:-}"
    echo "[orch-parallel] loaded OPENROUTER_API_KEY from /home/woori/.openrouter_key"
  fi
fi
if [ -z "${OPENROUTER_KEY:-}" ]; then
  echo "[orch-parallel] ERROR OPENROUTER_KEY missing" >&2; exit 1
fi
export OPENROUTER_KEY

QWEN_VEC="$PHASE2/steering_vectors_qwen7b.pt"
HERMES_VEC="$PHASE2/steering_vectors_hermes3.pt"

# ============================================================
# Phase A — extracts in parallel
# ============================================================
QWEN_EXTRACT_PID=""
HERMES_EXTRACT_PID=""

# Qwen extract is probably already running from prior watchdog. Reuse if so.
if [ -f "$PHASE2/extract_qwen7b.pid" ]; then
  P=$(cat "$PHASE2/extract_qwen7b.pid")
  if ps -p "$P" >/dev/null 2>&1; then
    echo "[orch-parallel] Qwen extract already running pid=$P"
    QWEN_EXTRACT_PID=$P
  fi
fi
if [ -z "$QWEN_EXTRACT_PID" ] && [ ! -f "$QWEN_VEC" ]; then
  echo "[orch-parallel] launching Qwen-7B extract on cuda:0"
  PHASE2A_EXTRACT_DEVICE=cuda:0 bash _run_phase2a_extract_qwen7b.sh
  QWEN_EXTRACT_PID=$(cat "$PHASE2/extract_qwen7b.pid")
fi

if [ ! -f "$HERMES_VEC" ]; then
  if [ -f "$PHASE2/extract_hermes3.pid" ]; then
    P=$(cat "$PHASE2/extract_hermes3.pid")
    if ps -p "$P" >/dev/null 2>&1; then
      echo "[orch-parallel] Hermes-3 extract already running pid=$P"
      HERMES_EXTRACT_PID=$P
    fi
  fi
  if [ -z "$HERMES_EXTRACT_PID" ]; then
    echo "[orch-parallel] launching Hermes-3 extract on cuda:1"
    PHASE2A_EXTRACT_DEVICE=cuda:1 bash _run_phase2a_extract_hermes3.sh
    HERMES_EXTRACT_PID=$(cat "$PHASE2/extract_hermes3.pid")
  fi
fi

echo "[orch-parallel] waiting on extracts qwen=$QWEN_EXTRACT_PID hermes=$HERMES_EXTRACT_PID"
while true; do
  alive=0
  [ -n "$QWEN_EXTRACT_PID" ] && ps -p "$QWEN_EXTRACT_PID" >/dev/null 2>&1 && alive=$((alive+1))
  [ -n "$HERMES_EXTRACT_PID" ] && ps -p "$HERMES_EXTRACT_PID" >/dev/null 2>&1 && alive=$((alive+1))
  if [ "$alive" -eq 0 ]; then break; fi
  echo "[orch-parallel] $(date '+%H:%M:%S') extracts still alive ($alive)"
  sleep 30
done

if [ ! -f "$QWEN_VEC" ]; then echo "[orch-parallel] ERROR qwen vec missing"; exit 2; fi
if [ ! -f "$HERMES_VEC" ]; then echo "[orch-parallel] ERROR hermes vec missing"; exit 3; fi
echo "[orch-parallel] both extracts complete: $(ls -la $QWEN_VEC $HERMES_VEC)"

# ============================================================
# Phase B — servers + α grids in parallel
# ============================================================
echo "[orch-parallel] --- launching steering servers ---"
bash _run_phase2a_steerserv_qwen7b.sh   # GPU0:8200 — own health-wait
bash _run_phase2a_steerserv_hermes3.sh  # GPU1:8201 — own health-wait

echo "[orch-parallel] --- launching α grids in parallel ---"

QWEN_GRID_LOG="$PHASE2/alpha_grid_qwen7b_orch.log"
HERMES_GRID_LOG="$PHASE2/alpha_grid_hermes3_orch.log"

(
  python _phase2a_alpha_grid.py \
    --steer-url http://127.0.0.1:9200/v1 \
    --steer-base-model qwen7b-steer \
    --user-llm openrouter/openai/gpt-4o-mini \
    --user-base-url https://openrouter.ai/api/v1 \
    --user-api-key "$OPENROUTER_KEY" \
    --relation validates --layers 12,13,14 \
    --alphas 0.0,0.1,0.3,0.5,1.0,2.0 \
    --n 30 --trials 1 --max-steps 200 \
    --task-ids-file "$TASK_IDS_FILE" \
    --concurrency 3 --per-sim-timeout 600 \
    --tag qwen7b
) >> "$QWEN_GRID_LOG" 2>&1 &
QWEN_GRID_PID=$!
echo "[orch-parallel] qwen α-grid pid=$QWEN_GRID_PID log=$QWEN_GRID_LOG"

(
  python _phase2a_alpha_grid.py \
    --steer-url http://127.0.0.1:9201/v1 \
    --steer-base-model hermes3-steer \
    --user-llm openrouter/openai/gpt-4o-mini \
    --user-base-url https://openrouter.ai/api/v1 \
    --user-api-key "$OPENROUTER_KEY" \
    --relation validates --layers 14,15,16 \
    --alphas 0.0,0.1,0.3,0.5,1.0,2.0 \
    --n 30 --trials 1 --max-steps 200 \
    --task-ids-file "$TASK_IDS_FILE" \
    --concurrency 3 --per-sim-timeout 600 \
    --tag hermes3
) >> "$HERMES_GRID_LOG" 2>&1 &
HERMES_GRID_PID=$!
echo "[orch-parallel] hermes α-grid pid=$HERMES_GRID_PID log=$HERMES_GRID_LOG"

echo "[orch-parallel] waiting on both α grids"
wait $QWEN_GRID_PID
QRC=$?
wait $HERMES_GRID_PID
HRC=$?
echo "[orch-parallel] qwen α-grid rc=$QRC hermes α-grid rc=$HRC"

# stop servers
[ -f "$PHASE2/steerserv_qwen7b.pid" ] && kill "$(cat $PHASE2/steerserv_qwen7b.pid)" 2>/dev/null || true
[ -f "$PHASE2/steerserv_hermes3.pid" ] && kill "$(cat $PHASE2/steerserv_hermes3.pid)" 2>/dev/null || true

date > "$PHASE2/phase2a_done.txt"
echo "[orch-parallel] DONE $(date)"
