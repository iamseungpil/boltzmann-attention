#!/bin/bash
# Phase 2a α grid extension — α=0.4, 0.6 added for finer peak resolution.
#
# Watches the original orchestrator (per model). When it finishes,
# launches an extension orchestrator on the SAME GPU with the new alphas.
#
# Conditions: identical to original orchestrator (baseline-equivalent).
#
# Env (or auto-detect):
#   QWEN_ORCH_PID    — original Qwen orchestrator bash PID
#   HERMES_ORCH_PID  — original Hermes orchestrator bash PID
#
# Usage:
#   QWEN_ORCH_PID=2033348 HERMES_ORCH_PID=2034966 \
#     nohup bash _phase2a_alpha_extension.sh > _ext.nohup.log 2>&1 &
set -uo pipefail
cd /home/woori/workspace_common/boltzmann-attention-pi

PHASE2="reports/facet_rft_2026/phase2_steering"
mkdir -p "$PHASE2"
LOG="$PHASE2/_alpha_ext_$(date +%Y%m%d_%H%M%S).log"
exec >>"$LOG" 2>&1
echo "[alpha-ext] start $(date)"

EXT_ALPHAS="${EXT_ALPHAS:-0.4 0.6}"
QWEN_ORCH_PID="${QWEN_ORCH_PID:-}"
HERMES_ORCH_PID="${HERMES_ORCH_PID:-}"

# auto-detect orchestrator pids if not provided
if [ -z "$QWEN_ORCH_PID" ]; then
  QWEN_ORCH_PID=$(pgrep -of "_phase2a_vllm_orchestrator.sh" | head -n1)
fi
echo "[alpha-ext] watching QWEN_ORCH=$QWEN_ORCH_PID HERMES_ORCH=$HERMES_ORCH_PID ALPHAS=$EXT_ALPHAS"

launch_ext() {
  # $1: tag  $2: orch pid to watch
  local tag="$1" pid="$2"
  if [ -z "$pid" ] || ! ps -p "$pid" >/dev/null 2>&1; then
    echo "[ext:$tag] orch pid '$pid' not alive — launching immediately"
  else
    echo "[ext:$tag] waiting for orchestrator pid=$pid to finish"
    while ps -p "$pid" >/dev/null 2>&1; do sleep 60; done
    echo "[ext:$tag] orchestrator pid=$pid exited at $(date)"
  fi

  # safety: wait briefly for GPU/vLLM teardown
  sleep 15

  # check phase2a_vllm_done marker
  [ -f "$PHASE2/phase2a_vllm_done_${tag}.txt" ] && echo "[ext:$tag] done marker confirmed"

  # Launch extension orchestrator on this model only
  echo "[ext:$tag] launching extension α grid (ALPHAS=$EXT_ALPHAS, MODEL=$tag, TRIALS=4) at $(date)"
  ALPHAS="$EXT_ALPHAS" TRIALS=4 MODEL="$tag" \
    nohup bash _phase2a_vllm_orchestrator.sh \
    >> "$PHASE2/_orch_vllm_${tag}_ext.nohup.log" 2>&1 &
  local OPID=$!
  echo "$OPID" > "$PHASE2/_orch_vllm_${tag}_ext.pid"
  echo "[ext:$tag] extension orchestrator pid=$OPID"
}

# parallel watchers
launch_ext qwen7b  "$QWEN_ORCH_PID"  &
WQ=$!
launch_ext hermes3 "$HERMES_ORCH_PID" &
WH=$!

wait $WQ; WQRC=$?
wait $WH; WHRC=$?
echo "[alpha-ext] qwen ext launcher rc=$WQRC hermes ext launcher rc=$WHRC"

date > "$PHASE2/alpha_ext_done.txt"
echo "[alpha-ext] DONE $(date)"
