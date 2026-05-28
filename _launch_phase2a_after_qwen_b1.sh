#!/bin/bash
# Wait until Qwen+OR B1 completes (n_sims=456 in results.json), then:
#   1) SIGTERM Qwen wrapper (prevents B2/B3)
#   2) SIGTERM both vLLMs
#   3) launch Phase 2a watchdog in background
#
# Safe to nohup; logs to phase2_steering/_launch_phase2a.log.
set -uo pipefail
cd /home/woori/workspace_common/boltzmann-attention-pi

PHASE2="reports/facet_rft_2026/phase2_steering"
mkdir -p "$PHASE2"
LOG="$PHASE2/_launch_phase2a_$(date +%Y%m%d_%H%M%S).log"
exec >>"$LOG" 2>&1

echo "[launch-phase2a] start $(date)"

QWEN_WRAPPER_PID=3791112
QWEN_B1_JSON="reports/facet_rft_2026/phase1_baseline/base_n114_qwen_openrouter_mini/B1_telecom_base.json/results.json"

count_b1() {
  python3 -c "
import json, sys
try:
    d = json.load(open('$QWEN_B1_JSON'))
    sims = d.get('simulations', [])
    completed = sum(1 for s in sims if s.get('end_time'))
    print(completed)
except Exception as e:
    print(-1)
" 2>/dev/null
}

echo "[launch-phase2a] polling B1 completion (target 456)..."
while true; do
  if ! ps -p "$QWEN_WRAPPER_PID" >/dev/null 2>&1; then
    echo "[launch-phase2a] Qwen wrapper pid=$QWEN_WRAPPER_PID exited"
    break
  fi
  n=$(count_b1)
  echo "[launch-phase2a] B1 completed=$n / 456 @ $(date '+%H:%M:%S')"
  if [ "$n" -ge 456 ]; then
    echo "[launch-phase2a] B1 complete — stopping Qwen wrapper before B2 starts"
    # SIGTERM wrapper + child python
    CHILD=$(pgrep -P "$QWEN_WRAPPER_PID" -f phase1_runner 2>/dev/null || true)
    echo "[launch-phase2a] kill -TERM wrapper=$QWEN_WRAPPER_PID child=$CHILD"
    kill -TERM "$QWEN_WRAPPER_PID" $CHILD 2>/dev/null || true
    sleep 10
    # if still alive, SIGKILL
    kill -KILL "$QWEN_WRAPPER_PID" $CHILD 2>/dev/null || true
    break
  fi
  sleep 30
done

# Kill vLLMs to free GPUs
echo "[launch-phase2a] stopping vLLMs"
for q in 'Hermes-3' 'Qwen2.5-7B'; do
  pids=$(pgrep -f "vllm.entrypoints.openai.api_server.*$q" || true)
  if [ -n "$pids" ]; then
    echo "[launch-phase2a] kill vLLM $q pids=$pids"
    kill -TERM $pids 2>/dev/null || true
  fi
done
sleep 15
# force-kill remaining vLLMs
for q in 'Hermes-3' 'Qwen2.5-7B'; do
  pids=$(pgrep -f "vllm.entrypoints.openai.api_server.*$q" || true)
  if [ -n "$pids" ]; then
    kill -KILL $pids 2>/dev/null || true
  fi
done

echo "[launch-phase2a] GPU state after vLLM shutdown:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

# Launch Phase 2a watchdog
# HERMES_PID=1 (init) is always alive → watchdog will hang on it. Use a fake-already-dead pid.
# Pick a recently-exited pid: 3785321 (Hermes wrapper, dead). Watchdog ps -p will return false → exits the wait quickly.
echo "[launch-phase2a] launching Phase 2a watchdog (HERMES already dead, QWEN about to die)"
HERMES_PID=3785321 QWEN_PID=$QWEN_WRAPPER_PID nohup bash _watchdog_phase2a_after_baseline.sh \
  >> "$PHASE2/_watchdog_phase2a.nohup.log" 2>&1 &
WD_PID=$!
echo "[launch-phase2a] watchdog launched pid=$WD_PID"
echo "$WD_PID" > "$PHASE2/_watchdog_phase2a.pid"

echo "[launch-phase2a] DONE $(date)"
