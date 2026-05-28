#!/bin/bash
# Per-model auto-chain: independent Qwen and Hermes detection.
#
# Watches each model's phase1_runner PID. When a PID exits:
#   1. Wait briefly for its vLLM to be torn down (~10s)
#   2. Verify final pass^1 on subset baseline
#   3. Launch THAT MODEL's α grid (MODEL=qwen7b or hermes3)
#
# This means Qwen α grid can start before Hermes sanity finishes.
#
# Required env (or auto-detect):
#   QWEN_PID    — Qwen phase1_runner PID  (default: detect via pgrep)
#   HERMES_PID  — Hermes phase1_runner PID
set -uo pipefail
cd /home/woori/workspace_common/boltzmann-attention-pi

PHASE2="reports/facet_rft_2026/phase2_steering"
mkdir -p "$PHASE2"
LOG="$PHASE2/_auto_chain_perm_$(date +%Y%m%d_%H%M%S).log"
exec >>"$LOG" 2>&1
echo "[auto-perm] start $(date)"

QWEN_PID="${QWEN_PID:-}"
HERMES_PID="${HERMES_PID:-}"

if [ -z "$QWEN_PID" ]; then
  QWEN_PID=$(pgrep -f 'phase1_runner.py.*--base-url http://127.0.0.1:9000/v1' | head -n1)
fi
if [ -z "$HERMES_PID" ]; then
  HERMES_PID=$(pgrep -f 'phase1_runner.py.*--base-url http://127.0.0.1:9001/v1' | head -n1)
fi

echo "[auto-perm] watching qwen_pid=$QWEN_PID hermes_pid=$HERMES_PID"

# ---- Per-model worker (background) ----
launch_after_sanity() {
  # $1: tag (qwen7b|hermes3)  $2: phase1_runner PID  $3: vllm port
  local tag="$1" pid="$2" port="$3"
  if [ -z "$pid" ]; then
    echo "[$tag] no live PID — skipping"; return 0
  fi

  echo "[$tag] watching phase1_runner pid=$pid"
  while ps -p "$pid" >/dev/null 2>&1; do sleep 60; done
  echo "[$tag] phase1_runner exited at $(date)"

  # wait for vLLM to be killed (sanity script does this in run_one)
  for i in $(seq 1 30); do
    if ! curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[$tag] vllm :$port stopped"; break
    fi
    sleep 2
  done

  # verify pass^1
  python3 << PYEOF
import json, os, glob, sys
P='/home/woori/workspace_common/boltzmann-attention/external/tau2-bench/data/simulations/reports/facet_rft_2026/phase2_steering/'
dirs=sorted(glob.glob(P+'sanity_equiv_${tag}_*_235432'))
if dirs:
    p=f'{dirs[-1]}/B0_telecom_base.json/results.json'
    if os.path.exists(p):
        d=json.load(open(p)); sims=d['simulations']
        done=[s for s in sims if s.get('end_time')]
        rs=[(s.get('reward_info') or {}).get('reward') for s in done]
        rs=[r for r in rs if r is not None]
        n_pass=sum(1 for r in rs if r>=1.0)
        p1=n_pass/max(len(rs),1)
        print(f'[auto-perm:${tag}] sanity final: n={len(rs)} pass={n_pass} pass^1={p1:.4f}')
        baseline={'qwen7b':0.1833, 'hermes3':0.1250}
        ci={'qwen7b':(0.124,0.262), 'hermes3':(0.077,0.196)}
        lo,hi=ci['${tag}']
        ok = lo <= p1 <= hi
        print(f'[auto-perm:${tag}] baseline={baseline[\"${tag}\"]:.4f} CI=[{lo},{hi}] in_CI={ok}')
PYEOF

  # Launch this model's α grid (sequential α=0.1, 0.3, 0.5, 1.0, 2.0)
  echo "[$tag] launching α grid (MODEL=${tag}, ALPHAS=0.1 0.3 0.5 1.0 2.0, TRIALS=4) at $(date)"
  ALPHAS="0.1 0.3 0.5 1.0 2.0" TRIALS=4 MODEL="${tag}" \
    nohup bash _phase2a_vllm_orchestrator.sh \
    >> "$PHASE2/_orch_vllm_${tag}.nohup.log" 2>&1 &
  local OPID=$!
  echo "$OPID" > "$PHASE2/_orch_vllm_${tag}.pid"
  echo "[$tag] α grid pid=$OPID — running in background"
}

# launch both watchers in parallel
launch_after_sanity qwen7b  "$QWEN_PID"  9000 &
WQ=$!
launch_after_sanity hermes3 "$HERMES_PID" 9001 &
WH=$!

wait $WQ; WQRC=$?
wait $WH; WHRC=$?
echo "[auto-perm] qwen watcher rc=$WQRC hermes watcher rc=$WHRC"

date > "$PHASE2/auto_perm_done.txt"
echo "[auto-perm] DONE $(date)"
