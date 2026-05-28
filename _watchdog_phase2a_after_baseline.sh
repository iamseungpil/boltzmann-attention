#!/bin/bash
# Phase 2a auto-trigger watchdog.
#
# Waits for both Phase-1 baseline bash wrappers (Hermes-3+OR PID=3785321,
# Qwen+OR PID=3791112) to exit AND for their B0..B3 trial JSONs to appear,
# then for each (qwen7b, hermes3) chains:
#   1) extract steering vectors
#   2) wait for extract to finish
#   3) launch steering server (with vectors)
#   4) run alpha grid smoke (N=20, trials=1)
#
# Output state under reports/facet_rft_2026/phase2_steering/.
#
# Env required:
#   OPENROUTER_KEY  (auto-loaded from /home/woori/_openrouter_key if missing)
#
# Optional overrides:
#   HERMES_PID, QWEN_PID  — set to skip auto-detect
#   N_TRIALS              — expected trial count, default 4
#
# Usage:
#   nohup bash _watchdog_phase2a_after_baseline.sh > _watchdog_phase2a.log 2>&1 &
set -uo pipefail
cd /home/woori/workspace_common/boltzmann-attention-pi

PHASE1="reports/facet_rft_2026/phase1_baseline"
PHASE2="reports/facet_rft_2026/phase2_steering"
mkdir -p "$PHASE2"
LOG_DIR="$PHASE2"
LOG="$LOG_DIR/watchdog_phase2a_$(date +%Y%m%d_%H%M%S).log"
exec >>"$LOG" 2>&1

echo "[wd-phase2a] start $(date)"

N_TRIALS="${N_TRIALS:-4}"
HERMES_PID="${HERMES_PID:-3785321}"
QWEN_PID="${QWEN_PID:-3791112}"

HERMES_OUT="$PHASE1/base_n114_hermes3_openrouter_mini"
QWEN_OUT="$PHASE1/base_n114_qwen_openrouter_mini"

if [ -z "${OPENROUTER_KEY:-}" ]; then
  if [ -f /home/woori/.openrouter_key ]; then
    # file uses `export OPENROUTER_API_KEY='...'` form
    set -a; . /home/woori/.openrouter_key; set +a
    OPENROUTER_KEY="${OPENROUTER_API_KEY:-}"
    echo "[wd-phase2a] loaded OPENROUTER_API_KEY from /home/woori/.openrouter_key"
  elif [ -f /home/woori/_openrouter_key ]; then
    OPENROUTER_KEY="$(cat /home/woori/_openrouter_key | tr -d '[:space:]')"
    echo "[wd-phase2a] loaded OPENROUTER_KEY from /home/woori/_openrouter_key"
  elif [ -f _openrouter_key ]; then
    OPENROUTER_KEY="$(cat _openrouter_key | tr -d '[:space:]')"
    echo "[wd-phase2a] loaded OPENROUTER_KEY from ./_openrouter_key"
  fi
  if [ -z "${OPENROUTER_KEY:-}" ]; then
    echo "[wd-phase2a] ERROR: OPENROUTER_KEY not set and key file absent" >&2
    exit 1
  fi
  export OPENROUTER_KEY
fi

trial_files_complete() {
  local out_dir="$1"
  local got=0
  for i in $(seq 0 $((N_TRIALS - 1))); do
    if [ -f "$out_dir/B${i}_telecom_base.json" ]; then
      got=$((got + 1))
    fi
  done
  if [ "$got" -ge "$N_TRIALS" ]; then return 0; else return 1; fi
}

wait_for_baseline() {
  # $1: label   $2: pid   $3: out_dir
  local label="$1" pid="$2" out_dir="$3"
  echo "[wd-phase2a] waiting for $label (pid=$pid out=$out_dir)"
  while true; do
    if ! ps -p "$pid" >/dev/null 2>&1; then
      echo "[wd-phase2a] $label pid=$pid exited"
      if trial_files_complete "$out_dir"; then
        echo "[wd-phase2a] $label trial files complete ($N_TRIALS/$N_TRIALS)"
        return 0
      else
        echo "[wd-phase2a] WARN $label pid gone but trials incomplete; proceeding anyway"
        return 0
      fi
    fi
    if trial_files_complete "$out_dir"; then
      echo "[wd-phase2a] $label all $N_TRIALS trial files present (pid=$pid may still cleaning up)"
      # wait for pid to clear too, up to 5min
      for _ in $(seq 1 30); do
        ps -p "$pid" >/dev/null 2>&1 || break
        sleep 10
      done
      return 0
    fi
    sleep 60
  done
}

wait_for_baseline "hermes3+OR" "$HERMES_PID" "$HERMES_OUT"
wait_for_baseline "qwen+OR"    "$QWEN_PID"   "$QWEN_OUT"

echo "[wd-phase2a] both baselines done — checking that vLLM servers can be shut down for GPU"
# Both Phase-1 vLLMs occupy GPU0 (Hermes-3 :9001) and GPU1 (Qwen :9000).
# Phase 2a steering server needs the same GPUs. Killing vLLMs frees them.
for label_pid_pair in "hermes3-vllm:$(pgrep -f 'vllm.entrypoints.openai.api_server.*Hermes-3' || true)" \
                       "qwen7b-vllm:$(pgrep -f 'vllm.entrypoints.openai.api_server.*Qwen2.5-7B' || true)"; do
  label="${label_pid_pair%%:*}"
  pids="${label_pid_pair#*:}"
  if [ -n "$pids" ]; then
    echo "[wd-phase2a] stopping $label pids=$pids"
    for p in $pids; do kill "$p" 2>/dev/null || true; done
  fi
done
sleep 10
nvidia-smi | head -20 || true

# ---- Qwen-7B chain (GPU0) ----
echo "[wd-phase2a] --- Qwen-7B extract ---"
bash _run_phase2a_extract_qwen7b.sh
EXTRACT_PID=$(cat "$PHASE2/extract_qwen7b.pid")
while ps -p "$EXTRACT_PID" >/dev/null 2>&1; do sleep 30; done
if [ ! -f "$PHASE2/steering_vectors_qwen7b.pt" ]; then
  echo "[wd-phase2a] ERROR qwen7b extract failed" >&2
  exit 2
fi
echo "[wd-phase2a] qwen7b extract done — launching steering server"
bash _run_phase2a_steerserv_qwen7b.sh

echo "[wd-phase2a] alpha-grid qwen7b"
python _phase2a_alpha_grid.py \
  --steer-url http://127.0.0.1:8200/v1 \
  --steer-base-model qwen7b-steer \
  --user-llm openrouter/openai/gpt-4o-mini \
  --user-base-url https://openrouter.ai/api/v1 \
  --user-api-key "$OPENROUTER_KEY" \
  --relation validates --layers 12,13,14 \
  --alphas 0.0,0.1,0.3,0.5,1.0,2.0 \
  --n 20 --trials 1 --max-steps 200 \
  --concurrency 3 --per-sim-timeout 600 \
  --tag qwen7b

if [ -f "$PHASE2/steerserv_qwen7b.pid" ]; then
  kill "$(cat $PHASE2/steerserv_qwen7b.pid)" 2>/dev/null || true
  sleep 5
fi

# ---- Hermes-3 chain (GPU1) ----
echo "[wd-phase2a] --- Hermes-3 extract ---"
bash _run_phase2a_extract_hermes3.sh
EXTRACT_PID=$(cat "$PHASE2/extract_hermes3.pid")
while ps -p "$EXTRACT_PID" >/dev/null 2>&1; do sleep 30; done
if [ ! -f "$PHASE2/steering_vectors_hermes3.pt" ]; then
  echo "[wd-phase2a] ERROR hermes3 extract failed" >&2
  exit 3
fi
echo "[wd-phase2a] hermes3 extract done — launching steering server"
bash _run_phase2a_steerserv_hermes3.sh

echo "[wd-phase2a] alpha-grid hermes3"
python _phase2a_alpha_grid.py \
  --steer-url http://127.0.0.1:8201/v1 \
  --steer-base-model hermes3-steer \
  --user-llm openrouter/openai/gpt-4o-mini \
  --user-base-url https://openrouter.ai/api/v1 \
  --user-api-key "$OPENROUTER_KEY" \
  --relation validates --layers 14,15,16 \
  --alphas 0.0,0.1,0.3,0.5,1.0,2.0 \
  --n 20 --trials 1 --max-steps 200 \
  --concurrency 3 --per-sim-timeout 600 \
  --tag hermes3

if [ -f "$PHASE2/steerserv_hermes3.pid" ]; then
  kill "$(cat $PHASE2/steerserv_hermes3.pid)" 2>/dev/null || true
fi

date > "$PHASE2/phase2a_done.txt"
echo "[wd-phase2a] DONE $(date)"
