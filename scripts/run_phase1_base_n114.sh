#!/bin/bash
# Phase 1 Baseline v2 — N=114 (telecom base split), trials=4, max_steps=200 (official tau3 protocol)
# v2 changes vs v1:
#   - vLLM on port 9000 (GPU1, max-model-len 32768)
#   - All three baselines (B0/B1/B2) re-run from scratch for cross-baseline consistency
#   - v1 partial (16K, B0 only valid) preserved at base_n114_v1_16k_partial/

LOG_DIR=/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/phase1_baseline/base_n114
PYBIN=/home/woori/venvs/seka_env/bin/python
RUNNER=/home/woori/workspace_common/boltzmann-attention-pi/scripts/phase1_runner.py
BASE_URL=http://127.0.0.1:9000/v1

mkdir -p "$LOG_DIR"
date '+started=%Y-%m-%d %H:%M:%S %Z' > "$LOG_DIR/status.txt"
echo "pid=$$" >> "$LOG_DIR/status.txt"
echo "host=$(hostname)" >> "$LOG_DIR/status.txt"
echo "vllm: GPU1 port 9000 max-model-len 32768" >> "$LOG_DIR/status.txt"
echo "args: --task-split base --num-trials 4 --max-steps 200 --max-concurrency 8 --base-url $BASE_URL" >> "$LOG_DIR/status.txt"

# Pre-flight: confirm vLLM is up
if ! curl -sf "$BASE_URL/models" >/dev/null 2>&1; then
  echo "ERROR: vLLM not reachable at :9000" >> "$LOG_DIR/status.txt"
  touch "$LOG_DIR/FAILED"
  exit 1
fi
echo "preflight=ok" >> "$LOG_DIR/status.txt"

# Run baselines (B0 -> B1 -> B2 sequentially, runner handles it)
"$PYBIN" "$RUNNER" \
  --task-split base \
  --num-trials 4 \
  --max-steps 200 \
  --max-concurrency 8 \
  --base-url "$BASE_URL" \
  --out-dir "$LOG_DIR" \
  >> "$LOG_DIR/run.log" 2>&1
EXIT=$?

date '+ended=%Y-%m-%d %H:%M:%S %Z' >> "$LOG_DIR/status.txt"
echo "exit_code=$EXIT" >> "$LOG_DIR/status.txt"
if [ $EXIT -eq 0 ]; then
  touch "$LOG_DIR/DONE"
else
  touch "$LOG_DIR/FAILED"
fi
exit $EXIT
