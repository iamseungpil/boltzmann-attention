#!/bin/bash
# Watcher that kicks off run_post_sweep_reruns.sh when the main Llama
# cross-model sweep completes. Polls logs/llama_sweep_master.log for the
# final completion marker every 60s. Survives login shell exits because
# it's started with nohup.
#
# Invocation:
#   nohup bash scripts/ocq/chain_post_sweep.sh > logs/chain_watcher.log 2>&1 &

set -u
REPO=/home/v-seungplee/boltzmann-attention
cd "$REPO"

MASTER_LOG="$REPO/logs/llama_sweep_master.log"
POST_SCRIPT="$REPO/scripts/ocq/run_post_sweep_reruns.sh"
POST_LOG="$REPO/logs/post_sweep_master.log"
COMPLETE_MARKER="L-strand sweep complete"

echo "[chain] $(date +%H:%M:%S) watcher armed, polling $MASTER_LOG every 60s"

# Wait until the master log contains the completion marker, or the main
# sweep process disappears (whichever first).
while true; do
    if grep -q "$COMPLETE_MARKER" "$MASTER_LOG" 2>/dev/null; then
        echo "[chain] $(date +%H:%M:%S) master sweep completion marker found"
        break
    fi
    # Safety: if no eval_tau2_bench python process is running anymore AND
    # master log is older than 10 minutes, the sweep likely died. Abort
    # rather than loop forever.
    if ! pgrep -f "eval_tau2_bench.py" > /dev/null; then
        AGE_MIN=$(( ( $(date +%s) - $(stat -c %Y "$MASTER_LOG") ) / 60 ))
        if [ "$AGE_MIN" -gt 10 ]; then
            echo "[chain] $(date +%H:%M:%S) ABORT — eval process dead, log idle >10min"
            exit 2
        fi
    fi
    sleep 60
done

# Small grace period so any last-method JSON writes settle to disk.
sleep 30

echo "[chain] $(date +%H:%M:%S) launching post-sweep reruns"
nohup bash "$POST_SCRIPT" > "$POST_LOG" 2>&1 &
POST_PID=$!
echo "[chain] $(date +%H:%M:%S) post-sweep PID=$POST_PID, log=$POST_LOG"

# Wait for post-sweep to finish, then emit a final summary.
wait "$POST_PID"
RC=$?
echo "[chain] $(date +%H:%M:%S) post-sweep exited rc=$RC"
if [ "$RC" -eq 0 ]; then
    echo "[chain] SUCCESS — all cells ready for paper update"
else
    echo "[chain] FAIL — inspect $POST_LOG"
fi
