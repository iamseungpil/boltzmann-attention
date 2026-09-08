#!/usr/bin/env bash
# LB lane - the B arm of the paired A/B (A = base, --gate 0, already persisted as bank_x806_base_nt4_*).
#   tree   = /home/woori/scratch/repo_lb (branch lb)      engine = PORT (default 8141, in-house GPU0)
#   queue  = /home/woori/scratch/x768/q_lb.txt (one task id per line; flock-popped so lanes can share)
#   out    = /home/woori/scratch/x768/out_lb/<TAG>.results.json.gz + lb sidecar
# Guards: served model id must match; a task finishing in <120s is a harness failure -> requeue, exit 3.
set -u
PORT="${1:-8141}"; CONC="${2:-1}"; NT="${3:-4}"; PREFIX="${4:-lb}"
QUEUE="${LB_QUEUE:-/home/woori/scratch/x768/q_lb.txt}"; LOCK="$QUEUE.lock"
R=/home/woori/scratch/repo_lb; LB="$R/scripts/distill/lb"; GO=/home/woori/iso_tau3/tau2-bench
OUT=/home/woori/scratch/x768/out_lb; LOG=/home/woori/scratch/logs; mkdir -p "$OUT" "$LOG"
PY=/home/woori/venvs/seka_env/bin/python
M="Qwen/Qwen3.8-27B-FP8"
GOT=$(curl -s -m 20 "http://localhost:$PORT/v1/models" | grep -oE '"id":"[^"]+"' | head -1 | cut -d'"' -f4)
[ "$GOT" = "$M" ] || { echo "[lb] refusing - port $PORT serves '$GOT'"; exit 1; }
source /home/woori/.openrouter_key; [ -f /home/woori/.openai_key ] && source /home/woori/.openai_key
export LB_DOCS_DIR="$GO/data/tau2/domains/banking_knowledge/documents"
export PYTHONPATH="src:$LB"
echo "[lb $(date '+%m-%d %H:%M')] start port=$PORT model=$GOT conc=$CONC nt=$NT prefix=$PREFIX queue=$(wc -l < "$QUEUE") sha=$(git -C "$R" rev-parse --short HEAD)"
pop(){ local t; exec 9>"$LOCK"; flock 9; t=$(head -1 "$QUEUE" 2>/dev/null); [ -n "$t" ] && sed -i '1d' "$QUEUE"; flock -u 9; exec 9>&-; echo "$t"; }
cd "$GO" || exit 1
while true; do
  T=$(pop); [ -z "$T" ] && { echo "[lb] queue empty"; break; }
  TAG="${PREFIX}_$T"; T0=$(date +%s)
  echo "[lb $(date '+%m-%d %H:%M')] -> $T (left $(wc -l < "$QUEUE"))"
  rm -rf "$GO/data/simulations/$TAG"; rm -f "$LOG/fb_${TAG}.jsonl"     # a rerun starts its own sidecar
  LB_SIDECAR="$LOG/fb_${TAG}.jsonl" $PY -u "$LB/lb_run.py" --domain banking_knowledge --retrieval_config alltools \
    --agent_model "$M" --agent_base "http://localhost:$PORT/v1" \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 --user_reasoning_effort low \
    --task_ids "$T" --num_trials "$NT" --max_concurrency "$CONC" --max_steps 200 --max_retries 8 --retry_delay 20 \
    --save_to "$TAG" > "$LOG/${TAG}_drv.log" 2>&1 || echo "  [lb] FAIL $T"
  EL=$(( $(date +%s) - T0 ))
  if [ "$EL" -lt 120 ]; then
    exec 9>"$LOCK"; flock 9; { echo "$T"; cat "$QUEUE"; } > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"; flock -u 9; exec 9>&-
    echo "  [lb] $T ended in ${EL}s = harness failure; requeued, stopping"; tail -5 "$LOG/${TAG}_drv.log"; exit 3
  fi
  d="$GO/data/simulations/$TAG"
  [ -f "$d/results.json" ] && gzip -c "$d/results.json" > "$OUT/$TAG.results.json.gz"
  [ -f "$LOG/${TAG}_drv.log" ] && gzip -c "$LOG/${TAG}_drv.log" > "$OUT/${TAG}_drv.log.gz"
  [ -f "$LOG/fb_${TAG}.jsonl" ] && gzip -c "$LOG/fb_${TAG}.jsonl" > "$OUT/fb_${TAG}.jsonl.gz"
  echo "  [lb] persisted $TAG (${EL}s)"
done
date; echo lb_DONE
