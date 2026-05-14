#!/bin/bash
# E11 cross-domain transfer sweep -- 6 directional (source -> target) pairs
# for one model on one GPU. Usage:
#   ./e11_sweep.sh <model_name> <model_id> <device>
# Example:
#   ./e11_sweep.sh qwen Qwen/Qwen2.5-7B-Instruct cuda:0
#   ./e11_sweep.sh llama NousResearch/Meta-Llama-3.1-8B-Instruct cuda:1

set -e
cd "$(dirname "$0")/../.."   # placeholder; will be overridden by absolute path
ROOT=/home/woori/workspace_common/boltzmann-attention-pi
cd "$ROOT"

MODEL_NAME="${1:?model_name (qwen or llama) required}"
MODEL_ID="${2:?model_id (HF path) required}"
DEVICE="${3:-cuda:0}"

PYBIN=/home/woori/venvs/seka_env/bin/python3.12
OUT=reports/facet_ontology_2026_04/e11
LOG_DIR=reports/facet_ontology_2026_04/logs
mkdir -p "$OUT" "$LOG_DIR"

PAIRS=(
  "retail telecom"
  "retail airline"
  "telecom retail"
  "telecom airline"
  "airline retail"
  "airline telecom"
)

CONDS=nl_full,nl_full_source,facet_full,facet_xfer,facet_compact,noprompt

echo "[$(date)] === E11 sweep start: $MODEL_NAME on $DEVICE ==="
for pair in "${PAIRS[@]}"; do
  SRC=$(echo "$pair" | cut -d' ' -f1)
  TGT=$(echo "$pair" | cut -d' ' -f2)
  TAG="${MODEL_NAME}_${SRC}_to_${TGT}_n64"
  OUT_JSON="$OUT/${TAG}.json"
  CELL_LOG="$LOG_DIR/${TAG}.log"
  if [ -f "$OUT_JSON" ]; then
    echo "[$(date)] SKIP $TAG (already exists)"
    continue
  fi
  echo "[$(date)] === ${TAG} ==="
  $PYBIN scripts/rank_replaceability/facet_eval.py \
    --model "$MODEL_ID" \
    --task "tau2_${TGT}" \
    --schema "data/facet_schemas/tau2_${TGT}.yaml" \
    --source-schema "data/facet_schemas/tau2_${SRC}.yaml" \
    --max-samples 64 --device "$DEVICE" \
    --conditions "$CONDS" \
    --max-new-tokens 192 \
    --out "$OUT_JSON" \
    > "$CELL_LOG" 2>&1
  tail -10 "$CELL_LOG"
done
echo "[$(date)] === E11 sweep done: $MODEL_NAME on $DEVICE ==="
