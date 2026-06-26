#!/bin/bash
# Step 1 (Facet-Aware Verifier-Guided RFT pivot, sprint plan §5.1):
# tau2 retail/telecom/airline x {qwen, llama} x 8 conditions baseline.
# Goal: stronger replication or refutation of E10 +9.7% MetaTool gap, and
# verification of training-data-mirroring hypothesis (pivot memo §2.2).
#
# Usage:
#   ./step1_sweep.sh <model_name> <model_id> <device>
# Example:
#   ./step1_sweep.sh qwen Qwen/Qwen2.5-7B-Instruct cuda:0
#   ./step1_sweep.sh llama NousResearch/Meta-Llama-3.1-8B-Instruct cuda:1
#
# N per domain caps at min(spec, available_in_tau2):
#   retail=114, telecom=256, airline=50

set -e
ROOT=/home/woori/workspace_common/boltzmann-attention-pi
cd "$ROOT"

MODEL_NAME="${1:?model_name (qwen or llama) required}"
MODEL_ID="${2:?model_id (HF path) required}"
DEVICE="${3:-cuda:0}"

PYBIN=/home/woori/venvs/seka_env/bin/python3.12
OUT=reports/facet_rft_2026/step1
LOG_DIR=reports/facet_rft_2026/logs
mkdir -p "$OUT" "$LOG_DIR"

CONDS=nl_full,nl_with_desc,facet_full,facet_compact,list_only,list_anon,facet_anon,noprompt

declare -A NMAX=( [retail]=114 [telecom]=256 [airline]=50 )

echo "[$(date)] === Step 1 sweep start: $MODEL_NAME on $DEVICE ==="
for domain in retail telecom airline; do
  N=${NMAX[$domain]}
  TAG="${MODEL_NAME}_${domain}_n${N}"
  OUT_JSON="$OUT/${TAG}.json"
  CELL_LOG="$LOG_DIR/${TAG}.log"
  if [ -f "$OUT_JSON" ]; then
    echo "[$(date)] SKIP $TAG (already exists)"
    continue
  fi
  echo "[$(date)] === ${TAG} ==="
  $PYBIN scripts/rank_replaceability/facet_eval.py \
    --model "$MODEL_ID" \
    --task "tau2_${domain}" \
    --schema "data/facet_schemas/tau2_${domain}.yaml" \
    --max-samples "$N" \
    --device "$DEVICE" \
    --conditions "$CONDS" \
    --max-new-tokens 192 \
    --out "$OUT_JSON" \
    > "$CELL_LOG" 2>&1
  tail -12 "$CELL_LOG"
  echo ""
done
echo "[$(date)] === Step 1 sweep done: $MODEL_NAME on $DEVICE ==="
