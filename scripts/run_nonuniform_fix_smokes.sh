#!/usr/bin/env bash
# Non-uniform K-bias fix smokes. Runs after V-bias smoke supervisor finishes.
# Tests: (Y1) α sweep, (Y3) projection-normalized, (Y4) contrastive-top-k.
set -u

REPO=/home/woori/workspace_common/boltzmann-attention
cd "$REPO"
source /home/woori/workspace_common/CDP/poc/set.env

MODEL="Qwen/Qwen2.5-7B-Instruct"
BONT="external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt"
LOG_DIR=logs/nonuniform_smoke
OUT_DIR=reports/nonuniform_smoke
mkdir -p "$LOG_DIR" "$OUT_DIR"

# Wait for V-bias supervisor 876676 to finish
WAIT_PID=876676
echo "[nonuniform] waiting for V-bias supervisor PID $WAIT_PID at $(date)" >> "$LOG_DIR/chain.log"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 120
done
echo "[nonuniform] V-bias done at $(date), launching non-uniform smokes" >> "$LOG_DIR/chain.log"

run_st4_smoke() {
  local TAG=$1; shift
  local METHODS=$@
  python scripts/ocq/eval_metatool_subtask4.py \
    --model "$MODEL" --device cuda:1 \
    --methods $METHODS \
    --b-ont "$BONT" \
    --max-samples 20 \
    --out "$OUT_DIR/st4_${TAG}.json" \
    > "$LOG_DIR/st4_${TAG}.log" 2>&1
}

# --- Y1: α sweep of standard K-bias (no Thm 6.9.5 fix, just α smaller) ---
echo "[nonuniform] Y1 α sweep at $(date)" >> "$LOG_DIR/chain.log"
run_st4_smoke "alpha_sweep_005" "no_steer" "ocq_bias_a0.05"
run_st4_smoke "alpha_sweep_010" "no_steer" "ocq_bias_a0.10"
run_st4_smoke "alpha_sweep_015" "no_steer" "ocq_bias_a0.15"
run_st4_smoke "alpha_sweep_020" "no_steer" "ocq_bias_a0.20"

# --- Y3: Projection-normalized K-bias (Thm 6.9.5) ---
echo "[nonuniform] Y3 normalized at $(date)" >> "$LOG_DIR/chain.log"
for A in 0.1 0.3 0.5 1.0; do
  run_st4_smoke "normalized_a${A}" "no_steer" "ocq_normbias_a${A}"
done

# --- Y4: Contrastive K-bias (top-k singular subtract) ---
echo "[nonuniform] Y4 contrastive at $(date)" >> "$LOG_DIR/chain.log"
for D in 1 2 3; do
  for A in 0.3 0.5; do
    run_st4_smoke "contrastive_a${A}_d${D}" "no_steer" "ocq_cbias_a${A}_d${D}"
  done
done

echo "[nonuniform] all done at $(date)" >> "$LOG_DIR/chain.log"

# Summary
python3 -c "
import json, glob
print('=== Non-uniform K-bias smoke summary ===')
for p in sorted(glob.glob('$OUT_DIR/*.json')):
    d = json.load(open(p))
    for r in d['results']:
        m = r['macro']
        print(f\"{p.split('/')[-1]:40s} {r['method']:30s} F1={m['F1']:.3f} rec={m['recall']:.3f} Exact={m['Exact']:.3f}\")
" >> "$LOG_DIR/summary.log" 2>&1
