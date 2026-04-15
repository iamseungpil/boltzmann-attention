#!/usr/bin/env bash
# Overnight Subtask4 smoke + full on Qwen-Instruct (GPU1 when free, else GPU0).
#
# Schedule:
#   1. Wait for Mistral mean a0.3 on GPU1 to finish (PID 703655)
#   2. Smoke N=20 × 6 configs (no_steer/a0.3 × real/random/featshuffle) on GPU1
#   3. Full 497 × 6 configs on GPU1
#
# Total: ~20-30 min smoke + ~3 h full = ~3.5 h wall-clock.
set -u

REPO=/home/woori/workspace_common/boltzmann-attention
cd "$REPO"
source /home/woori/workspace_common/CDP/poc/set.env

MODEL="Qwen/Qwen2.5-7B-Instruct"
LOG_DIR=logs/subtask4_overnight
OUT_DIR=reports/subtask4_overnight
mkdir -p "$LOG_DIR" "$OUT_DIR"

B_REAL="external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt"
B_RAND="external/SEKA/seka_projections/ontology-qwen25-7b-metatool-random/B_ont.pt"
B_FSHUF="external/SEKA/seka_projections/ontology-qwen25-7b-metatool-featshuffle/B_ont.pt"

# --- Wait for Mistral GPU1 run to finish ---
WAIT_PID=703655
echo "[overnight] waiting for Mistral PID $WAIT_PID at $(date)" >> "$LOG_DIR/chain.log"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done
echo "[overnight] GPU1 free at $(date)" >> "$LOG_DIR/chain.log"

run_st4() {
  local TAG=$1 BONT=$2 N=$3
  python scripts/ocq/eval_metatool_subtask4.py \
    --model "$MODEL" --device cuda:1 \
    --methods no_steer ocq_bias_a0.3 \
    --b-ont "$BONT" \
    --max-samples "$N" \
    --out "$OUT_DIR/st4_${TAG}_N${N}.json" \
    > "$LOG_DIR/st4_${TAG}_N${N}.log" 2>&1
}

# --- Phase 1: Smoke N=20 (6 configs) ---
echo "[overnight] Phase 1 smoke N=20 at $(date)" >> "$LOG_DIR/chain.log"
run_st4 "real" "$B_REAL" 20
run_st4 "random" "$B_RAND" 20
run_st4 "featshuffle" "$B_FSHUF" 20
echo "[overnight] Phase 1 smoke complete at $(date)" >> "$LOG_DIR/chain.log"

# --- Phase 2: Full 497 (3 configs × no_steer + a0.3 each) ---
echo "[overnight] Phase 2 full 497 at $(date)" >> "$LOG_DIR/chain.log"
run_st4 "real" "$B_REAL" 0       # 0 = all
run_st4 "random" "$B_RAND" 0
run_st4 "featshuffle" "$B_FSHUF" 0
echo "[overnight] Phase 2 complete at $(date)" >> "$LOG_DIR/chain.log"

# Summary
echo "[overnight] ALL DONE at $(date)" >> "$LOG_DIR/chain.log"
python3 -c "
import json, glob
rows = []
for p in sorted(glob.glob('$OUT_DIR/st4_*.json')):
    d = json.load(open(p))
    for r in d['results']:
        m = r['macro']
        rows.append((p.split('/')[-1], r['method'], d['n_queries'],
                     m['F1'], m['F_0.5'], m['EU'], m['Jaccard'], m['Exact']))
for name, meth, n, f1, f05, eu, jac, ex in rows:
    print(f'{name:40s} {meth:20s} n={n:4d} F1={f1:.3f} F0.5={f05:.3f} EU={eu:.3f} Jac={jac:.3f} Exact={ex:.3f}')
" >> "$LOG_DIR/summary.log" 2>&1
