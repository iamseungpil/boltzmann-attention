#!/bin/bash
# Chain runner for Next experiments 1, 2, 3
# Next-1: Full Fisher-avg Mahalanobis Lloyd PPL on Mistral
# Next-2: Outlier layer bit preservation on Mistral
# Next-3: Per-layer Lloyd breakdown on Qwen-7B
set -u

REPO=/home/woori/workspace_common/boltzmann-attention
OUT_DIR=$REPO/reports/axis2_theoretical_verification
mkdir -p "$OUT_DIR"

# shellcheck disable=SC1091
source /home/woori/workspace_common/CDP/poc/set.env

cd "$REPO" || exit 1

MASTER_LOG="$OUT_DIR/chain_next_master.log"
START=$(date +%s)

echo "=== CHAIN NEXT START: $(date) ===" | tee "$MASTER_LOG"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader)" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Next-1
echo "--- [1/3] Next-1: Full Fisher Mahalanobis Lloyd PPL (Mistral) ---" | tee -a "$MASTER_LOG"
T1=$(date +%s)
python3 scripts/exp_next_1_full_fisher_lloyd_ppl.py > "$OUT_DIR/exp_next1.log" 2>&1
RC1=$?
E1=$(($(date +%s) - T1))
echo "[1/3] Next-1 rc=$RC1 elapsed=${E1}s ($((E1/60))m)" | tee -a "$MASTER_LOG"
echo "  Last lines of exp_next1.log:" | tee -a "$MASTER_LOG"
tail -20 "$OUT_DIR/exp_next1.log" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Next-2
echo "--- [2/3] Next-2: Outlier Layer Bit Preservation (Mistral) ---" | tee -a "$MASTER_LOG"
T2=$(date +%s)
python3 scripts/exp_next_2_outlier_layer_preservation.py > "$OUT_DIR/exp_next2.log" 2>&1
RC2=$?
E2=$(($(date +%s) - T2))
echo "[2/3] Next-2 rc=$RC2 elapsed=${E2}s ($((E2/60))m)" | tee -a "$MASTER_LOG"
echo "  Last lines of exp_next2.log:" | tee -a "$MASTER_LOG"
tail -20 "$OUT_DIR/exp_next2.log" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Next-3
echo "--- [3/3] Next-3: Per-layer Lloyd breakdown (Qwen-7B) ---" | tee -a "$MASTER_LOG"
T3=$(date +%s)
python3 scripts/exp_next_3_qwen_per_layer_lloyd.py > "$OUT_DIR/exp_next3.log" 2>&1
RC3=$?
E3=$(($(date +%s) - T3))
echo "[3/3] Next-3 rc=$RC3 elapsed=${E3}s ($((E3/60))m)" | tee -a "$MASTER_LOG"
echo "  Last lines of exp_next3.log:" | tee -a "$MASTER_LOG"
tail -25 "$OUT_DIR/exp_next3.log" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

TOTAL=$(($(date +%s) - START))
echo "=== CHAIN NEXT END: $(date) ===" | tee -a "$MASTER_LOG"
echo "Total: ${TOTAL}s ($((TOTAL/60))m)" | tee -a "$MASTER_LOG"
echo "Exit codes: next1=$RC1 next2=$RC2 next3=$RC3" | tee -a "$MASTER_LOG"
