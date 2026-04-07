#!/bin/bash
# Chain runner for Next experiments 4, 5
# Next-4: Qwen outlier preservation
# Next-5: Sensitivity-based bit allocation (Mistral + Qwen)
set -u

REPO=/home/woori/workspace_common/boltzmann-attention
OUT_DIR=$REPO/reports/axis2_theoretical_verification
mkdir -p "$OUT_DIR"

source /home/woori/workspace_common/CDP/poc/set.env
cd "$REPO" || exit 1

MASTER_LOG="$OUT_DIR/chain_next45_master.log"
START=$(date +%s)

echo "=== CHAIN 4/5 START: $(date) ===" | tee "$MASTER_LOG"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader)" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Next-4: Qwen outlier preservation
echo "--- [1/2] Next-4: Qwen Outlier Preservation ---" | tee -a "$MASTER_LOG"
T1=$(date +%s)
python3 scripts/exp_next_4_qwen_outlier_preservation.py > "$OUT_DIR/exp_next4.log" 2>&1
RC1=$?
E1=$(($(date +%s) - T1))
echo "[1/2] Next-4 rc=$RC1 elapsed=${E1}s" | tee -a "$MASTER_LOG"
echo "  Last lines of exp_next4.log:" | tee -a "$MASTER_LOG"
tail -25 "$OUT_DIR/exp_next4.log" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Next-5: Sensitivity-based allocation
echo "--- [2/2] Next-5: Sensitivity-based Bit Allocation (Mistral + Qwen) ---" | tee -a "$MASTER_LOG"
T2=$(date +%s)
python3 scripts/exp_next_5_sensitivity_bit_allocation.py > "$OUT_DIR/exp_next5.log" 2>&1
RC2=$?
E2=$(($(date +%s) - T2))
echo "[2/2] Next-5 rc=$RC2 elapsed=${E2}s" | tee -a "$MASTER_LOG"
echo "  Last lines of exp_next5.log:" | tee -a "$MASTER_LOG"
tail -60 "$OUT_DIR/exp_next5.log" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

TOTAL=$(($(date +%s) - START))
echo "=== CHAIN 4/5 END: $(date) ===" | tee -a "$MASTER_LOG"
echo "Total: ${TOTAL}s ($((TOTAL/60))m)" | tee -a "$MASTER_LOG"
echo "Exit codes: next4=$RC1 next5=$RC2" | tee -a "$MASTER_LOG"
