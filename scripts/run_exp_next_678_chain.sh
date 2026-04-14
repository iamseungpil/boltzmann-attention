#!/bin/bash
# Chain runner for Next experiments 6, 7, 8
# Next-6: Mistral-Nemo-Base-2407 full suite (third model point)
# Next-7: Fine-grained mixed-precision on Mistral-7B
# Next-8: Per-head sensitivity on Mistral-7B
set -u

REPO=/home/woori/workspace_common/boltzmann-attention
OUT_DIR=$REPO/reports/axis2_theoretical_verification
mkdir -p "$OUT_DIR"

source /home/woori/workspace_common/CDP/poc/set.env
cd "$REPO" || exit 1

MASTER_LOG="$OUT_DIR/chain_next678_master.log"
START=$(date +%s)

echo "=== CHAIN 6/7/8 START: $(date) ===" | tee "$MASTER_LOG"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader)" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Next-6: Mistral-Nemo full suite
echo "--- [1/3] Next-6: Mistral-Nemo-Base-2407 full suite (third model point) ---" | tee -a "$MASTER_LOG"
T1=$(date +%s)
python3 scripts/exp_next_6_mistral_nemo_full.py > "$OUT_DIR/exp_next6.log" 2>&1
RC1=$?
E1=$(($(date +%s) - T1))
echo "[1/3] Next-6 rc=$RC1 elapsed=${E1}s ($((E1/60))m)" | tee -a "$MASTER_LOG"
echo "  Last 30 lines of exp_next6.log:" | tee -a "$MASTER_LOG"
tail -30 "$OUT_DIR/exp_next6.log" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Next-7: Fine-grained mix on Mistral
echo "--- [2/3] Next-7: Fine-grained Mixed-Precision (Mistral-7B) ---" | tee -a "$MASTER_LOG"
T2=$(date +%s)
python3 scripts/exp_next_7_mistral_fine_grained_mix.py > "$OUT_DIR/exp_next7.log" 2>&1
RC2=$?
E2=$(($(date +%s) - T2))
echo "[2/3] Next-7 rc=$RC2 elapsed=${E2}s ($((E2/60))m)" | tee -a "$MASTER_LOG"
echo "  Last 40 lines of exp_next7.log:" | tee -a "$MASTER_LOG"
tail -40 "$OUT_DIR/exp_next7.log" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Next-8: Per-head sensitivity
echo "--- [3/3] Next-8: Per-Head Sensitivity (Mistral-7B) ---" | tee -a "$MASTER_LOG"
T3=$(date +%s)
python3 scripts/exp_next_8_per_head_sensitivity.py > "$OUT_DIR/exp_next8.log" 2>&1
RC3=$?
E3=$(($(date +%s) - T3))
echo "[3/3] Next-8 rc=$RC3 elapsed=${E3}s ($((E3/60))m)" | tee -a "$MASTER_LOG"
echo "  Last 40 lines of exp_next8.log:" | tee -a "$MASTER_LOG"
tail -40 "$OUT_DIR/exp_next8.log" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

TOTAL=$(($(date +%s) - START))
echo "=== CHAIN 6/7/8 END: $(date) ===" | tee -a "$MASTER_LOG"
echo "Total: ${TOTAL}s ($((TOTAL/60))m)" | tee -a "$MASTER_LOG"
echo "Exit codes: next6=$RC1 next7=$RC2 next8=$RC3" | tee -a "$MASTER_LOG"
