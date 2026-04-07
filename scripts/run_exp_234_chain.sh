#!/bin/bash
# Chain runner for experiments 2, 3, 4
# Runs them sequentially, logs each to separate file
set -u

REPO=/home/woori/workspace_common/boltzmann-attention
OUT_DIR=$REPO/reports/axis2_theoretical_verification
mkdir -p "$OUT_DIR"

# shellcheck disable=SC1091
source /home/woori/workspace_common/CDP/poc/set.env

cd "$REPO" || exit 1

START=$(date +%s)
echo "=== CHAIN START: $(date) ===" | tee "$OUT_DIR/chain_master.log"

# Experiment 2: Spherical quantizer on Mistral
echo "" | tee -a "$OUT_DIR/chain_master.log"
echo "--- [1/3] Experiment 2: Spherical Quantizer ---" | tee -a "$OUT_DIR/chain_master.log"
T2=$(date +%s)
python3 scripts/exp_2_spherical_quantizer_mistral.py > "$OUT_DIR/exp2_spherical.log" 2>&1
RC2=$?
ELAPSED2=$(($(date +%s) - T2))
echo "[1/3] Exp2 rc=$RC2 elapsed=${ELAPSED2}s" | tee -a "$OUT_DIR/chain_master.log"

# Experiment 3: Per-token Fisher prototype
echo "" | tee -a "$OUT_DIR/chain_master.log"
echo "--- [2/3] Experiment 3: Per-token Fisher ---" | tee -a "$OUT_DIR/chain_master.log"
T3=$(date +%s)
python3 scripts/exp_3_per_token_fisher_prototype.py > "$OUT_DIR/exp3_fisher.log" 2>&1
RC3=$?
ELAPSED3=$(($(date +%s) - T3))
echo "[2/3] Exp3 rc=$RC3 elapsed=${ELAPSED3}s" | tee -a "$OUT_DIR/chain_master.log"

# Experiment 4: Per-layer Lloyd breakdown
echo "" | tee -a "$OUT_DIR/chain_master.log"
echo "--- [3/3] Experiment 4: Per-layer Lloyd Breakdown ---" | tee -a "$OUT_DIR/chain_master.log"
T4=$(date +%s)
python3 scripts/exp_4_per_layer_lloyd_breakdown.py > "$OUT_DIR/exp4_lloyd_breakdown.log" 2>&1
RC4=$?
ELAPSED4=$(($(date +%s) - T4))
echo "[3/3] Exp4 rc=$RC4 elapsed=${ELAPSED4}s" | tee -a "$OUT_DIR/chain_master.log"

TOTAL=$(($(date +%s) - START))
echo "" | tee -a "$OUT_DIR/chain_master.log"
echo "=== CHAIN END: $(date) ===" | tee -a "$OUT_DIR/chain_master.log"
echo "Total: ${TOTAL}s ($((TOTAL/60))m)" | tee -a "$OUT_DIR/chain_master.log"
echo "Exit codes: exp2=$RC2 exp3=$RC3 exp4=$RC4" | tee -a "$OUT_DIR/chain_master.log"
