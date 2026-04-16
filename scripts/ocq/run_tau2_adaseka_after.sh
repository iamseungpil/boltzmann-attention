#!/bin/bash
# Run AdaSEKA on tau2-bench after current experiments finish
# Execute: bash scripts/ocq/run_tau2_adaseka_after.sh <GPU> <DOMAIN>

source /home/woori/venvs/seka_env/bin/activate

GPU=${1:-0}
DOMAIN=${2:-retail}
SHORT=$(echo $DOMAIN | sed 's/banking_knowledge/banking/')

MODEL="Qwen/Qwen2.5-7B-Instruct"
BONT="external/SEKA/seka_projections/ontology-qwen25-7b-tau2-${SHORT}/B_ont.pt"
OUTDIR="reports/tau2_2026_04_17"
mkdir -p "$OUTDIR"

echo "=== AdaSEKA on tau2-${DOMAIN}, GPU=$GPU ==="

# Sweep M (expert count) x alpha
CUDA_VISIBLE_DEVICES=$GPU python3 scripts/ocq/eval_tau2_bench.py \
    --model "$MODEL" \
    --device cuda:0 \
    --b-ont "$BONT" \
    --domain "$DOMAIN" \
    --methods no_steer adaseka_M2_a0.3_T0.5 adaseka_M3_a0.3_T0.5 adaseka_M3_a0.5_T0.5 adaseka_M4_a0.3_T0.5 \
    --max-samples 0 \
    --max-new-tokens 300 \
    --out "$OUTDIR/${SHORT}_adaseka.json"

echo "=== AdaSEKA ${DOMAIN} complete ==="
