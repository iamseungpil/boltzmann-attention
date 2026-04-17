#!/bin/bash
# Llama-3.1-8B-Instruct cross-model τ² sweep runner.
# Observed baseline: 8s/task on A100 bf16 eager attn.
# Total budget: L1b (5m) + L2 (2.0h) + L3 (3.6h) + L4 (1.7h) = 7.4h worst case.
#
# Order: L1b (telecom B_ont) → L2 retail N=114 (2h, fast signal)
#      → L3 telecom N=200 (3.6h) → L4 MetaTool ST4 (1.7h, optional).
# L1a (retail B_ont) and P1 preflight already passed (F1=0.78).
#
# Invocation:
#   nohup bash scripts/ocq/run_llama_cross_model_sweep.sh > logs/llama_sweep_master.log 2>&1 &

set -e
set -o pipefail

REPO=/home/v-seungplee/boltzmann-attention
cd "$REPO"

DEVICE="cuda:0"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUT_BASE="reports/tau2_2026_04_18"
BONT_BASE="external/SEKA/seka_projections"
mkdir -p "$OUT_BASE" logs reports/llama_cross_model_2026_04_18

echo "[$(date +%H:%M:%S)] ========== L1b: Llama telecom B_ont build =========="
if [[ ! -s "$BONT_BASE/ontology-llama31-8b-tau2-telecom/B_ont.pt" ]]; then
    python scripts/ocq/build_qwen_metatool_b_ont.py \
        --model "$MODEL" \
        --ontology-json reports/axis2_theoretical_verification/tau2_telecom_ontology.json \
        --out "$BONT_BASE/ontology-llama31-8b-tau2-telecom/B_ont.pt" \
        --target-layers all \
        --device "$DEVICE" \
        2>&1 | tee -a logs/l1b_llama_telecom_bont.log
    echo "[L1b] build done"
else
    echo "[L1b] already built, skipping"
fi

echo "[$(date +%H:%M:%S)] ========== L2: Llama retail N=114 Q-sweep =========="
python scripts/ocq/eval_tau2_bench.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --b-ont "$BONT_BASE/ontology-llama31-8b-tau2-retail/B_ont.pt" \
    --domain retail \
    --methods no_steer \
        ocq_qbias_b-0.10 ocq_qbias_b-0.05 ocq_qbias_b-0.03 \
        ocq_qbias_b0.01 ocq_qbias_b0.03 ocq_qbias_b0.05 ocq_qbias_b0.10 \
        ocq_ladapt_k0.05_q-0.03 \
    --max-samples 114 \
    --out "$OUT_BASE/llama31_retail_N114.json" \
    2>&1 | tee -a logs/l2_llama_retail_sweep.log

echo "[$(date +%H:%M:%S)] ========== L3: Llama telecom N=200 Q-sweep =========="
python scripts/ocq/eval_tau2_bench.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --b-ont "$BONT_BASE/ontology-llama31-8b-tau2-telecom/B_ont.pt" \
    --domain telecom \
    --methods no_steer \
        ocq_qbias_b-0.10 ocq_qbias_b-0.05 ocq_qbias_b-0.03 \
        ocq_qbias_b0.01 ocq_qbias_b0.03 ocq_qbias_b0.05 ocq_qbias_b0.10 \
        ocq_ladapt_k0.05_q-0.03 \
    --max-samples 200 \
    --out "$OUT_BASE/llama31_telecom_N200.json" \
    2>&1 | tee -a logs/l3_llama_telecom_sweep.log

echo "[$(date +%H:%M:%S)] ========== L-strand sweep complete (retail + telecom) =========="
echo "Outputs:"
ls -la "$OUT_BASE"/llama31_*.json
echo ""
echo "L4 (MetaTool ST4 ladapt) is NOT launched by this script — it needs a separate"
echo "MetaTool Llama B_ont build (r~24) and is dropped from critical path."
