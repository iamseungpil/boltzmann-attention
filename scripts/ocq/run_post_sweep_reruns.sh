#!/bin/bash
# Post-sweep reruns to (a) relock layer-adaptive under the paper §4-faithful
# `install_layer_adaptive_hooks` fix, and (b) fill the τ²-telecom β+0.05
# headline cell (N2 gap) that was never locked on develop.
#
# Prereqs: the main Llama sweep (run_llama_cross_model_sweep.sh) must have
# completed, and `install_layer_adaptive_hooks` in eval_metatool_subtask1.py
# must carry the 2026-04-17 fix that makes Q-coverage run on ALL layers.
#
# Cells produced:
#   - Llama-3.1-8B retail   ladapt (paper Table 2b)
#   - Llama-3.1-8B telecom  ladapt (paper Table 2b)
#   - Qwen2.5-7B retail     ladapt (paper Table 1 relock)
#   - Qwen2.5-7B telecom    ladapt + Q+ sweep (paper Table 1 relock + N2 fill)
#
# Cost: ~1.3 GPU-h on A100 (estimated from observed 8s/task on Llama, 6s/task on Qwen).
#
# Invocation (after main sweep finishes):
#   nohup bash scripts/ocq/run_post_sweep_reruns.sh > logs/post_sweep_master.log 2>&1 &

set -e
set -o pipefail

REPO=/home/v-seungplee/boltzmann-attention
cd "$REPO"

DEVICE="cuda:0"
OUT_BASE="reports/tau2_2026_04_18"
BONT_BASE="external/SEKA/seka_projections"
mkdir -p "$OUT_BASE" logs

echo "[$(date +%H:%M:%S)] ========== Post-sweep rerun (ladapt fix + N2) =========="

# ---- Qwen τ² B_ont build (missing on local main) -------------------------
for dom in retail telecom; do
    if [[ ! -s "$BONT_BASE/ontology-qwen25-7b-tau2-$dom/B_ont.pt" ]]; then
        echo "[$(date +%H:%M:%S)] Qwen $dom B_ont build"
        python scripts/ocq/build_qwen_metatool_b_ont.py \
            --model Qwen/Qwen2.5-7B-Instruct \
            --ontology-json "reports/axis2_theoretical_verification/tau2_${dom}_ontology.json" \
            --out "$BONT_BASE/ontology-qwen25-7b-tau2-$dom/B_ont.pt" \
            --target-layers all \
            --device "$DEVICE" \
            2>&1 | tee -a "logs/post_qwen_${dom}_bont.log"
    else
        echo "[Qwen $dom B_ont] already built"
    fi
done

# ---- Llama ladapt reruns -----------------------------------------------
for dom in retail telecom; do
    if [[ "$dom" == "retail" ]]; then N=114; else N=200; fi
    echo "[$(date +%H:%M:%S)] Llama $dom ladapt (N=$N, single method)"
    python scripts/ocq/eval_tau2_bench.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --device "$DEVICE" \
        --b-ont "$BONT_BASE/ontology-llama31-8b-tau2-$dom/B_ont.pt" \
        --domain "$dom" \
        --methods no_steer ocq_ladapt_k0.05_q-0.03 \
        --max-samples "$N" \
        --out "$OUT_BASE/llama31_${dom}_ladapt_paper4.json" \
        2>&1 | tee -a "logs/post_llama_${dom}_ladapt.log"
done

# ---- Qwen ladapt reruns (paper Table 1 relock) + Qwen telecom β+ sweep (N2) ----
echo "[$(date +%H:%M:%S)] Qwen retail ladapt (N=114, paper §4 fix)"
python scripts/ocq/eval_tau2_bench.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --device "$DEVICE" \
    --b-ont "$BONT_BASE/ontology-qwen25-7b-tau2-retail/B_ont.pt" \
    --domain retail \
    --methods no_steer ocq_ladapt_k0.05_q-0.03 \
    --max-samples 114 \
    --out "$OUT_BASE/qwen25_retail_ladapt_paper4.json" \
    2>&1 | tee -a logs/post_qwen_retail_ladapt.log

echo "[$(date +%H:%M:%S)] Qwen telecom ladapt + Q+ sweep (N=200, paper §4 fix + N2)"
python scripts/ocq/eval_tau2_bench.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --device "$DEVICE" \
    --b-ont "$BONT_BASE/ontology-qwen25-7b-tau2-telecom/B_ont.pt" \
    --domain telecom \
    --methods no_steer ocq_ladapt_k0.05_q-0.03 \
        ocq_qbias_b0.03 ocq_qbias_b0.05 ocq_qbias_b0.10 \
    --max-samples 200 \
    --out "$OUT_BASE/qwen25_telecom_ladapt_qpos_paper4.json" \
    2>&1 | tee -a logs/post_qwen_telecom_ladapt_qpos.log

echo "[$(date +%H:%M:%S)] ========== Post-sweep complete =========="
echo "Outputs:"
ls -la "$OUT_BASE"/*_paper4.json
