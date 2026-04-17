#!/bin/bash
# ToolBench scaling sweep: three subsets × four distractor counts × three methods.
# Assumes ontology-qwen25-7b-toolbench/B_ont.pt is already built.
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "${SCRIPT_DIR}/../.." && pwd)
source /home/woori/venvs/seka_env/bin/activate

MODEL="Qwen/Qwen2.5-7B-Instruct"
DEVICE="${DEVICE:-cuda:0}"
BONT="${REPO}/external/SEKA/seka_projections/ontology-qwen25-7b-toolbench/B_ont.pt"
OUTDIR="${REPO}/reports/toolbench_2026_04_17"
mkdir -p "${OUTDIR}"

SUBSET="${SUBSET:-G1_instruction}"   # default = 163 queries
N_SAMPLES="${N_SAMPLES:-0}"          # 0 = all
METHODS='no_steer ocq_qbias_b-0.03 ocq_qbias_b0.03 ocq_ladapt_k0.05_q-0.03 adaseka_M2_a0.05_T0.1 adaseka_M3_a0.1_T0.1'

for nd in 0 50 200 500; do
    out="${OUTDIR}/${SUBSET}_nd${nd}.json"
    if [[ -f "${out}" ]]; then
        echo "[skip] ${out} exists"
        continue
    fi
    echo "=== ${SUBSET} n_distractors=${nd} ==="
    CUDA_VISIBLE_DEVICES="${DEVICE#cuda:}" python3 "${SCRIPT_DIR}/eval_toolbench.py" \
        --model "${MODEL}" \
        --device cuda:0 \
        --b-ont "${BONT}" \
        --subsets "${SUBSET}" \
        --n-distractors "${nd}" \
        --methods ${METHODS} \
        --max-samples "${N_SAMPLES}" \
        --max-new-tokens 256 \
        --out "${out}"
done

echo "=== sweep complete ==="
