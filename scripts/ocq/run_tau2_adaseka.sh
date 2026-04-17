#!/bin/bash
# AdaSEKA cross-domain sweep on τ²-bench (retail, airline, telecom).
# Completes the coverage gap identified 2026-04-17: MetaTool + Banking + CounterFact
# only. Runs three α-configurations per domain (safe / mid / known-fragile) to
# profile amplitude sensitivity, matching the style used on MetaTool / Banking.
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "${SCRIPT_DIR}/../.." && pwd)
source /home/woori/venvs/seka_env/bin/activate

MODEL="Qwen/Qwen2.5-7B-Instruct"
DEVICE="${DEVICE:-cuda:0}"
OUTDIR="${REPO}/reports/tau2_2026_04_17"
mkdir -p "${OUTDIR}"

# Shared method list (known-good on MetaTool: M2 a0.05 T0.1 → +3.68pp)
METHODS='no_steer adaseka_M2_a0.05_T0.1 adaseka_M3_a0.1_T0.1 adaseka_M3_a0.3_T0.5'
MAXTOK=300

run_domain() {
    local DOMAIN=$1
    local BONT=$2
    local NSAMP=$3
    local OUT="${OUTDIR}/${DOMAIN}_adaseka_sweep.json"
    if [[ -f "${OUT}" ]]; then
        echo "[skip] ${OUT} exists"
        return
    fi
    echo "=== τ² ${DOMAIN} AdaSEKA sweep (N=${NSAMP}) ==="
    CUDA_VISIBLE_DEVICES="${DEVICE#cuda:}" python3 "${SCRIPT_DIR}/eval_tau2_bench.py" \
        --model "${MODEL}" \
        --device cuda:0 \
        --b-ont "${BONT}" \
        --domain "${DOMAIN}" \
        --methods ${METHODS} \
        --max-samples "${NSAMP}" \
        --max-new-tokens "${MAXTOK}" \
        --out "${OUT}"
}

run_domain retail \
    "${REPO}/external/SEKA/seka_projections/ontology-qwen25-7b-tau2-retail/B_ont.pt" \
    114

run_domain airline \
    "${REPO}/external/SEKA/seka_projections/ontology-qwen25-7b-tau2-airline/B_ont.pt" \
    50

run_domain telecom \
    "${REPO}/external/SEKA/seka_projections/ontology-qwen25-7b-tau2-telecom/B_ont.pt" \
    200

echo "=== AdaSEKA cross-domain sweep complete ==="
