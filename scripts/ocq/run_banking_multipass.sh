#!/bin/bash
# Banking recovery experiment: multipass + smaller params + narrower K
# Run after current retail/telecom experiments finish

source /home/woori/venvs/seka_env/bin/activate

GPU=${1:-0}
N=${2:-0}  # 0 = all 97 tasks

MODEL="Qwen/Qwen2.5-7B-Instruct"
BONT="external/SEKA/seka_projections/ontology-qwen25-7b-tau2-banking/B_ont.pt"
OUTDIR="reports/tau2_2026_04_17"
mkdir -p "$OUTDIR"

echo "=== Banking recovery experiments, GPU=$GPU, N=$N ==="

# All recovery methods in one run for efficiency
CUDA_VISIBLE_DEVICES=$GPU python3 scripts/ocq/eval_tau2_bench.py \
    --model "$MODEL" \
    --device cuda:0 \
    --b-ont "$BONT" \
    --domain banking_knowledge \
    --methods no_steer \
              multipass_ocq_ladapt_k0.05_q-0.03 \
              multipass_ocq_qbias_b-0.03 \
              ocq_ladapt_k0.01_q-0.01 \
              ocq_ladapt_k0.05_q-0.03_f0.15 \
    --max-samples "$N" \
    --max-new-tokens 300 \
    --multipass-n 3 \
    --out "$OUTDIR/banking_recovery.json"

echo "=== Banking recovery complete ==="
