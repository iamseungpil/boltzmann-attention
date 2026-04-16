#!/bin/bash
# Full v0/v1/v2 comparison + K-side ablation after bug fixes
# Tests all combinations: K-only, Q-only, Q+K at varying alpha

source /home/woori/venvs/seka_env/bin/activate

MODEL="Qwen/Qwen2.5-7B-Instruct"
BONT="external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt"
DATASET="/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json"
OUTDIR="reports/kside_ablation_2026_04_17"
mkdir -p "$OUTDIR"

GPU=${1:-0}
N=${2:-50}

echo "=== v0/v1/v2 comparison + K-side ablation, GPU=$GPU, N=$N ==="

# ---- v1: K-only (alpha > 0, beta = 0) ----
for ALPHA in 0.01 0.03 0.05; do
    echo ""
    echo ">>> [v1 K-only] alpha_K=$ALPHA, beta=0, N=$N"
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/ocq/eval_subtask4_dynamic_qk_v2.py \
        --model "$MODEL" \
        --device cuda:0 \
        --b-ont "$BONT" \
        --dataset "$DATASET" \
        --max-samples "$N" \
        --alpha "$ALPHA" \
        --beta 0.0 \
        --eps-threshold 0.10 \
        --max-tools 5 \
        --max-new-tokens-per-tool 100 \
        --out "$OUTDIR/v1_Konly_aK${ALPHA}_N${N}.json"
done

# ---- v2: Q+K dynamic (alpha > 0, beta < 0) ----
for ALPHA in 0.01 0.03 0.05; do
    echo ""
    echo ">>> [v2 Q+K] alpha_K=$ALPHA, beta=-0.03, N=$N"
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/ocq/eval_subtask4_dynamic_qk_v2.py \
        --model "$MODEL" \
        --device cuda:0 \
        --b-ont "$BONT" \
        --dataset "$DATASET" \
        --max-samples "$N" \
        --alpha "$ALPHA" \
        --beta -0.03 \
        --eps-threshold 0.10 \
        --max-tools 5 \
        --max-new-tokens-per-tool 100 \
        --out "$OUTDIR/v2_QK_aK${ALPHA}_bQ-0.03_N${N}.json"
done

echo ""
echo "=== Ablation complete ==="
