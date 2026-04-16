#!/bin/bash
# K-side ablation: test dynamic P_remaining K-bias after bug fixes
# Run after N=497 Q-only experiments complete
# Uses v2 (dynamic_qk) with iterative mode and varying alpha_K

source /home/woori/venvs/seka_env/bin/activate

MODEL="Qwen/Qwen2.5-7B-Instruct"
BONT="external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt"
DATASET="/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json"
OUTDIR="reports/kside_ablation_2026_04_16"
mkdir -p "$OUTDIR"

GPU=${1:-0}
N=${2:-50}

echo "=== K-side ablation on GPU $GPU, N=$N ==="

# Alpha sweep: 0.01, 0.03, 0.05 (very small, post-bugfix)
for ALPHA in 0.01 0.03 0.05; do
    echo ""
    echo ">>> alpha_K=$ALPHA, beta=-0.03, N=$N"
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
        --out "$OUTDIR/qwen_st4_aK${ALPHA}_bQ-0.03_N${N}.json"
done

echo ""
echo "=== K-side ablation complete ==="
