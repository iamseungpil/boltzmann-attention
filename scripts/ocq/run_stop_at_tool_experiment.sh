#!/bin/bash
# Stop-at-tool iterative experiment: true P_emitted subtraction
# Each generation step stops at </tool_call>, then hooks are updated
# before the next tool is generated.
#
# Combines with layer-adaptive K (early layers only) + Q (late layers only)
# This is the theoretically correct approach:
#   - P_emitted tracks only selected facets (not full B_ont)
#   - Each tool gets its own Q-coverage pass
#   - K helps precision in early layers without disrupting output convergence

source /home/woori/venvs/seka_env/bin/activate

MODEL="Qwen/Qwen2.5-7B-Instruct"
BONT="external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt"
OUTDIR="reports/stop_at_tool_2026_04_17"
mkdir -p "$OUTDIR"

GPU=${1:-0}
N=${2:-50}

echo "=== Stop-at-tool iterative experiment, GPU=$GPU, N=$N ==="

# 1. Stop-at-tool + k_early_only (best layer schedule)
echo ""
echo ">>> [stop_at_tool + k_early_only] alpha=0.05, beta=-0.05"
CUDA_VISIBLE_DEVICES=$GPU python3 scripts/ocq/eval_subtask4_dynamic_qk_v2.py \
    --model "$MODEL" \
    --device cuda:0 \
    --b-ont "$BONT" \
    --max-samples "$N" \
    --alpha 0.05 \
    --beta -0.05 \
    --layer-mode k_early_only \
    --stop-at-tool \
    --eps-threshold 0.10 \
    --max-tools 5 \
    --max-new-tokens-per-tool 80 \
    --out "$OUTDIR/stop_k_early_a0.05_b-0.05_N${N}.json"

# 2. Stop-at-tool + uniform (Q all layers, K all layers) for comparison
echo ""
echo ">>> [stop_at_tool + uniform] alpha=0.05, beta=-0.05"
CUDA_VISIBLE_DEVICES=$GPU python3 scripts/ocq/eval_subtask4_dynamic_qk_v2.py \
    --model "$MODEL" \
    --device cuda:0 \
    --b-ont "$BONT" \
    --max-samples "$N" \
    --alpha 0.05 \
    --beta -0.05 \
    --layer-mode uniform \
    --stop-at-tool \
    --eps-threshold 0.10 \
    --max-tools 5 \
    --max-new-tokens-per-tool 80 \
    --out "$OUTDIR/stop_uniform_a0.05_b-0.05_N${N}.json"

# 3. Stop-at-tool + Q-only (no K, beta=-0.05) — tests if true iterative Q works
echo ""
echo ">>> [stop_at_tool + Q-only] alpha=0, beta=-0.05"
CUDA_VISIBLE_DEVICES=$GPU python3 scripts/ocq/eval_subtask4_dynamic_qk_v2.py \
    --model "$MODEL" \
    --device cuda:0 \
    --b-ont "$BONT" \
    --max-samples "$N" \
    --alpha 0.0 \
    --beta -0.05 \
    --layer-mode uniform \
    --stop-at-tool \
    --eps-threshold 0.10 \
    --max-tools 5 \
    --max-new-tokens-per-tool 80 \
    --out "$OUTDIR/stop_qonly_b-0.05_N${N}.json"

echo ""
echo "=== Stop-at-tool experiment complete ==="
