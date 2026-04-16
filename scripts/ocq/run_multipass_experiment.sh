#!/bin/bash
# Multi-pass experiment: same question asked multiple times with P_emitted update
# Each pass is a FULL independent generation (no context breaking).
# After each pass, tools found are added to P_emitted, and Q is modified
# to subtract already-found facet directions on the next pass.
#
# This tests whether the P_emitted theory actually works when the model's
# generation context is not broken.

source /home/woori/venvs/seka_env/bin/activate

MODEL="Qwen/Qwen2.5-7B-Instruct"
BONT="external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt"
OUTDIR="reports/multipass_2026_04_17"
mkdir -p "$OUTDIR"

GPU=${1:-0}
N=${2:-50}

echo "=== Multi-pass P_emitted experiment, GPU=$GPU, N=$N ==="

# Multi-pass + k_early_only (best layer schedule)
echo ""
echo ">>> [multipass + k_early_only] alpha=0.05, beta=-0.05"
CUDA_VISIBLE_DEVICES=$GPU python3 scripts/ocq/eval_subtask4_dynamic_qk_v2.py \
    --model "$MODEL" \
    --device cuda:0 \
    --b-ont "$BONT" \
    --max-samples "$N" \
    --alpha 0.05 \
    --beta -0.05 \
    --layer-mode k_early_only \
    --max-tools 4 \
    --max-new-tokens-per-tool 80 \
    --out "$OUTDIR/multipass_k_early_a0.05_b-0.05_N${N}.json"

echo ""
echo "=== Multi-pass experiment complete ==="
