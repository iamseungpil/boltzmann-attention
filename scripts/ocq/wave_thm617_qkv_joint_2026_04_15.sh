#!/usr/bin/env bash
# Thm 6.17 QKV-joint small-α full 497 wave
# Decisive cell: ocq_qkv_a0.05_v0.05_q-0.1 (full trio)
# Waits for GPU1 current job (Llama Subtask4 Q full497, PID 2674269) before launch.
set -u

REPO=/home/woori/workspace_common/boltzmann-attention
cd "$REPO"
source /home/woori/workspace_common/CDP/poc/set.env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

QWEN="Qwen/Qwen2.5-7B-Instruct"
BONT_QWEN="external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt"

LOG_DIR=logs/qkv_joint_2026_04_15
OUT_DIR=reports/qkv_joint_2026_04_15
mkdir -p "$LOG_DIR" "$OUT_DIR"

WAIT_PID=2674269
echo "[qkv617] wait GPU1 PID=$WAIT_PID at $(date)" >> "$LOG_DIR/wave.log"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[qkv617] GPU1 free at $(date), launching" >> "$LOG_DIR/wave.log"

# 5 methods: baseline, Q-only ref, V+Q (no K), K+Q (no V), full trio
python scripts/ocq/eval_metatool_subtask4.py \
    --model "$QWEN" --device cuda:1 \
    --methods \
        no_steer \
        ocq_qbias_b-0.1 \
        ocq_qkv_a0_v0.05_q-0.1 \
        ocq_qkv_a0.05_v0_q-0.1 \
        ocq_qkv_a0.05_v0.05_q-0.1 \
    --b-ont "$BONT_QWEN" \
    --max-new-tokens 256 \
    --out "$OUT_DIR/full497_smallA_trio.json" \
    > "$LOG_DIR/eval.log" 2>&1
RC=$?
echo "[qkv617] eval done rc=$RC at $(date)" >> "$LOG_DIR/wave.log"

# Quick summary to log
python3 -c "
import json
d=json.load(open('$OUT_DIR/full497_smallA_trio.json'))
print('=== Thm 6.17 small-α QKV joint full 497 ===')
for r in d['results']:
    m=r['macro']
    print(f\"  {r['method']:35s} F1={m['F1']:.4f} Exact={m['Exact']:.4f}\")
" >> "$LOG_DIR/wave.log" 2>&1

echo "[qkv617] WAVE COMPLETE at $(date)" >> "$LOG_DIR/wave.log"
