#!/usr/bin/env bash
set -u
REPO=/home/woori/workspace_common/boltzmann-attention
cd "$REPO"
source /home/woori/workspace_common/CDP/poc/set.env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="Qwen/Qwen2.5-7B-Instruct"
BONT="external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt"
LOG_DIR=logs/contrastive_full_2026_04_15
OUT_DIR=reports/contrastive_full_2026_04_15
mkdir -p "$LOG_DIR" "$OUT_DIR"

echo "[ctr497] start $(date)" >> "$LOG_DIR/chain.log"

# Full 497 contrastive a=0.3 d=3 (smoke winner +5.8pp)
python scripts/ocq/eval_metatool_subtask4.py \
    --model "$MODEL" --device cuda:0 \
    --methods no_steer ocq_cbias_a0.3_d3 \
    --b-ont "$BONT" \
    --out "$OUT_DIR/st4_contrastive_a0.3_d3_full497.json" \
    > "$LOG_DIR/ctr_a0.3_d3.log" 2>&1

echo "[ctr497] a0.3 d3 done $(date)" >> "$LOG_DIR/chain.log"

# Also run d=1 (smoke +3.3pp) and α-sweep 0.15 as companions
python scripts/ocq/eval_metatool_subtask4.py \
    --model "$MODEL" --device cuda:0 \
    --methods no_steer ocq_cbias_a0.3_d1 \
    --b-ont "$BONT" \
    --out "$OUT_DIR/st4_contrastive_a0.3_d1_full497.json" \
    > "$LOG_DIR/ctr_a0.3_d1.log" 2>&1

echo "[ctr497] a0.3 d1 done $(date)" >> "$LOG_DIR/chain.log"

python scripts/ocq/eval_metatool_subtask4.py \
    --model "$MODEL" --device cuda:0 \
    --methods no_steer ocq_bias_a0.15 \
    --b-ont "$BONT" \
    --out "$OUT_DIR/st4_alpha_sweep_015_full497.json" \
    > "$LOG_DIR/alpha_015.log" 2>&1

echo "[ctr497] all done $(date)" >> "$LOG_DIR/chain.log"

python3 -c "
import json, glob
print('=== Contrastive full 497 summary ===')
for p in sorted(glob.glob('$OUT_DIR/*.json')):
    d=json.load(open(p))
    for r in d['results']:
        m=r['macro']
        print(f\"{p.split('/')[-1]:50s} {r['method']:30s} n={d['n_queries']} F1={m['F1']:.3f} rec={m['recall']:.3f} Exact={m['Exact']:.3f}\")
" >> "$LOG_DIR/summary.log" 2>&1
