#!/bin/bash
# Banking variant experiment: test 3 ontology rebalancing strategies
# Run after B_ont for variants A/B/C are built

source /home/woori/venvs/seka_env/bin/activate

GPU=${1:-0}
N=${2:-0}  # 0 = all 97 tasks

MODEL="Qwen/Qwen2.5-7B-Instruct"
OUTDIR="reports/tau2_2026_04_17"
mkdir -p "$OUTDIR"

for V in A B C; do
    BONT="external/SEKA/seka_projections/ontology-qwen25-7b-tau2-banking-${V}/B_ont.pt"
    if [ ! -f "$BONT" ]; then
        echo "[skip] $BONT not found"
        continue
    fi
    echo ""
    echo "=== Banking variant $V, GPU=$GPU ==="
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/ocq/eval_tau2_bench.py \
        --model "$MODEL" \
        --device cuda:0 \
        --b-ont "$BONT" \
        --domain banking_knowledge \
        --methods no_steer ocq_qbias_b-0.03 ocq_ladapt_k0.05_q-0.03 \
                  multipass_ocq_ladapt_k0.05_q-0.03 \
        --max-samples "$N" \
        --max-new-tokens 300 \
        --multipass-n 3 \
        --out "$OUTDIR/banking_variant${V}.json"
done

echo ""
echo "=== Banking variants complete ==="
