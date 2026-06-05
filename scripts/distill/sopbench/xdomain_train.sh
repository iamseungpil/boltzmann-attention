#!/bin/bash
# Cross-domain transfer adapter training (t1c recipe: alias+source1+treeval+scratchpad, LoRA r16, 3ep).
# Usage: xdomain_train.sh NAME GPU "train_domain_list"
#   LODO-holdout-X: train = 7 domains minus X  (test X = held-out)
#   train-1:        train = single domain       (test the other 6 = held-out)
# Adapter -> $RUNS/qwen7b_tbox_<NAME>. Per-domain SFT jsonl already built (sft_tbox_<d>_alias_gate_scratch_treeval.jsonl).
NAME=$1; GPU=$2; shift 2; TRAIN_DOMS="$@"
REPO=/home/woori/workspace_common/boltzmann-attention-pi
OUT=/home/woori/scratch/sft_alias_run
RUNS=$REPO/reports/facet_rft_2026/phase4_distill/sft_runs
PY=/home/woori/venvs/seka_env/bin/python
TR=$REPO/scripts/distill/lora_train_chat_toolcall.py
LOG=$OUT/xtrain_${NAME}.log
exec > $LOG 2>&1
set -x
TRAIN=$OUT/lodo_train_${NAME}.jsonl
: > $TRAIN
for d in $TRAIN_DOMS; do cat $OUT/sft_tbox_${d}_alias_gate_scratch_treeval.jsonl >> $TRAIN 2>/dev/null; done
wc -l $TRAIN
echo "TRAIN_DOMS=$TRAIN_DOMS  -> $NAME on GPU $GPU"
rm -f $RUNS/qwen7b_tbox_${NAME}/train_meta.json
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $TR \
  --base-model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --max-seq-len 2048 --epochs 3 --lora-r 16 \
  --val-frac 0.05 --skip-overlong \
  --train-jsonl $TRAIN --out-dir $RUNS/qwen7b_tbox_${NAME}
echo "TRAIN_DONE_${NAME} $(date)"
