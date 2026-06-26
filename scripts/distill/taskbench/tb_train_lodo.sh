#!/bin/bash
# TaskBench LODO distill-SFT (Exp-A, HANDOFF_2026_06_10 §4).
# Usage: tb_train_lodo.sh NAME GPU "dom1 dom2"   (train domains; held-out = the third)
# e.g.   tb_train_lodo.sh lodo_mm 1 "data_huggingface data_dailylifeapis"
NAME=$1; GPU=$2; shift 2; TRAIN_DOMS="$@"
# optional env overrides: BASE (HF model id), PREFIX (adapter dir prefix), EPOCHS
BASE=${BASE:-Qwen/Qwen2.5-7B-Instruct}; PREFIX=${PREFIX:-qwen7b}; EPOCHS=${EPOCHS:-2}
REPO=/home/woori/workspace_common/boltzmann-attention-pi
TB=/home/woori/scratch/JARVIS_tb/taskbench
OUT=/home/woori/scratch/tb_sft
RUNS=$REPO/reports/facet_rft_2026/phase4_distill/sft_runs
PY=/home/woori/venvs/seka_env/bin/python
mkdir -p $OUT
LOG=$OUT/tbtrain_${NAME}.log
exec > $LOG 2>&1
set -x
TRAIN=$OUT/train_${NAME}.jsonl
: > $TRAIN
for d in $TRAIN_DOMS; do
  $PY $REPO/scripts/distill/taskbench/tb_build_sft.py --tb_dir $TB --domain $d \
    --out $OUT/sft_${d}.jsonl --n_single 400 --n_chain 1000 --n_dag 0 || exit 1
  cat $OUT/sft_${d}.jsonl >> $TRAIN
done
wc -l $TRAIN
echo "TRAIN_DOMS=$TRAIN_DOMS -> $NAME on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY \
  $REPO/scripts/distill/lora_train_chat_toolcall.py \
  --base-model $BASE --device cuda:0 --max-seq-len 6144 --epochs $EPOCHS \
  --lora-r 16 --val-frac 0.02 --skip-overlong \
  --train-jsonl $TRAIN --out-dir $RUNS/${PREFIX}_tb_${NAME}
echo "TRAIN_DONE_${NAME} $(date)"
