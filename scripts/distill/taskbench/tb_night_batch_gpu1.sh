#!/bin/bash
# Overnight batch queue, GPU1 (2026-06-10 night):
#   [wait] 14B lodo_mm SFT (already running, long) -> eval 14B adapter.
R=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/tb_night_gpu1.log
exec > $LOG 2>&1
set -x

# [wait] 14B training end (up to 26h)
for i in $(seq 1 1560); do
  grep -q "TRAIN_DONE_lodo_mm_14b" /home/woori/scratch/tb_sft/tbtrain_lodo_mm_14b.log && break
  sleep 60
done
grep -q "TRAIN_DONE_lodo_mm_14b" /home/woori/scratch/tb_sft/tbtrain_lodo_mm_14b.log || { echo "ERR_14B_TIMEOUT"; exit 1; }

BASEM=Qwen/Qwen2.5-14B-Instruct ADAPTER=$R/reports/facet_rft_2026/phase4_distill/sft_runs/qwen14b_tb_lodo_mm_14b \
  bash $R/scripts/distill/taskbench/tb_eval_adapter.sh lodo_mm_14b data_multimedia "data_huggingface data_dailylifeapis" 1 0.85
echo "NIGHT_BATCH_GPU1_DONE $(date)"
