#!/bin/bash
# Overnight batch queue, GPU0 (2026-06-10 night):
#   [wait] RFT round-1 chain (already running) -> [2] 1.5B lodo_mm SFT+eval -> [3] Qwen3 curve.
# Self-sequencing on the remote — survives client disconnects.
R=/home/woori/workspace_common/boltzmann-attention-pi
RFT=/home/woori/scratch/tb_rft
LOG=/home/woori/scratch/tb_night_gpu0.log
exec > $LOG 2>&1
set -x

# [wait] RFT chain end (up to 12h)
for i in $(seq 1 720); do
  grep -q "RFT_ROUND_DONE_rft_mm" $RFT/round_rft_mm.log && break
  sleep 60
done
grep -q "RFT_ROUND_DONE_rft_mm" $RFT/round_rft_mm.log || echo "WARN_RFT_TIMEOUT_PROCEEDING"

# free GPU0 of any leftover vllm
for p in $(nvidia-smi --id=0 --query-compute-apps=pid,process_name --format=csv,noheader | grep -i vllm | cut -d, -f1); do kill -9 $p; done
sleep 15

# [2] 1.5B lodo_mm SFT (same protocol as 7B/14B) + eval
BASE=Qwen/Qwen2.5-1.5B-Instruct PREFIX=qwen15b bash $R/scripts/distill/taskbench/tb_train_lodo.sh lodo_mm_15b 0 "data_huggingface data_dailylifeapis"
BASEM=Qwen/Qwen2.5-1.5B-Instruct ADAPTER=$R/reports/facet_rft_2026/phase4_distill/sft_runs/qwen15b_tb_lodo_mm_15b \
  bash $R/scripts/distill/taskbench/tb_eval_adapter.sh lodo_mm_15b data_multimedia "data_huggingface data_dailylifeapis" 0 0.85
echo "BATCH_15B_DONE $(date)"

# [3] Qwen3 prompted curve (deferred filler; non-thinking enforced via inference.py patch)
for p in $(nvidia-smi --id=0 --query-compute-apps=pid,process_name --format=csv,noheader | grep -i vllm | cut -d, -f1); do kill -9 $p; done
sleep 15
bash $R/scripts/distill/taskbench/tb_scale_curve_qwen3.sh 0 8000 "0.6 1.7 4 14"
echo "NIGHT_BATCH_GPU0_DONE $(date)"
