#!/bin/bash
# node_run_sft32b.sh — COWORKER plan v1.42 #1 (train node): 32B t1c LODO-bank SFT.
# Reproduces the 7B recipe (qwen7b_tbox_t1c_lodo_bank) with --base-model swapped to 32B:
#   data  = build_tbox_planner_sft --alias --source 1 --treeval  (slot-fix HEAD), 6 non-bank domains
#   train = lora_train_chat_toolcall defaults (r16 a32 ep3 lr1e-4 bs1 ga16), max-seq-len 4096
#           (LODO-bank data measured max ~2051 tok)
# Single H200 141GB: 32B bf16 (~64GB) + LoRA + seq4096 bs1 grad-accum fits.
set -ex
export HF_HUB_CACHE=/scratch/hf_cache
PYT=/scratch/venvs/sop_env/bin/python
REPO=/scratch/boltzmann-attention
CL=/scratch/SOPBench
cd $CL

# 1. induced ABox + getter map + t1c SFT data (idempotent, verified locally 2026-06-10)
[ -f induced/ontology_bank.json ] || PYTHONPATH=. $PYT $REPO/scripts/distill/sopbench/induce_ontology_zekun.py
[ -f induced/getter_map.json ]   || PYTHONPATH=. $PYT $REPO/scripts/distill/sopbench/autoderive_getter_map.py --out induced/getter_map.json
if [ ! -f sft_tbox/lodo_train_holdout_bank.jsonl ]; then
  for d in bank dmv healthcare hotel library online_market university; do
    PYTHONPATH=. $PYT $REPO/scripts/distill/sopbench/build_tbox_planner_sft.py \
      --domain $d --alias --source 1 --treeval --out sft_tbox
  done
  cat sft_tbox/sft_tbox_{dmv,healthcare,hotel,library,online_market,university}_alias_gate_scratch_treeval.jsonl \
    > sft_tbox/lodo_train_holdout_bank.jsonl
fi
wc -l sft_tbox/lodo_train_holdout_bank.jsonl   # expect 4189

# 2. train — 32B bf16 (~65GB) needs 2x H100 80GB: --device auto shards across the pair
CUDA_VISIBLE_DEVICES=${TRAIN_GPUS:-0,1} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PYT $REPO/scripts/distill/lora_train_chat_toolcall.py \
  --base-model Qwen/Qwen2.5-32B-Instruct --device auto \
  --train-jsonl $CL/sft_tbox/lodo_train_holdout_bank.jsonl \
  --out-dir /scratch/sft_runs/qwen32b_tbox_t1c_lodo_bank \
  --max-seq-len 4096 \
  --save-every 100 --resume \
  > /scratch/logs/sft32b_lodo_bank.log 2>&1
echo "SFT_DONE -> /scratch/sft_runs/qwen32b_tbox_t1c_lodo_bank"
