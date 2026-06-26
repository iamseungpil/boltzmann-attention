#!/bin/bash
# DPO penalty (fetch-first schema-copy prior 억제) — cfbsynth verifier-labeled pairs, DPO from base.
# SFT 실패(schema_copy 52) → DPO가 copy 행동을 *벌점*으로 직접 누름(선행연구 NPO/DPO=prior 억제).
# Usage: c4_dpo_build.sh <gpu> [npairs] [beta] [epochs] [lr]
set -u
GPU=${1:?}; NP=${2:-4000}; BETA=${3:-0.1}; EP=${4:-2}; LR=${5:-5e-6}
REPO=/home/woori/workspace_common/boltzmann-attention-pi; DIST=$REPO/scripts/distill; T2=$DIST/tau2
PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; B=$S/c4_dpo; TAG=qwen7b_c4_dpo
mkdir -p $B; LOG=$S/c4_dpo_build.log; exec > $LOG 2>&1; set -x; date
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. verifier-labeled pairs (chosen=copy real id / rejected=copy schema-example)
$PY $T2/cfbsynth_dpo_pairs.py --out $B/dpo_pairs.jsonl --n $NP --seed 0
wc -l $B/dpo_pairs.jsonl
# tau2 학습 금지 게이트
HIT=$(grep -cE "get_user_details|get_order_details|cancel_pending|exchange_delivered|modify_pending|modify_user_address|return_delivered|find_user_id|book_reservation|cancel_reservation|get_reservation|update_reservation|send_certificate|transfer_to_human" $B/dpo_pairs.jsonl || true)
echo "tau2-tool grep = $HIT (expect 0)"; [ "$HIT" != "0" ] && { echo TAU2_CONTAMINATION_ABORT; exit 2; }

# 2. fresh identity LoRA (B=0 → policy init=base·ref=base = DPO from base)
$PY - <<'EOF'
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16)
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
get_peft_model(m, cfg).save_pretrained("/home/woori/scratch/c4_dpo/fresh_lora")
print("fresh identity LoRA saved")
EOF

# 3. DPO (policy+ref 2x7B·gradient checkpointing·48GB 1-GPU·진행률 가시)
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 6
rm -rf $S/adapters/$TAG
CUDA_VISIBLE_DEVICES=$GPU $PY -u $DIST/sopbench/dpo_train.py \
  --base Qwen/Qwen2.5-7B-Instruct --sft-adapter $B/fresh_lora --pairs $B/dpo_pairs.jsonl \
  --out-dir $S/adapters/$TAG --device cuda:0 \
  --beta $BETA --lr $LR --epochs $EP --grad-accum 8 --max-seq-len 3072 --log-every 10
[ -d $S/adapters/$TAG ] && echo C4_DPO_DONE || echo DPO_FAIL
date
