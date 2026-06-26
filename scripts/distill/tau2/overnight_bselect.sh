#!/bin/bash
# B1-selection DPO (hard negative=틀린 실후보) + eval(autofetch ON=후보 제공). 무인 4hr·GPU당 1개.
# 재타깃 learn: A(날조)가 아니라 B1(가져온 실후보 중 의미선택)·`FETCH_SELECT_DIVISION §4`.
# Usage: overnight_bselect.sh <gpu> <beta> <port> <tag> <seed>
set -u
GPU=${1:?}; BETA=${2:-0.1}; PORT=${3:?}; TAG=${4:-qwen7b_bsel}; SEED=${5:-0}
REPO=/home/woori/workspace_common/boltzmann-attention-pi; DIST=$REPO/scripts/distill; T2=$DIST/tau2
PY=/home/woori/venvs/seka_env/bin/python; VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch; B=$S/bsel_$TAG; ADAPTER=$S/adapters/$TAG
mkdir -p $B; LOG=$S/overnight_bselect_$TAG.log; exec > $LOG 2>&1; set -x; date
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. select 페어 (rejected=틀린 실후보·의미 discrimination)
$PY $T2/cfbsynth_dpo_pairs.py --out $B/pairs.jsonl --n 2500 --seed $SEED --mode select
wc -l $B/pairs.jsonl
HIT=$(grep -cE "get_user_details|get_order_details|cancel_pending|exchange_delivered|find_user_id|book_reservation|get_reservation|get_user_information" $B/pairs.jsonl || true)
echo "tau2-tool grep=$HIT (expect 0)"; [ "$HIT" != "0" ] && { echo TAU2_ABORT; exit 2; }

# 2. fresh identity LoRA (DPO from base)
$PY - <<EOF
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
m=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16)
cfg=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.0,bias="none",task_type="CAUSAL_LM",
  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
get_peft_model(m,cfg).save_pretrained("$B/fresh_lora"); print("fresh LoRA saved")
EOF

# 3. DPO-select
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 6
rm -rf $ADAPTER
CUDA_VISIBLE_DEVICES=$GPU $PY -u $DIST/sopbench/dpo_train.py \
  --base Qwen/Qwen2.5-7B-Instruct --sft-adapter $B/fresh_lora --pairs $B/pairs.jsonl \
  --out-dir $ADAPTER --device cuda:0 --beta $BETA --lr 5e-6 --epochs 2 --grad-accum 8 --max-seq-len 2048 --log-every 10
[ -d $ADAPTER ] || { echo DPO_FAIL; exit 1; }
echo BSEL_TRAIN_DONE

# 4. eval — autofetch ON(후보 제공) 고정 + {base vs bsel}. 핵심지표=B_wrong_write(선택실패).
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 8
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 --enforce-eager \
  --enable-lora --max-lora-rank 16 --lora-modules bsel=$ADAPTER > $S/vllm_$TAG.log 2>&1 &
ok=0; for i in $(seq 1 120); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q Qwen && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -30 $S/vllm_$TAG.log; exit 1; }
set +x; source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
cd $S/tau2-bench; export PYTHONPATH=src:$T2
cell () {  # tag model
  local TG=$1 MODEL=$2 SAVE=bsel_${TAG}_$1
  export T2_PROVENANCE=1 T2_AUTOFETCH=1; unset T2_RETRY_CONTROLLER T2_MAXPROMPT
  rm -rf data/simulations/$SAVE
  $PY $T2/t2_run_gated.py --gate 1 --resolve 0 --domain retail --num_trials 1 --num_tasks 114 \
    --agent_model "$MODEL" --agent_base http://localhost:$PORT/v1 \
    --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 --save_to $SAVE > $S/bsel_${TAG}_$1.log 2>&1 || echo FAIL
  local P=$(grep -oE "pass1=[0-9]+/[0-9]+" $S/bsel_${TAG}_$1.log | tail -1)
  local F=$($PY $T2/t2_failcensus_deep.py data/simulations/$SAVE 2>/dev/null | grep -E "A_notfound|B_wrong" | tr '\n' ' ')
  echo "BSEL_ROW $TAG $1 (beta=$BETA) pass=$P | $F"
}
cell base_af Qwen/Qwen2.5-7B-Instruct
cell bsel_af bsel
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "=== BSEL SUMMARY ($TAG beta=$BETA) ==="; grep "BSEL_ROW $TAG" $LOG
date; echo BSEL_DONE_$TAG
