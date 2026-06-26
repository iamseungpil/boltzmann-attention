#!/bin/bash
# DPO 어댑터 eval — DPO penalty가 schema-copy prior를 죽였나(retail). 핵심=schema_copy → ~0?
# arms: base / dpo (gate-deny·prior 닫혔나 단독) / dpo-perform(autofetch 병행) / dpo-pure(gate0).
# Usage: c4_dpo_eval.sh <gpu> <port> [domain] [num_tasks]
set -u
GPU=${1:?}; PORT=${2:?}; DOM=${3:-retail}; NT=${4:-114}
BASE=Qwen/Qwen2.5-7B-Instruct
ADAPTER=/home/woori/scratch/adapters/qwen7b_c4_dpo
R=/home/woori/workspace_common/boltzmann-attention-pi; T2=$R/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python; VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
LOG=$S/c4_dpo_eval_${DOM}.log; exec > $LOG 2>&1; set -x; date
[ -d $ADAPTER ] || { echo "DPO_ADAPTER_MISSING"; exit 1; }

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 12
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 --enforce-eager \
  --enable-lora --max-lora-rank 16 --lora-modules dpo=$ADAPTER > $S/vllm_c4dpo_${DOM}.log 2>&1 &
ok=0; for i in $(seq 1 120); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$BASE" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -40 $S/vllm_c4dpo_${DOM}.log; exit 1; }
echo "[serve OK]"

set +x; source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
cd $S/tau2-bench; export PYTHONPATH=src:$T2

cell () {  # $1=tag $2=model $3=gate $4=prov $5=autofetch
  local TAG=$1 MODEL=$2 G=$3 PV=$4 AF=$5 SAVE=c4dpo_${DOM}_$1
  [ "$PV" = "1" ] && export T2_PROVENANCE=1 || unset T2_PROVENANCE
  [ "$AF" = "1" ] && export T2_AUTOFETCH=1 || unset T2_AUTOFETCH
  unset T2_RETRY_CONTROLLER T2_MAXPROMPT
  rm -rf data/simulations/$SAVE
  echo "######## $TAG (model=$MODEL gate=$G prov=$PV af=$AF) ########"; date
  $PY $T2/t2_run_gated.py --gate $G --resolve 0 --domain $DOM --num_trials 1 --num_tasks $NT \
    --agent_model "$MODEL" --agent_base http://localhost:$PORT/v1 \
    --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 --save_to $SAVE > $S/c4dpo_${DOM}_${TAG}.log 2>&1 || echo FAIL
  local P=$(grep -oE "pass1=[0-9]+/[0-9]+" $S/c4dpo_${DOM}_${TAG}.log | tail -1)
  local M=$($PY $T2/c4_prompt_mechanism.py data/simulations/$SAVE 2>/dev/null | grep M_MECH_ROW)
  echo "C4DPO_ROW $DOM $TAG pass=$P | $M"
}

cell base        $BASE 0 0 0
cell dpo-pure    dpo   0 0 0
cell dpo-deny    dpo   1 1 0
cell dpo-perform dpo   1 1 1

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "=== DPO EVAL ($DOM) ==="; grep "C4DPO_ROW $DOM" $LOG
date; echo C4DPO_EVAL_DONE
