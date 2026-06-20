#!/bin/bash
# ReST S1 eval — qwen7b_rest_s0 어댑터 GO/NO-GO (REST_INTERNALIZE_DESIGN §7).
# base+LoRA 한 serve서 셀 순차. 셀=pass^1 + fab-A(failcensus_deep). user-sim=gpt-4.1.
# GO 기준: (1)retail pass > base, (2)airline 전이>base(held-out·ABox-swap), (3)scaffold-drop(floor=gate0)서 날조 안 복귀.
# Usage: rest_s1_eval.sh <gpu> <port> [num_tasks]
set -u
GPU=${1:?}; PORT=${2:?}; NT=${3:-50}
ADAPTER=/home/woori/scratch/adapters/qwen7b_rest_s0_r16
R=/home/woori/workspace_common/boltzmann-attention-pi; T2=$R/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python; VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch; BASE=Qwen/Qwen2.5-7B-Instruct
LOG=$S/rest_s1_eval.log; exec > $LOG 2>&1; set -x; date

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 12
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 --enforce-eager \
  --enable-lora --max-lora-rank 16 --lora-modules rest_s0=$ADAPTER > $S/vllm_rest_s1eval.log 2>&1 &
ok=0; for i in $(seq 1 120); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$BASE" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -40 $S/vllm_rest_s1eval.log; exit 1; }
hc=$(curl -s localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$BASE\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3}" 2>/dev/null)
echo "$hc" | grep -q '"content"' || { echo "HC_FAIL: $hc"; exit 1; }; echo "[hc OK]"

set +x; source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
cd $S/tau2-bench; export PYTHONPATH=src:$T2

cell () {  # $1=served_model $2=domain $3=gate $4=resolve $5=tag
  local M=$1 D=$2 G=$3 RES=$4 TAG=$5 SAVE=s1eval_${5}
  [ "$G" = "1" ] && export T2_PROVENANCE=1 || unset T2_PROVENANCE
  rm -rf data/simulations/$SAVE
  echo "######## $TAG (model=$M domain=$D gate=$G res=$RES) ########"; date
  $PY $T2/t2_run_gated.py --gate $G --resolve $RES --domain $D --num_trials 1 --num_tasks $NT \
    --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
    --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 --save_to $SAVE > $S/s1eval_${TAG}.log 2>&1 || echo FAIL
  local P=$(grep -oE "pass1=[0-9]+/[0-9]+" $S/s1eval_${TAG}.log | tail -1)
  local F=$($PY $T2/t2_failcensus_deep.py data/simulations/$SAVE 2>/dev/null | grep -E "A_notfound|A\." | tr '\n' ' ')
  echo "S1_ROW $TAG pass=$P | $F"
}

# retail: floor(scaffold-drop=discipline 내재화 시험) + gated(배포config) · base vs rest_s0
cell $BASE   retail 0 0 base_retail_floor
cell rest_s0 retail 0 0 rest_retail_floor
cell rest_s0 retail 1 0 rest_retail_gated
# airline 전이(held-out·ABox-swap·floor=순수 내재화 discipline) · base vs rest_s0
cell $BASE   airline 0 0 base_airline_floor
cell rest_s0 airline 0 0 rest_airline_floor

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "=== REST S1 EVAL SUMMARY ==="; grep S1_ROW $LOG
date; echo REST_S1_EVAL_DONE
