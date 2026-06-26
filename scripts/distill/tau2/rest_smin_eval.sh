#!/bin/bash
# S-min 3-arm: base 7B의 grounding(A) 격차를 엔진이 닫나 (학습 전 판정).
# arm0=base / arm1a=+provenance(차단만) / arm1b=+provenance+autofetch(실값 주입).
# 한 serve(base 7B)서 셀 순차. 셀=pass^1 + failcensus(A=order_id grounding).
# Usage: rest_smin_eval.sh <gpu> <port> [num_tasks]
set -u
GPU=${1:?}; PORT=${2:?}; NT=${3:-114}
BASE=Qwen/Qwen2.5-7B-Instruct
R=/home/woori/workspace_common/boltzmann-attention-pi; T2=$R/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python; VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
LOG=$S/rest_smin_eval.log; exec > $LOG 2>&1; set -x; date

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 12
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 --enforce-eager \
  > $S/vllm_smin.log 2>&1 &
ok=0; for i in $(seq 1 120); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$BASE" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -40 $S/vllm_smin.log; exit 1; }
hc=$(curl -s localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$BASE\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3}" 2>/dev/null)
echo "$hc" | grep -q '"content"' || { echo "HC_FAIL: $hc"; exit 1; }; echo "[hc OK]"

set +x; source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
cd $S/tau2-bench; export PYTHONPATH=src:$T2

cell () {  # $1=tag $2=gate $3=prov $4=autofetch
  local TAG=$1 G=$2 PV=$3 AF=$4 SAVE=smin_$1
  [ "$PV" = "1" ] && export T2_PROVENANCE=1 || unset T2_PROVENANCE
  [ "$AF" = "1" ] && export T2_AUTOFETCH=1 || unset T2_AUTOFETCH
  rm -rf data/simulations/$SAVE
  echo "######## $TAG (gate=$G prov=$PV autofetch=$AF) ########"; date
  $PY $T2/t2_run_gated.py --gate $G --resolve 0 --domain retail --num_trials 1 --num_tasks $NT \
    --agent_model "$BASE" --agent_base http://localhost:$PORT/v1 \
    --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 --save_to $SAVE > $S/smin_${TAG}.log 2>&1 || echo FAIL
  local P=$(grep -oE "pass1=[0-9]+/[0-9]+" $S/smin_${TAG}.log | tail -1)
  local F=$($PY $T2/t2_failcensus_deep.py data/simulations/$SAVE 2>/dev/null | grep -E "A_notfound|B_wrong" | tr '\n' ' ')
  echo "SMIN_ROW $TAG pass=$P | $F"
}

cell arm0_base       0 0 0
cell arm1a_prov      1 1 0
cell arm1b_autofetch 1 1 1

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "=== S-MIN SUMMARY ==="; grep SMIN_ROW $LOG
date; echo REST_SMIN_DONE
