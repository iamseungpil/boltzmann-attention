#!/bin/bash
# C3 재측정 — fetch-first를 *다양한 강한 프롬프트* vs engine(autofetch) head-to-head (같은 serve·같은 태스크).
# 동기: "7B engine>prompt"가 배치 섞임+프롬프트 1종이라 결함(사용자). 깨끗한 paired 비교로 진실 확인.
# arms: base / NOFAB / fetchfirst / fewshot / structured / engine.  측정=pass^1 + A_notfound.
# Usage: c3_prompt_sweep.sh <gpu> <port> [num_tasks]
set -u
GPU=${1:?}; PORT=${2:?}; NT=${3:-114}
BASE=Qwen/Qwen2.5-7B-Instruct
R=/home/woori/workspace_common/boltzmann-attention-pi; T2=$R/scripts/distill/tau2; A2=$T2/a2
PY=/home/woori/venvs/seka_env/bin/python; VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
LOG=$S/c3_prompt_sweep.log; exec > $LOG 2>&1; set -x; date

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 12
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 --enforce-eager \
  > $S/vllm_c3sweep.log 2>&1 &
ok=0; for i in $(seq 1 120); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$BASE" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -40 $S/vllm_c3sweep.log; exit 1; }
hc=$(curl -s localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$BASE\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3}" 2>/dev/null)
echo "$hc" | grep -q '"content"' || { echo "HC_FAIL: $hc"; exit 1; }; echo "[hc OK]"

set +x; source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
cd $S/tau2-bench; export PYTHONPATH=src:$T2

cell () {  # $1=tag $2=gate $3=prov $4=autofetch $5=promptfile(or -)
  local TAG=$1 G=$2 PV=$3 AF=$4 PF=$5 SAVE=c3_$1 RARG=""
  [ "$PV" = "1" ] && export T2_PROVENANCE=1 || unset T2_PROVENANCE
  [ "$AF" = "1" ] && export T2_AUTOFETCH=1 || unset T2_AUTOFETCH
  unset T2_RETRY_CONTROLLER
  [ "$PF" != "-" ] && RARG="--rules_prompt $A2/$PF"
  rm -rf data/simulations/$SAVE
  echo "######## $TAG (gate=$G prov=$PV af=$AF prompt=$PF) ########"; date
  $PY $T2/t2_run_gated.py --gate $G --resolve 0 --domain retail --num_trials 1 --num_tasks $NT \
    $RARG --agent_model "$BASE" --agent_base http://localhost:$PORT/v1 \
    --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 --save_to $SAVE > $S/c3_${TAG}.log 2>&1 || echo FAIL
  local P=$(grep -oE "pass1=[0-9]+/[0-9]+" $S/c3_${TAG}.log | tail -1)
  local F=$($PY $T2/t2_failcensus_deep.py data/simulations/$SAVE 2>/dev/null | grep -E "A_notfound|B_wrong" | tr '\n' ' ')
  echo "C3SWEEP_ROW $TAG pass=$P | $F"
}

cell base       0 0 0 -
cell nofab      0 0 0 NOFAB_PROMPT.txt
cell fetchfirst 0 0 0 C3_FETCHFIRST.txt
cell fewshot    0 0 0 C3_FEWSHOT.txt
cell structured 0 0 0 C3_STRUCTURED.txt
cell engine     1 1 1 -

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "=== C3 SWEEP SUMMARY ==="; grep C3SWEEP_ROW $LOG
date; echo C3_SWEEP_DONE
