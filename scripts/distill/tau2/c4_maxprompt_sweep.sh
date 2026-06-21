#!/bin/bash
# 프롬프트 한계 실험 (사용자 2026-06-22): 7B에서 fetch-first를 *최대로 강하게* + *위치반복*
# (앞/중간/끝)해도 schema-example-copy prior가 안 죽나. arms 모두 base 7B·동일 태스크·paired.
#   max_begin    : P_MAX system(앞)만 = 최대 CONTENT 강도, 단일위치
#   max_be       : P_MAX 앞+끝(recency 반복)
#   max_bme      : P_MAX 앞+중간+끝 (위치 포화 + 반복)
# 대조(기존): fetchfirst(보통 지시 0.31)·fewshot(시연 0.00). 측정 = schema_copy 율(c4_prompt_mechanism).
# Usage: c4_maxprompt_sweep.sh <gpu> <port> [num_tasks]
set -u
GPU=${1:?}; PORT=${2:?}; NT=${3:-114}
BASE=Qwen/Qwen2.5-7B-Instruct
R=/home/woori/workspace_common/boltzmann-attention-pi; T2=$R/scripts/distill/tau2; A2=$T2/a2
PY=/home/woori/venvs/seka_env/bin/python; VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
LOG=$S/c4_maxprompt_sweep.log; exec > $LOG 2>&1; set -x; date

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 12
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 --enforce-eager \
  > $S/vllm_maxprompt.log 2>&1 &
ok=0; for i in $(seq 1 120); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$BASE" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -40 $S/vllm_maxprompt.log; exit 1; }
echo "[serve OK]"

set +x; source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
cd $S/tau2-bench; export PYTHONPATH=src:$T2

cell () {  # $1=tag  $2=positions(begin/middle/end csv or -)  $3=promptfile(for rules-only arms or -)
  local TAG=$1 POS=$2 PF=$3 SAVE=c4mp_$1 RARG=""
  unset T2_PROVENANCE T2_AUTOFETCH T2_RETRY_CONTROLLER T2_MAXPROMPT T2_MAXPROMPT_POS
  if [ "$POS" != "-" ]; then export T2_MAXPROMPT=$A2/C4_MAXPROMPT.txt T2_MAXPROMPT_POS=$POS; fi
  [ "$PF" != "-" ] && RARG="--rules_prompt $A2/$PF"
  rm -rf data/simulations/$SAVE
  echo "######## $TAG (maxpos=$POS prompt=$PF) ########"; date
  $PY $T2/t2_run_gated.py --gate 0 --resolve 0 --domain retail --num_trials 1 --num_tasks $NT \
    $RARG --agent_model "$BASE" --agent_base http://localhost:$PORT/v1 \
    --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 --save_to $SAVE > $S/c4mp_${TAG}.log 2>&1 || echo FAIL
  local P=$(grep -oE "pass1=[0-9]+/[0-9]+" $S/c4mp_${TAG}.log | tail -1)
  local F=$($PY $T2/c4_prompt_mechanism.py data/simulations/$SAVE 2>/dev/null | grep M_MECH_ROW)
  echo "C4MP_ROW $TAG pass=$P | $F"
}

cell max_begin  begin        -
cell max_be     begin,end    -
cell max_bme    begin,middle,end -

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "=== MAXPROMPT SWEEP ==="; grep -E "C4MP_ROW" $LOG
date; echo C4_MAXPROMPT_DONE
