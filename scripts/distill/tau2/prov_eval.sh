#!/bin/bash
# R1b provenance 프로토타입 (무재학습) — 깨끗한 A/B (동일 serve·max-len 32768로 ctx-window 혼란 제거).
#  BASE  = 게이트만(provenance off)              ← 같은 조건 baseline (이전 0.05 재측정)
#  L1    = bad_words 디코드-마스크(스키마-example+placeholder 차단)   ← ★당신 제안
#  L1L2  = bad_words + 검증기 내부재생성(세션 동적 블랙리스트)
# Usage: prov_eval.sh <adapter> <lora_name> <gpu> <port> <n> <label>
set -u
ADAPTER=${1:?adapter}; NAME=${2:?lora}; GPU=${3:?gpu}; PORT=${4:?port}; N=${5:-20}; LABEL=${6:?label}
T2=/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
LOG=$S/prov_${LABEL}.log
exec > $LOG 2>&1; set -x; date

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768 \
  --enable-lora --lora-modules ${NAME}=$ADAPTER --max-lora-rank 32 > $S/vllm_${LABEL}.log 2>&1 &
ok=0; for i in $(seq 1 70); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$NAME" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -25 $S/vllm_${LABEL}.log; exit 1; }

set +x; source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
cd $S/tau2-bench; export PYTHONPATH=src:$T2

run_arm () {  # $1=armtag  $2..=env assigns
  local tag=$1; shift
  rm -rf data/simulations/retail_${LABEL}_${tag}
  env "$@" $PY $T2/t2_run_gated.py --gate 1 --num_trials 1 --num_tasks $N --agent_model $NAME \
    --agent_base http://localhost:$PORT/v1 --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 \
    --save_to retail_${LABEL}_${tag} > $S/t2_${LABEL}_${tag}.log 2>&1 || echo "T2_FAIL_${tag}"
  echo "===== ARM ${tag} ====="
  $PY -c "import json;print('pass^1=',json.load(open('$S/tau2-bench/data/simulations/retail_${LABEL}_${tag}/compliance.json'))['bench']['pass^1'])" 2>/dev/null || echo NA
  grep -E "provenance L1|violations=" $S/t2_${LABEL}_${tag}.log | tail -2
  # 날조율 진단
  $PY -c "
import json,glob
d=json.load(open('$S/tau2-bench/data/simulations/retail_${LABEL}_${tag}/results.json'))
fab=guds=0
for s in d['simulations']:
  for m in s.get('messages') or []:
    if m.get('role')=='assistant':
      for tc in m.get('tool_calls') or []:
        fn=tc.get('function',tc); nm=fn.get('name') or ''; ar=str(fn.get('arguments'))
        if nm=='get_user_details': guds+=1
        if '#W000' in ar or 'example.com' in ar: fab+=1
print(f'  [diag] 날조(#W000/example.com)={fab} get_user_details={guds}')
" 2>/dev/null
}

run_arm BASE
run_arm L1   T2_PROV_BADWORDS=1
run_arm L1L2 T2_PROV_BADWORDS=1 T2_PROV_REGEN=1

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "===== PROV DONE [$LABEL] ====="; date; echo PROV_DONE_${LABEL}
