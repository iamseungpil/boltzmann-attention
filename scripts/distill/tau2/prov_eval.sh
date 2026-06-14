#!/bin/bash
# R1b provenance 게이트 프로토타입 (무재학습): 날조 인자 차단 시 τ² pass 변화 측정.
# Arm A = T2_PROVENANCE=1(거부=error) / Arm B = +T2_PROV_SOFT=1(거부 budget 안 셈).
# baseline = 게이트 없음(별도·coupling_v7 0.05). 한 번 serve → 두 arm.
# Usage: prov_eval.sh <adapter> <lora_name> <gpu> <port> <n> <label>
set -u
ADAPTER=${1:?adapter}; NAME=${2:?lora}; GPU=${3:?gpu}; PORT=${4:?port}; N=${5:-20}; LABEL=${6:?label}
T2=/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
LOG=$S/prov_${LABEL}.log
exec > $LOG 2>&1; set -x; date

# serve
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
  --enable-lora --lora-modules ${NAME}=$ADAPTER --max-lora-rank 32 > $S/vllm_${LABEL}.log 2>&1 &
ok=0; for i in $(seq 1 60); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$NAME" && ok=1 && break; sleep 10; done
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
  grep -E "violations=" $S/t2_${LABEL}_${tag}.log | tail -1
}

run_arm A T2_PROVENANCE=1
run_arm B T2_PROVENANCE=1 T2_PROV_SOFT=1

# free
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "===== PROV DONE [$LABEL] ====="; date; echo PROV_DONE_${LABEL}
