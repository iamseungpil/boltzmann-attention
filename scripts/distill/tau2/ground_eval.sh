#!/bin/bash
# ★read-id grounding leg e2e (HANDOFF_2026_06_18 §3) — base 7B native agent + config-도출
# candidate-surfacing resolver (T2_PROV_GROUND). clean A/B, 동일 serve:
#   BASE   = gate만(provenance off)                 ← base anchor 0.175 재현
#   GROUND = gate + GROUND(날조차단+grounded후보 surface+단일후보 자동치환)
# §25 처방의 결정론 leg 측정. 잔여 autopsy(tau2_collapse_autopsy)가 학습 leg 위치를 가림.
# Usage: ground_eval.sh <gpu> <port> <n> <label>
set -u
GPU=${1:?gpu}; PORT=${2:?port}; N=${3:-40}; LABEL=${4:?label}
R=/home/woori/workspace_common/boltzmann-attention-pi
T2=$R/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
LOG=$S/ground_${LABEL}.log
exec > $LOG 2>&1; set -x; date

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
# base Qwen2.5-7B (no LoRA) — served name = Qwen/Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768 \
  > $S/vllm_ground_${LABEL}.log 2>&1 &
ok=0; for i in $(seq 1 70); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "Qwen2.5-7B" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -25 $S/vllm_ground_${LABEL}.log; exit 1; }

set +x; source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
cd $S/tau2-bench; export PYTHONPATH=src:$T2

run_arm () {  # $1=armtag  $2..=env assigns
  local tag=$1; shift
  rm -rf data/simulations/retail_${LABEL}_${tag}
  env "$@" $PY $T2/t2_run_gated.py --gate 1 --num_trials 1 --num_tasks $N \
    --agent_model Qwen/Qwen2.5-7B-Instruct --agent_base http://localhost:$PORT/v1 \
    --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 \
    --save_to retail_${LABEL}_${tag} > $S/t2_ground_${LABEL}_${tag}.log 2>&1 || echo "T2_FAIL_${tag}"
  echo "===== ARM ${tag} ====="
  $PY -c "import json;print('pass^1=',json.load(open('$S/tau2-bench/data/simulations/retail_${LABEL}_${tag}/compliance.json'))['bench']['pass^1'])" 2>/dev/null || echo NA
  # 날조율 진단 (#W000 placeholder / example.com)
  $PY -c "
import json
d=json.load(open('$S/tau2-bench/data/simulations/retail_${LABEL}_${tag}/results.json'))
fab=guds=god=0
for s in d['simulations']:
  for m in s.get('messages') or []:
    if m.get('role')=='assistant':
      for tc in m.get('tool_calls') or []:
        fn=tc.get('function',tc); nm=fn.get('name') or ''; ar=str(fn.get('arguments'))
        if nm=='get_user_details': guds+=1
        if nm=='get_order_details': god+=1
        if '#W000' in ar or 'example.com' in ar: fab+=1
print(f'  [diag] placeholder-fab(#W000/example.com)={fab} get_user_details={guds} get_order_details={god}')
" 2>/dev/null
}

run_arm BASE
run_arm GROUND T2_PROV_GROUND=1

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "===== GROUND DONE [$LABEL] (anchor base=0.175 / frontier=0.81) ====="; date; echo GROUND_DONE_${LABEL}
