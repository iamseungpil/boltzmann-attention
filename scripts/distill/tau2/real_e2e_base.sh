#!/bin/bash
# 실제 τ² user_sim e2e (오프라인 op-eval 폐기·실벤치 세팅). base 7B·retail+airline·gpt-4.1 user-sim.
# = content-op(exchange/cabin)을 *실제 멀티턴 대화* 안에서 측정(user가 task_instructions를 점진 공개).
# Usage: real_e2e_base.sh <gpu> <port> [num_tasks]
set -u
GPU=${1:?gpu}; PORT=${2:?port}; NT=${3:-40}
R=/home/woori/workspace_common/boltzmann-attention-pi
T2=$R/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
BASE=Qwen/Qwen2.5-7B-Instruct
S=/home/woori/scratch
LOG=$S/real_e2e_base.log
exec > $LOG 2>&1; set -x; date

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 > $S/vllm_real_e2e.log 2>&1 &
ok=0; for i in $(seq 1 70); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "Qwen2.5-7B" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -25 $S/vllm_real_e2e.log; exit 1; }

set +x; source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
cd $S/tau2-bench; export PYTHONPATH=src:$T2

run_dom(){ # $1=domain  $2=gate
  rm -rf data/simulations/real_base_$1
  $PY $T2/t2_run_gated.py --gate $2 --domain $1 --num_trials 1 --num_tasks $NT \
    --agent_model $BASE --agent_base http://localhost:$PORT/v1 \
    --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 --save_to real_base_$1 \
    > $S/t2_real_$1.log 2>&1 || echo "FAIL $1"
  echo "=== $1 ==="; grep -E "RESULT|pass1" $S/t2_real_$1.log | tail -2
  $PY -c "import json;print('$1 pass^1',json.load(open('$S/tau2-bench/data/simulations/real_base_$1/compliance.json'))['bench']['pass^1'])" 2>/dev/null || \
  $PY -c "import json;d=json.load(open('$S/tau2-bench/data/simulations/real_base_$1/results.json'));r=[s['reward_info']['reward'] for s in d['simulations'] if s.get('reward_info')];print('$1 mean_reward',sum(r)/max(len(r),1),'n',len(r))" 2>/dev/null || echo NA
}
run_dom retail 1     # gate=retail
run_dom airline 1    # airline은 GATE_DOMAINS 밖이라 ungated 통과(agent+user-sim만)

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "=== REAL E2E DONE (user-sim·base 7B·retail+airline) ==="; date; echo REAL_E2E_DONE
