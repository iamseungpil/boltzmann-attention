#!/bin/bash
# Flow-discipline 배포-arm (compliant-pass 비교·사용자 지시 2026-06-22).
# G5-only(격리)와 별도: full-gate 위에서 G5 한계기여 + compliant-pass(full=G1-G4 무위반) 측정.
#   armA G1-G4 (auth,confirm,ownership,notice)      = compliant 베이스라인 scaffold
#   armB G1-G5 (+preconditions)                     = +G5 한계
#   armC G1-G5 + retry-controller                   = +retry
# 각 arm = t2_run_gated compliance 후크가 compliance.json(bench/write/strict/full pass^k) 자동산출.
set -u
GPU=0; PORT=8360
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
TB=/home/woori/scratch/tau2-bench
M=Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8
NT="${1:-1}"
LOG=$S/flow_disc_fullgate.log
exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --ff-only
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2

run_arm () {  # $1=save_to  $2..=env
  local save=$1; shift
  echo "######## ARM $save (env: $*) ########"; date
  cd $TB
  rm -rf "$TB/data/simulations/$save"
  env "$@" PYTHONPATH=src:$T2 $PY $T2/t2_run_gated.py \
    --gate 1 --domain retail \
    --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
    --num_trials $NT --max_concurrency 8 --save_to "$save" || echo "ARM_FAIL $save"
  echo "ARM_DONE $save"; date
}

# serve (reuse if up)
if curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$M"; then
  echo "SERVE_REUSE port=$PORT"; date
else
  for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve "$M" --port $PORT \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len 16384 --enforce-eager --gpu-memory-utilization 0.92 \
    > $S/vllm_flowdisc.log 2>&1 &
  ok=0; for i in $(seq 1 120); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$M" && ok=1 && break; sleep 10; done
  [ $ok = 1 ] || { echo "SERVE_FAIL"; tail -40 $S/vllm_flowdisc.log; exit 1; }
  echo "SERVE_OK port=$PORT"; date
fi

run_arm on_n32int8_g14_retail      T2_GATE_KINDS=auth,confirm,ownership,notice
run_arm on_n32int8_g15_retail      T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions
run_arm on_n32int8_g15retry_retail T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions T2_RETRY_CONTROLLER=1 T2_RETRY_K=3

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "FLOW_DISC_FULLGATE_DONE"; date
