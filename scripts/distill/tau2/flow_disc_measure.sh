#!/bin/bash
# Flow-discipline 측정 (FLOW_DISCIPLINE_SCAFFOLD_DESIGN §6·§8 step4).
# 32B-int8 retail e2e, gpt-4.1 user-sim. arm② G5-only / arm③ G5+retry. vs floor(on_n32int8_floor_retail).
# 1차 = num_trials=1 (floor trial-0 직접비교·pass^1 + §6c 클래스Δ + false-block). 신호 양성이면 3-trial.
set -u
GPU=0; PORT=8360
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
TB=/home/woori/scratch/tau2-bench
M=Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8
NT="${1:-1}"                       # num_trials (default 1)
LOG=$S/flow_disc_measure.log
exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --ff-only
source /home/woori/.openrouter_key                                  # = export OPENROUTER_API_KEY=...
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2

run_arm () {  # $1=save_to  $2..=env assignments
  local save=$1; shift
  echo "######## ARM $save (env: $*) ########"; date
  cd $TB
  env "$@" PYTHONPATH=src:$T2 $PY $T2/t2_run_gated.py \
    --gate 1 --domain retail \
    --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
    --num_trials $NT --max_concurrency 8 --save_to "$save" || echo "ARM_FAIL $save"
  echo "ARM_DONE $save"; date
}

# ---- serve 32B-int8 on GPU0 (tool-calling: hermes parser·enforce-eager); reuse if already healthy ----
if curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$M"; then
  echo "SERVE_REUSE port=$PORT (already up)"; date
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

# ---- arm② G5-only (precondition-steering 격리) ----
run_arm on_n32int8_g5_retail T2_GATE_KINDS=preconditions
# ---- arm③ G5 + retry-controller ----
run_arm on_n32int8_g5retry_retail T2_GATE_KINDS=preconditions T2_RETRY_CONTROLLER=1 T2_RETRY_K=3

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "FLOW_DISC_MEASURE_DONE"; date
