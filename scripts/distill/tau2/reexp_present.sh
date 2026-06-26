#!/bin/bash
# 재실험 (2026-06-25): select_confirm replay-break FIX 후 = READ-AUGMENT present (T2_PRESENT_READS=1).
# 앞 run(deny 방식)=infra_error 283/342로 무효 → 이번=replay-safe 읽기증강. 32B(GPU0)+14B(GPU1).
#   presentread (auth + present) vs floor · g14present (g14 + present) vs g14.
# ★검증의무([[08]]): termination_reason에 infrastructure_error 0 확인해야 측정 유효.
# 사용법: reexp_present.sh <GPU> <PORT> <MODEL> <TAG>
set -u
GPU=$1; PORT=$2; M=$3; TAG=$4
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch; TB=/home/woori/scratch/tau2-bench
LOG=$S/reexp_present_$TAG.log
exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --ff-only
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2

run () {  # $1=save $2..=env
  local save=$1; shift
  echo "######## RUN $save env=$* ########"; date
  cd $TB; rm -rf "$TB/data/simulations/$save"
  env "$@" PYTHONPATH=src:$T2 $PY $T2/t2_run_gated.py --gate 1 --domain retail \
    --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
    --num_trials 3 --max_concurrency 8 --save_to "$save" || echo "ARM_FAIL $save"
  echo "ARM_DONE $save"; date
}

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve "$M" --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-model-len 16384 --enforce-eager --gpu-memory-utilization 0.92 \
  > $S/vllm_reexp_$TAG.log 2>&1 &
ok=0; for i in $(seq 1 150); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$M" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo "SERVE_FAIL"; tail -40 $S/vllm_reexp_$TAG.log; exit 1; }
echo "SERVE_OK"; date

run ${TAG}_presentread_retail_t3  T2_GATE_KINDS=auth T2_PRESENT_READS=1
run ${TAG}_g14present_retail_t3   T2_GATE_KINDS=auth,confirm,ownership,notice T2_PRESENT_READS=1

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "REEXP_${TAG}_DONE"; date
