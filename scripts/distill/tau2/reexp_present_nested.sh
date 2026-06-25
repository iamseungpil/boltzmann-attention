#!/bin/bash
# 재실험3 (2026-06-25·priority-2): operand-grounding present 확장.
# present+g15는 order-pick(L1)·operator(L0)·over-action을 닫았으나 operand 잔여(L2 item·L3 variant)
# 가 남음(census·task58 wrong-variant). 이 arm = present(orders·L1) + NESTED(items L2/variants L3) + g15.
# nested = get_order_details/get_product_details 읽기응답에 operand choice-set 덧붙임(replay-safe·A2 present_specs).
# ★측정 = 결정론(escape_det_census·L2/L3 census↓ 보나) vs baseline present+g15(presentg15_retail_t3).
# 사용법: reexp_present_nested.sh <GPU> <PORT> <MODEL> <TAG>
set -u
GPU=$1; PORT=$2; M=$3; TAG=$4
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm; S=/home/woori/scratch; TB=/home/woori/scratch/tau2-bench
LOG=$S/reexp_nest_$TAG.log
exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --ff-only
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
run () { local save=$1; shift
  echo "######## RUN $save env=$* ########"; date
  cd $TB; rm -rf "$TB/data/simulations/$save"
  env "$@" PYTHONPATH=src:$T2 $PY $T2/t2_run_gated.py --gate 1 --domain retail \
    --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
    --num_trials 3 --max_concurrency 8 --save_to "$save" || echo "ARM_FAIL $save"
  echo "ARM_DONE $save"; date; }
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve "$M" --port $PORT --enable-auto-tool-choice \
  --tool-call-parser hermes --max-model-len 16384 --enforce-eager --gpu-memory-utilization 0.92 \
  > $S/vllm_nest_$TAG.log 2>&1 &
ok=0; for i in $(seq 1 150); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$M" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo "SERVE_FAIL"; tail -40 $S/vllm_nest_$TAG.log; exit 1; }
echo "SERVE_OK"; date
run ${TAG}_presentnest_g15_retail_t3  T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions T2_PRESENT_READS=1 T2_PRESENT_NESTED=1
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "REEXP_NEST_${TAG}_DONE"; date
