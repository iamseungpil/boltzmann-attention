#!/bin/bash
# 재실험2 (2026-06-25): 잔여분석 기반 *학습없는* 수정 = present + g15(precondition 포함).
# 잔여 최대 = operator 틀림(24/16)·over-action(14/12) → precondition 게이트(status→operator·status-lock)
# 가 닫을 영역. present 단독·g14present는 precondition *없었음* → 이번이 그 조합 첫 측정.
# ★측정 = 결정론(order/operator/over-action correct)·[[06]] "G5=0" 결정론 재검. pass는 pass^k 보조.
# 사용법: reexp_present_g15.sh <GPU> <PORT> <MODEL> <TAG>
set -u
GPU=$1; PORT=$2; M=$3; TAG=$4
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm; S=/home/woori/scratch; TB=/home/woori/scratch/tau2-bench
LOG=$S/reexp_g15_$TAG.log
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
  > $S/vllm_g15_$TAG.log 2>&1 &
ok=0; for i in $(seq 1 150); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$M" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo "SERVE_FAIL"; tail -40 $S/vllm_g15_$TAG.log; exit 1; }
echo "SERVE_OK"; date
run ${TAG}_presentg15_retail_t3  T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions T2_PRESENT_READS=1
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "REEXP_G15_${TAG}_DONE"; date
