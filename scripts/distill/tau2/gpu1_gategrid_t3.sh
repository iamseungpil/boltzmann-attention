#!/bin/bash
# 6hr 무인 (2026-06-24 외출·GPU1): 7B/14B retail gate 결측 robust(nt=3) 셀 보강.
# 목적 = scale×gate retail 표를 nt=3로 완성. 현재 7B/14B는 g15 t3만 존재 → g14·g15retry 결측.
#  완성 후: {7,14,32}B × {g14,g15,g15retry} 전부 nt=3 (32B는 on_n32int8_*_t3 기존).
# gate=A2-driven scaffold(KINDS, 도메인분기0·grep if-domain=0·[[05]])·tau2 학습0([[11]]).
# user-sim=gpt-4.1(COST GUARD). 회수: morning_tables.py. 마커=GPU1_GATEGRID_DONE.
set -u
GPU=1; PORT=8361
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch; TB=/home/woori/scratch/tau2-bench
NT=3
LOG=$S/gpu1_gategrid_t3.log
exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --ff-only
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
MODELS=("n7b:Qwen/Qwen2.5-7B-Instruct" "n14b:Qwen/Qwen2.5-14B-Instruct")

run () {  # $1=save $2..=env
  local save=$1; shift
  echo "######## RUN $save env=$* ########"; date
  cd $TB; rm -rf "$TB/data/simulations/$save"
  env "$@" PYTHONPATH=src:$T2 $PY $T2/t2_run_gated.py --gate 1 --domain retail \
    --agent_model "$CKPT" --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
    --num_trials $NT --max_concurrency 8 --save_to "$save" || echo "ARM_FAIL $save"
  echo "ARM_DONE $save"; date
}

for entry in "${MODELS[@]}"; do
  TAG=${entry%%:*}; CKPT=${entry#*:}
  echo "============ MODEL $TAG ============"; date
  for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve "$CKPT" --port $PORT \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len 16384 --enforce-eager --gpu-memory-utilization 0.92 \
    > $S/vllm_gpu1_gategrid_$TAG.log 2>&1 &
  ok=0; for i in $(seq 1 150); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$CKPT" && ok=1 && break; sleep 10; done
  [ $ok = 1 ] || { echo "SERVE_FAIL $TAG"; tail -40 $S/vllm_gpu1_gategrid_$TAG.log; continue; }
  echo "SERVE_OK $TAG"; date
  run ours_${TAG}_g14_retail_t3      T2_GATE_KINDS=auth,confirm,ownership,notice
  run ours_${TAG}_g15retry_retail_t3 T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions T2_RETRY_CONTROLLER=1 T2_RETRY_K=3
  for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 3
done
echo "GPU1_GATEGRID_DONE"; date
