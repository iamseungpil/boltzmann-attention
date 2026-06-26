#!/bin/bash
# 밤샘 GPU0 보강(2026-06-22): flagship 32B-int8 retail flow-discipline nt=3 denoise.
# GPU1 overnight(nt=1 breadth)와 상보 — 3-trial floor(on_n32int8_floor_retail)와 깨끗 비교.
# arms: g14(G1-4)/g15(G1-5)/g15retry(+retry). save=on_n32int8_<arm>_retail_t3 (충돌없음).
set -u
GPU=0; PORT=8360
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch; TB=/home/woori/scratch/tau2-bench
M=Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8
LOG=$S/gpu0_retail_denoise.log
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
  > $S/vllm_gpu0_denoise.log 2>&1 &
ok=0; for i in $(seq 1 150); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$M" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo "SERVE_FAIL"; tail -40 $S/vllm_gpu0_denoise.log; exit 1; }
echo "SERVE_OK"; date

run on_n32int8_g14_retail_t3      T2_GATE_KINDS=auth,confirm,ownership,notice
run on_n32int8_g15_retail_t3      T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions
run on_n32int8_g15retry_retail_t3 T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions T2_RETRY_CONTROLLER=1 T2_RETRY_K=3

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "GPU0_DENOISE_DONE"; date
