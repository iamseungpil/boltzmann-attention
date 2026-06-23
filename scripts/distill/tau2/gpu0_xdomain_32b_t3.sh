#!/bin/bash
# 6hr 무인 (2026-06-24 외출·GPU0): 32B-int8 robust(nt=3+) 결측 셀 보강.
# 목적 = TCO scale×domain 표(북극성 #6)의 32B cross-domain FLOOR robust 셀 채움.
#  - 32B floor airline/banking nt=3  (현재 nt=1만 존재 = 결측 robust)
#  - 32B floor retail nt=5           (flagship CI 좁히기·NOW "pass^1 CI~0.06→<0.05 다수trial 필요")
# 전부 floor(--gate 0)=scaffold 없음·도메인분기0·tau2 학습0([[05]][[11]]). user-sim=gpt-4.1(COST GUARD).
# 회수(아침): morning_tables.py / tco_table.py + tco_cost.py. 마커=GPU0_XDOMAIN_32B_DONE.
set -u
GPU=0; PORT=8360
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch; TB=/home/woori/scratch/tau2-bench
M=Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8
LOG=$S/gpu0_xdomain_32b_t3.log
exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --ff-only
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2

run () {  # $1=save $2=domain $3=ntrials
  local save=$1 dom=$2 nt=$3
  echo "######## RUN $save dom=$dom nt=$nt ########"; date
  cd $TB; rm -rf "$TB/data/simulations/$save"
  PYTHONPATH=src:$T2 $PY $T2/t2_run_gated.py --gate 0 --domain "$dom" \
    --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
    --num_trials $nt --max_concurrency 8 --save_to "$save" || echo "ARM_FAIL $save"
  echo "ARM_DONE $save"; date
}

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve "$M" --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-model-len 16384 --enforce-eager --gpu-memory-utilization 0.92 \
  > $S/vllm_gpu0_xdomain.log 2>&1 &
ok=0; for i in $(seq 1 150); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$M" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo "SERVE_FAIL"; tail -40 $S/vllm_gpu0_xdomain.log; exit 1; }
echo "SERVE_OK"; date

# 결측 robust 셀 우선
run ours_n32int8_floor_airline_t3 airline           3
run ours_n32int8_floor_bank_t3    banking_knowledge 3
# flagship CI 좁히기
run on_n32int8_floor_retail_t5    retail            5

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "GPU0_XDOMAIN_32B_DONE"; date
