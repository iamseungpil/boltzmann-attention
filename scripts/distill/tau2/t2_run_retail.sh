#!/bin/bash
# τ² retail 7B base ±게이트 첫 측정 (BENCH_PORTFOLIO §3.6 ③) — day 배치 종료 후 발사.
# GPU0=agent 7B(8351) / GPU1=user-sim 32B-Int8(8352). num_trials=4 → pass^1/pass^k.
# log: /home/woori/scratch/t2_retail.log, sentinel T2_RETAIL_DONE
R=/home/woori/workspace_common/boltzmann-attention-pi
T2=/home/woori/scratch/tau2-bench
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch
exec > $S/t2_retail.log 2>&1
set -x
cd $R && git pull --ff-only -q

# 양 포트가 이미 건강하면 serve 재사용 (재시작 경로), 아니면 사전-kill 후 기동
if curl -s localhost:8351/v1/models | grep -q Qwen && curl -s localhost:8352/v1/models | grep -q Qwen; then
  echo "REUSING_LIVE_VLLMS"
else
for g in 0 1; do
  for p in $(nvidia-smi --id=$g --query-compute-apps=pid --format=csv,noheader); do
    kill -9 $p 2>/dev/null; done; done
sleep 10

CUDA_VISIBLE_DEVICES=0 $VLLM serve Qwen/Qwen2.5-7B-Instruct --port 8351 \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
  > $S/vllm_t2_agent.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $VLLM serve Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --port 8352 \
  --max-model-len 16384 > $S/vllm_t2_user.log 2>&1 &
fi

for port in 8351 8352; do
  ok=0
  for i in $(seq 1 120); do
    curl -s localhost:$port/v1/models | grep -q Qwen && ok=1 && break
    sleep 10
  done
  [ $ok = 1 ] || { echo SERVE_FAIL_$port; exit 1; }
done

cd $T2
export PYTHONPATH=src:$R/scripts/distill/tau2
$PY $R/scripts/distill/tau2/t2_run_gated.py --gate 0 --num_trials 4 --save_to retail_7b_nogate
$PY $R/scripts/distill/tau2/t2_run_gated.py --gate 1 --num_trials 4 --save_to retail_7b_gate

echo "T2_RETAIL_DONE $(date)"
