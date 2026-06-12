#!/bin/bash
# τ² retail 7B base ±게이트 첫 측정 (BENCH_PORTFOLIO §3.6 ③) — v2 구성 (2026-06-12):
#   GPU0 = agent 7B vllm(8351) 전용 / user-sim·judge = OpenRouter(gpt-4.1-mini)
#   (GPU1은 v3 학습 등 타 작업용으로 비움 — 사용자 지시)
# num_trials=4 → pass^1/pass^k. log: /home/woori/scratch/t2_retail.log, sentinel T2_RETAIL_DONE
R=/home/woori/workspace_common/boltzmann-attention-pi
T2=/home/woori/scratch/tau2-bench
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch
exec > $S/t2_retail.log 2>&1
set -x
cd $R && git pull --ff-only -q

source /home/woori/.openrouter_key  # 파일 = `export OPENROUTER_API_KEY=...` 형식
USER_LLM="openrouter/openai/gpt-4.1-mini"

# agent vllm(8351)이 살아있으면 재사용, 아니면 GPU0 정리 후 기동
if curl -s localhost:8351/v1/models | grep -q Qwen; then
  echo "REUSING_AGENT_VLLM"
else
  for p in $(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader); do
    kill -9 $p 2>/dev/null; done
  sleep 10
  CUDA_VISIBLE_DEVICES=0 $VLLM serve Qwen/Qwen2.5-7B-Instruct --port 8351 \
    --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
    > $S/vllm_t2_agent.log 2>&1 &
  ok=0
  for i in $(seq 1 120); do
    curl -s localhost:8351/v1/models | grep -q Qwen && ok=1 && break
    sleep 10
  done
  [ $ok = 1 ] || { echo SERVE_FAIL_8351; exit 1; }
fi

cd $T2
export PYTHONPATH=src:$R/scripts/distill/tau2
$PY $R/scripts/distill/tau2/t2_run_gated.py --gate 0 --num_trials 4 \
  --user_llm "$USER_LLM" --save_to retail_7b_nogate
$PY $R/scripts/distill/tau2/t2_run_gated.py --gate 1 --num_trials 4 \
  --user_llm "$USER_LLM" --save_to retail_7b_gate

echo "T2_RETAIL_DONE $(date)"
