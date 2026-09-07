#!/usr/bin/env bash
# 우리-팔 1태스크 런 (F1 검증). 사용: ours.sh <TAG> <PORT> <TASK> <TRIALS> <CONC>
set -u
TAG="$1"; PORT="$2"; TASK="$3"; TR="${4:-4}"; CC="${5:-2}"
cd /home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2 || exit 1
source ./go_stack.sh >/dev/null 2>&1          # ⛔[[19]]/[[60]] 레버 전부 ON
source ./arms/viewmax2.env >/dev/null 2>&1    # 팔
source ~/.openrouter_key; [ -f ~/.openai_key ] && source ~/.openai_key
export HF_HOME=/workspace/.hf_home
export T2_MAX_MODEL_LEN=131072
export T2_FB_SIDECAR=/root/logs/fb_${TAG}.jsonl
export T2_FB_SIDECAR_TEXT=1                    # ★[[81]] 문면 확인용
export PYTHONPATH=src:/root/t2/tau2
cd /root/tau2-bench || exit 1
rm -rf data/simulations/$TAG
/root/tau2-bench/venv/bin/python -u /root/t2/tau2/t2_run_gated.py \
  --domain banking_knowledge --gate 1 --retrieval_config alltools \
  --agent_model Qwen/Qwen3.8-27B-FP8 --agent_base "http://localhost:$PORT/v1" \
  --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 --user_reasoning_effort low \
  --task_ids "$TASK" --num_trials "$TR" --max_concurrency "$CC" --max_steps 200 \
  --save_to "$TAG" > /root/logs/${TAG}.log 2>&1
echo "OURS_DONE rc=$?"
