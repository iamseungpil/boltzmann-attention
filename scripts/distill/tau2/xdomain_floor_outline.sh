#!/bin/bash
# Cross-domain × scale floor outline (사용자 지시: 우리쪽 윤곽 먼저).
# {7B,14B,32B-int8} × {airline, banking_knowledge} × floor(--gate 0). retail floor×scale은 기존(on_n*_floor_retail).
# bench-pass(+compliant; airline/banking은 auth-모델 caveat). gpt-4.1 user-sim. num_trials=1(outline).
set -u
GPU=0; PORT=8360
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
TB=/home/woori/scratch/tau2-bench
NT="${1:-1}"
LOG=$S/xdomain_floor_outline.log
exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --ff-only
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2

# (model_tag : checkpoint) — 7B/14B fp16 단일GPU, 32B int8
MODELS=("n7b:Qwen/Qwen2.5-7B-Instruct" "n14b:Qwen/Qwen2.5-14B-Instruct" "n32int8:Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
DOMAINS=("airline" "banking_knowledge")

run_floor () {  # $1=tag $2=domain
  local tag=$1 dom=$2
  local short=$dom; [ "$dom" = "banking_knowledge" ] && short=bank
  local save=ours_${tag}_floor_${short}
  echo "######## FLOOR $save ($dom) ########"; date
  cd $TB; rm -rf "$TB/data/simulations/$save"
  PYTHONPATH=src:$T2 $PY $T2/t2_run_gated.py --gate 0 --domain "$dom" \
    --agent_model "$CKPT" --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
    --num_trials $NT --max_concurrency 8 --save_to "$save" || echo "ARM_FAIL $save"
  echo "ARM_DONE $save"; date
}

for entry in "${MODELS[@]}"; do
  TAG=${entry%%:*}; CKPT=${entry#*:}
  echo "============ MODEL $TAG ($CKPT) ============"; date
  for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve "$CKPT" --port $PORT \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len 16384 --enforce-eager --gpu-memory-utilization 0.92 \
    > $S/vllm_xdomain_$TAG.log 2>&1 &
  ok=0; for i in $(seq 1 120); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$CKPT" && ok=1 && break; sleep 10; done
  [ $ok = 1 ] || { echo "SERVE_FAIL $TAG"; tail -40 $S/vllm_xdomain_$TAG.log; continue; }
  echo "SERVE_OK $TAG"; date
  for dom in "${DOMAINS[@]}"; do run_floor "$TAG" "$dom"; done
done
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "XDOMAIN_FLOOR_OUTLINE_DONE"; date
