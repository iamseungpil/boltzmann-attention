#!/bin/bash
# resolve_selection wiring smoke (C0 전제 B 검증): base 또는 facet3 LoRA + --resolve, retail 소수.
# 목적 = crash 없이 (a)도구 노출 (b)인터셉트 (c)엔진 grounding 시도가 도나. 측정 아님.
# Usage: t2_resolve_smoke.sh <gpu> <port> [num_tasks] [lora_path(empty=base)]
set -u
GPU=${1:-0}; PORT=${2:-8000}; NT=${3:-3}; LORA=${4:-}
R=/home/woori/workspace_common/boltzmann-attention-pi
T2=$R/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
BASE=Qwen/Qwen2.5-7B-Instruct
TAG=facet3_native_ep1
S=/home/woori/scratch
LOG=$S/t2_resolve_smoke.log
SAVE=resolve_smoke
exec > $LOG 2>&1; set -x; date

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
if [ -n "$LORA" ]; then
  AGENT=$TAG; SAVE=resolve_smoke_facet3
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT \
    --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
    --enable-lora --max-lora-rank 64 --lora-modules ${TAG}=${LORA} > $S/vllm_resolve_smoke.log 2>&1 &
else
  AGENT=$BASE
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT \
    --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 > $S/vllm_resolve_smoke.log 2>&1 &
fi
ok=0; for i in $(seq 1 70); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -qE "${TAG}|Qwen2.5-7B" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -25 $S/vllm_resolve_smoke.log; exit 1; }

set +x; source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
cd $S/tau2-bench; export PYTHONPATH=src:$T2
rm -rf data/simulations/$SAVE
$PY $T2/t2_run_gated.py --resolve 1 --domain retail --num_trials 1 --num_tasks $NT \
  --agent_model "$AGENT" --agent_base http://localhost:$PORT/v1 \
  --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 --save_to $SAVE 2>&1 | tail -35

echo "=== resolve_selection 호출 흔적 (대화서 grep) ==="
grep -o "resolve_selection" $S/tau2-bench/data/simulations/$SAVE/results.json 2>/dev/null | wc -l
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo SMOKE_DONE; date
