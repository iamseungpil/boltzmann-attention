#!/bin/bash
# τ² retail eval for a LoRA adapter snapshot (native-FC, gated compliant-pass, without-L2).
# 어댑터를 주어진 GPU/포트에 serve → t2_run_gated 실행 → pass^1 보고 → GPU 해제.
# 전송은 git pull (이 파일은 repo). 출력/스냅샷은 리모트 scratch.
#
# Usage: tau2_eval_adapter.sh <adapter_dir> <model_name> <gpu_id> <port> <save_to> [num_tasks]
#   예: bash scripts/distill/tau2/tau2_eval_adapter.sh \
#         /home/woori/scratch/v4_final_adapter v4f 0 8366 retail_v4f 20
set -u
ADAPTER=${1:?adapter dir}; NAME=${2:?model name}; GPU=${3:?gpu id}; PORT=${4:?port}; SAVE=${5:?save_to}; NT=${6:-20}
R=/home/woori/workspace_common/boltzmann-attention-pi
T2=$R/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
LOG=$S/eval_${NAME}.log
exec > $LOG 2>&1; set -x; date

source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")

# serve adapter (GPU 지정·named lora=$NAME)
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
  --enable-lora --lora-modules ${NAME}=$ADAPTER --max-lora-rank 32 > $S/vllm_${NAME}.log 2>&1 &
ok=0; for i in $(seq 1 60); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$NAME" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -25 $S/vllm_${NAME}.log; exit 1; }

# τ² retail gated eval (without-L2; user-sim = gpt-4.1 via openrouter)
cd $S/tau2-bench; export PYTHONPATH=src:$T2
rm -rf data/simulations/$SAVE
$PY $T2/t2_run_gated.py --gate 1 --num_trials 1 --num_tasks $NT --agent_model $NAME \
  --agent_base http://localhost:$PORT/v1 --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 \
  --save_to $SAVE || echo FAIL_TEST
cd $R

# free GPU (이 GPU의 compute proc = 우리 vllm만)
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done

echo "=== $NAME RESULTS (τ² retail without-L2, n=$NT) ==="
echo -n "$SAVE pass^1 "
$PY -c "import json;print(json.load(open('$S/tau2-bench/data/simulations/$SAVE/compliance.json'))['bench']['pass^1'])" 2>/dev/null || echo NA
echo "(base 0.17 / v4 step150 0.10 / v4b step1200 0.11 / frontier 0.81)"
date; echo EVAL_DONE_$NAME
