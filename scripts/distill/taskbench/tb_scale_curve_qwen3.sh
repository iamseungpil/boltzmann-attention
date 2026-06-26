#!/bin/bash
# Qwen3 scale curve (family switch, COWORKER_REQUEST_TB_SCALE.md v2 — Track A side).
# Same protocol as tb_scale_curve.sh but: Qwen3 models, GPU/PORT parameterized,
# non-thinking enforced via inference.py chat_template_kwargs patch (applied 2026-06-10).
# Usage: tb_scale_curve_qwen3.sh GPU PORT "0.6 1.7 14 4"
GPU=${1:-1}; PORT=${2:-8001}; SIZES=${3:-"0.6 1.7 14 4"}
TB=/home/woori/scratch/JARVIS_tb/taskbench
R=/home/woori/workspace_common/boltzmann-attention-pi
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
IP=/home/woori/scratch/tbeval_venv/bin/python
LOG=/home/woori/scratch/tb_scale_curve_qwen3.log
exec > $LOG 2>&1
set -x

kill_gpu_vllm() {
  for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid,process_name --format=csv,noheader | grep -i vllm | cut -d, -f1); do kill -9 $p; done
  for i in $(seq 1 30); do
    [ -z "$(nvidia-smi --id=$GPU --query-compute-apps=pid,process_name --format=csv,noheader | grep -i vllm)" ] && return 0
    sleep 5
  done
}

for sz in $SIZES; do
  tag=qwen3_$(echo $sz | tr -d .)b
  kill_gpu_vllm
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve Qwen/Qwen3-${sz}B \
    --port $PORT --served-model-name $tag --max-model-len 8192 --gpu-memory-utilization 0.85 \
    > /home/woori/scratch/vllm_${tag}.log 2>&1 &
  for i in $(seq 1 90); do
    curl -s localhost:$PORT/v1/models | grep -q "\"$tag\"" && break
    sleep 10
  done
  curl -s localhost:$PORT/v1/models | grep -q "\"$tag\"" || { echo "SERVE_FAIL_$tag"; continue; }
  # non-thinking sanity: response must not contain <think>
  SAN=$(curl -s localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$tag\",\"messages\":[{\"role\":\"user\",\"content\":\"say ok\"}],\"max_tokens\":50,\"chat_template_kwargs\":{\"enable_thinking\":false}}")
  echo "SANITY_$tag: $(echo $SAN | head -c 300)"
  echo "$SAN" | grep -q "<think>" && { echo "THINKING_LEAK_$tag"; }
  for spec in "data_huggingface resource" "data_multimedia resource" "data_dailylifeapis temporal"; do
    set -- $spec
    (cd $TB && $IP inference.py --data_dir ${1}_sub500 --api_addr localhost --api_port $PORT \
      --api_key dummy --llm $tag --multiworker 8 --dependency_type $2)
    $IP $R/scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain $1 \
      --pred_file $TB/${1}_sub500/predictions/${tag}.json \
      --dst $TB/${1}_sub500_eval_${tag} --llm $tag
  done
  echo "MODEL_DONE_$tag $(date)"
done
kill_gpu_vllm
echo "Q3_SCALE_CURVE_ALL_DONE $(date)"
