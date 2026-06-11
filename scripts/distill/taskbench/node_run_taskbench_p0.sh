#!/bin/bash
# node_run_taskbench_p0.sh — TB_SCALE v3 P0 prompted curves (EVAL node).
#   Phase A (GPU1,2 TP2 — leaves GPU0 to the SOPBench sanity server, GPU3 spare):
#     P0a Qwen2.5-32B  ->  P0b Qwen3-32B            (sequential on the pair)
#   Phase B (TP4 — waits until the SOPBench sanity driver exits):
#     P0a Qwen2.5-72B  ->  P0b Qwen3-235B-A22B-GPTQ-Int4
# All runs: 500-sub x 3 domains. Qwen3 non-thinking is enforced by the patched
# inference.py payload (tb_patch_inference.py); a serve-time sanity probe re-checks.
# Preemption-safe: inference.py natively skips already-predicted ids (predictions
# live under /scratch/taskbench_runs/preds via symlink -> HF-synced); per-model
# done-markers skip finished models on restart.
set -x
export HF_HUB_CACHE=/scratch/hf_cache
TB=/scratch/JARVIS/taskbench
R=/scratch/boltzmann-attention
VLLM=/scratch/venvs/sop_env/bin/vllm
HF=/scratch/venvs/sop_env/bin/hf
IP=/scratch/venvs/tb_env/bin/python
PORT=${TB_PORT:-8300}
OUT=/scratch/taskbench_runs
DONE=$OUT/p0_done
mkdir -p $DONE /scratch/logs

kill_port_vllm() {
  pkill -9 -f "vllm serve.*--port $PORT"
  for i in $(seq 1 24); do pgrep -f "vllm serve.*--port $PORT" >/dev/null || break; sleep 5; done
  sleep 10
}

run_model() {  # run_model TAG HF_ID GPUS TPN NOTHINK_SANITY(0/1)
  local tag=$1 hfid=$2 gpus=$3 tpn=$4 nothink=${5:-0}
  [ -f $DONE/$tag ] && { echo "SKIP_DONE_$tag"; return 0; }
  $HF download "$hfid" >> /scratch/logs/hfdl_${tag}.log 2>&1
  kill_port_vllm
  CUDA_VISIBLE_DEVICES=$gpus setsid nohup $VLLM serve "$hfid" \
    --port $PORT --served-model-name $tag --tensor-parallel-size $tpn \
    --max-model-len 8192 --gpu-memory-utilization 0.90 \
    > /scratch/logs/vllm_${tag}.log 2>&1 &
  for i in $(seq 1 180); do
    curl -s localhost:$PORT/v1/models | grep -q "\"$tag\"" && break; sleep 10
  done
  curl -s localhost:$PORT/v1/models | grep -q "\"$tag\"" || { echo "SERVE_FAIL_$tag"; kill_port_vllm; return 1; }
  if [ "$nothink" = "1" ]; then
    SAN=$(curl -s localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" \
      -d "{\"model\":\"$tag\",\"messages\":[{\"role\":\"user\",\"content\":\"say ok\"}],\"max_tokens\":50,\"chat_template_kwargs\":{\"enable_thinking\":false}}")
    echo "SANITY_$tag: $(echo "$SAN" | head -c 300)"
    echo "$SAN" | grep -q "<think>" && echo "THINKING_LEAK_$tag"
  fi
  local ok=1
  for spec in "data_huggingface resource" "data_multimedia resource" "data_dailylifeapis temporal"; do
    set -- $spec
    (cd $TB && $IP inference.py --data_dir ${1}_sub500 --api_addr localhost --api_port $PORT \
      --api_key dummy --llm $tag --multiworker 8 --dependency_type $2) || ok=0
    $IP $R/scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain $1 \
      --pred_file $TB/${1}_sub500/predictions/${tag}.json \
      --dst $OUT/${1}_sub500_eval_${tag} --llm $tag || ok=0
  done
  kill_port_vllm
  [ "$ok" = "1" ] && touch $DONE/$tag
  echo "MODEL_DONE_$tag ok=$ok $(date)"
}

# Phase A — 32B class, TP2 on GPU1,2
run_model qwen25_32b Qwen/Qwen2.5-32B-Instruct 1,2 2 0
run_model qwen3_32b  Qwen/Qwen3-32B            1,2 2 1

# Phase B — TP4; wait for the SOPBench sanity driver (GPU0,3 server) to finish,
# then retire its server to free all 4 GPUs
while pgrep -f node_run_sanity32b.sh > /dev/null; do sleep 60; done
pkill -9 -f "vllm serve.*--port 9100"; sleep 15
run_model qwen25_72b Qwen/Qwen2.5-72B-Instruct 0,1,2,3 4 0
run_model qwen3_235b_a22b_int4 Qwen/Qwen3-235B-A22B-GPTQ-Int4 0,1,2,3 4 1

echo "P0_ALL_DONE $(date)"
