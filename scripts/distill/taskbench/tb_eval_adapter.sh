#!/bin/bash
# Exp-A adapter eval (HANDOFF_2026_06_10 §4): serve base+LoRA on chosen GPU, run held-out
# full inference + in-domain 500-subset sanity, then eval. Usage:
#   tb_eval_adapter.sh lodo_mm data_multimedia "data_huggingface data_dailylifeapis" [GPU]
NAME=$1; HOLDOUT=$2; TRAIN_DOMS=$3; GPU=${4:-0}; UTIL=${5:-0.85}; PORT=$((8000+GPU))
TB=/home/woori/scratch/JARVIS_tb/taskbench
R=/home/woori/workspace_common/boltzmann-attention-pi
RUNS=$R/reports/facet_rft_2026/phase4_distill/sft_runs
# env overrides for non-7B adapters: BASEM (HF base model id), ADAPTER (adapter dir)
BASEM=${BASEM:-Qwen/Qwen2.5-7B-Instruct}
ADAPTER=${ADAPTER:-$RUNS/qwen7b_tb_${NAME}}
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
IP=/home/woori/scratch/tbeval_venv/bin/python
TAG=tb_${NAME}
LOG=/home/woori/scratch/tb_evaladapter_${NAME}.log
exec > $LOG 2>&1
set -x

kill_gpu_vllm() {
  # only kill vLLM serve processes on $GPU (never trainers)
  for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid,process_name --format=csv,noheader | grep -i vllm | cut -d, -f1); do kill -9 $p; done
  for i in $(seq 1 30); do
    [ -z "$(nvidia-smi --id=$GPU --query-compute-apps=pid,process_name --format=csv,noheader | grep -i vllm)" ] && return 0
    sleep 5
  done
}

dep_of() { case $1 in *dailylife*) echo temporal;; *) echo resource;; esac; }

kill_gpu_vllm
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASEM \
  --port $PORT --served-model-name base_model --enable-lora \
  --lora-modules ${TAG}=$ADAPTER \
  --max-model-len 8192 --gpu-memory-utilization $UTIL \
  > /home/woori/scratch/vllm_${TAG}.log 2>&1 &
for i in $(seq 1 90); do
  curl -s localhost:$PORT/v1/models | grep -q "\"$TAG\"" && break
  sleep 10
done
curl -s localhost:$PORT/v1/models | grep -q "\"$TAG\"" || { echo "SERVE_FAIL_$TAG"; exit 1; }

# held-out: full domain
(cd $TB && $IP inference.py --data_dir $HOLDOUT --api_addr localhost --api_port $PORT \
  --api_key dummy --llm $TAG --multiworker 8 --dependency_type $(dep_of $HOLDOUT))
$IP $R/scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain $HOLDOUT --llm $TAG \
  --dst $TB/${HOLDOUT}_evalfull_${TAG}

# in-domain sanity: 500-subsets (created by tb_scale_curve.sh; create if absent)
for d in $TRAIN_DOMS; do
  s=$TB/${d}_sub500
  if [ ! -f $s/user_requests.json ]; then
    mkdir -p $s; head -500 $TB/$d/user_requests.json > $s/user_requests.json
    cp $TB/$d/tool_desc.json $TB/$d/graph_desc.json $s/
  fi
  (cd $TB && $IP inference.py --data_dir ${d}_sub500 --api_addr localhost --api_port $PORT \
    --api_key dummy --llm $TAG --multiworker 8 --dependency_type $(dep_of $d))
  $IP $R/scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain $d \
    --pred_file $TB/${d}_sub500/predictions/${TAG}.json \
    --dst $TB/${d}_sub500_eval_${TAG} --llm $TAG
done
echo "ADAPTER_EVAL_DONE_${NAME} $(date)"
