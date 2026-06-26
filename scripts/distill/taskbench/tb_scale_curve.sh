#!/bin/bash
# Exp-C scale curve (HANDOFF_2026_06_10 §4): Qwen2.5-{0.5,1.5,3,14}B on 500-subset/domain.
# Sequentially serves each model on GPU0 (kills existing GPU0 vllm by PID), runs 3-domain
# subset inference + eval. Usage: tb_scale_curve.sh "0.5 1.5 3 14"
SIZES=${1:-"0.5 1.5 3 14"}
TB=/home/woori/scratch/JARVIS_tb/taskbench
R=/home/woori/workspace_common/boltzmann-attention-pi
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
IP=/home/woori/scratch/tbeval_venv/bin/python
LOG=/home/woori/scratch/tb_scale_curve.log
exec > $LOG 2>&1
set -x

# idempotent 500-subsets
for d in data_huggingface data_multimedia data_dailylifeapis; do
  s=$TB/${d}_sub500
  if [ ! -f $s/user_requests.json ]; then
    mkdir -p $s
    head -500 $TB/$d/user_requests.json > $s/user_requests.json
    cp $TB/$d/tool_desc.json $TB/$d/graph_desc.json $s/
  fi
done

kill_gpu0_vllm() {
  for p in $(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done
  for i in $(seq 1 30); do
    [ -z "$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader)" ] && return 0
    sleep 5
  done
}

for sz in $SIZES; do
  tag=qwen$(echo $sz | tr -d .)b
  kill_gpu0_vllm
  CUDA_VISIBLE_DEVICES=0 setsid nohup $VLLM serve Qwen/Qwen2.5-${sz}B-Instruct \
    --port 8000 --served-model-name $tag --max-model-len 8192 --gpu-memory-utilization 0.85 \
    > /home/woori/scratch/vllm_${tag}.log 2>&1 &
  for i in $(seq 1 60); do
    curl -s localhost:8000/v1/models | grep -q "\"$tag\"" && break
    sleep 10
  done
  curl -s localhost:8000/v1/models | grep -q "\"$tag\"" || { echo "SERVE_FAIL_$tag"; continue; }
  for spec in "data_huggingface resource" "data_multimedia resource" "data_dailylifeapis temporal"; do
    set -- $spec
    (cd $TB && $IP inference.py --data_dir ${1}_sub500 --api_addr localhost --api_port 8000 \
      --api_key dummy --llm $tag --multiworker 8 --dependency_type $2)
    $IP $R/scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain $1 \
      --pred_file $TB/${1}_sub500/predictions/${tag}.json \
      --dst $TB/${1}_sub500_eval_${tag} --llm $tag
  done
  echo "MODEL_DONE_$tag $(date)"
done
kill_gpu0_vllm
echo "SCALE_CURVE_ALL_DONE $(date)"
