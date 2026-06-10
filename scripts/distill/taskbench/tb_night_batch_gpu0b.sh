#!/bin/bash
# GPU0 follow-on (after tb_night_batch_gpu0.sh finishes): Qwen3-8B FULL 3-domain baseline
# = family-switch counterpart of the Qwen2.5-7B full baseline. Non-thinking via
# inference.py chat_template_kwargs patch.
TB=/home/woori/scratch/JARVIS_tb/taskbench
R=/home/woori/workspace_common/boltzmann-attention-pi
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
IP=/home/woori/scratch/tbeval_venv/bin/python
LOG=/home/woori/scratch/tb_night_gpu0b.log
exec > $LOG 2>&1
set -x

# wait for the GPU0 queue to drain (up to 18h)
for i in $(seq 1 1080); do
  grep -q "NIGHT_BATCH_GPU0_DONE" /home/woori/scratch/tb_night_gpu0.log && break
  sleep 60
done
grep -q "NIGHT_BATCH_GPU0_DONE" /home/woori/scratch/tb_night_gpu0.log || { echo "ERR_GPU0_QUEUE_TIMEOUT"; exit 1; }

for p in $(nvidia-smi --id=0 --query-compute-apps=pid,process_name --format=csv,noheader | grep -i vllm | cut -d, -f1); do kill -9 $p; done
sleep 15
CUDA_VISIBLE_DEVICES=0 setsid nohup $VLLM serve Qwen/Qwen3-8B \
  --port 8000 --served-model-name qwen3_8b --max-model-len 8192 --gpu-memory-utilization 0.85 \
  > /home/woori/scratch/vllm_qwen3_8b_full.log 2>&1 &
for i in $(seq 1 90); do
  curl -s localhost:8000/v1/models | grep -q "qwen3_8b" && break
  sleep 10
done
curl -s localhost:8000/v1/models | grep -q "qwen3_8b" || { echo "SERVE_FAIL_qwen3_8b"; exit 1; }

for spec in "data_huggingface resource" "data_multimedia resource" "data_dailylifeapis temporal"; do
  set -- $spec
  (cd $TB && $IP inference.py --data_dir $1 --api_addr localhost --api_port 8000 \
    --api_key dummy --llm qwen3_8b --multiworker 8 --dependency_type $2)
  $IP $R/scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain $1 --llm qwen3_8b \
    --dst $TB/${1}_evalfull_qwen3_8b
done
for p in $(nvidia-smi --id=0 --query-compute-apps=pid,process_name --format=csv,noheader | grep -i vllm | cut -d, -f1); do kill -9 $p; done
echo "NIGHT_BATCH_GPU0B_DONE $(date)"
