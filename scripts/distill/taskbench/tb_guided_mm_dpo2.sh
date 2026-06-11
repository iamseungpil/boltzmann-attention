#!/bin/bash
# best-stack 합성: dpo2 어댑터 + guided decoding, held-out MM full.
# 대조군: dpo2 unguided 55.95 / dpo2+snap 57.30 (§9.6) — guided가 snap 상위호환인지.
# GPU0. log: /home/woori/scratch/tb_guided_mm_dpo2.log, sentinel GUIDED_MM_DONE.
R=/home/woori/workspace_common/boltzmann-attention-pi
TB=/home/woori/scratch/JARVIS_tb/taskbench
RUNS=$R/reports/facet_rft_2026/phase4_distill/sft_runs
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
IP=/home/woori/scratch/tbeval_venv/bin/python
S=/home/woori/scratch
TAG=tb_dpo2_mm_guided
GPU=0; PORT=8000
exec > $S/tb_guided_mm_dpo2.log 2>&1
set -x

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid,process_name --format=csv,noheader | grep -i vllm | cut -d, -f1); do kill -9 $p; done
sleep 10

$IP $R/scripts/distill/taskbench/tb_guided_patch.py $TB/inference.py || exit 1
$IP $R/scripts/distill/taskbench/tb_guided_schema.py \
  --tool_desc $TB/data_multimedia/tool_desc.json --dep resource \
  --out $S/tb_guided_mm_schema.json || exit 1

CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
  --port $PORT --served-model-name base_model --enable-lora \
  --lora-modules ${TAG}=$RUNS/qwen7b_tb_dpo2_mm \
  --max-model-len 8192 --gpu-memory-utilization 0.85 \
  > $S/vllm_${TAG}.log 2>&1 &
for i in $(seq 1 90); do
  curl -s localhost:$PORT/v1/models | grep -q "\"$TAG\"" && break
  sleep 10
done
curl -s localhost:$PORT/v1/models | grep -q "\"$TAG\"" || { echo "SERVE_FAIL_$TAG"; exit 1; }

curl -s -m 120 localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "{
  \"model\": \"$TAG\", \"max_tokens\": 32,
  \"messages\": [{\"role\": \"user\", \"content\": \"emit a tiny plan\"}],
  \"structured_outputs\": {\"json\": $(cat $S/tb_guided_mm_schema.json)}}" | head -c 300
echo

(cd $TB && TB_GUIDED=1 TB_GUIDED_SCHEMA=$S/tb_guided_mm_schema.json \
  $IP inference.py --data_dir data_multimedia --api_addr localhost --api_port $PORT \
  --api_key dummy --llm $TAG --multiworker 8 --dependency_type resource)
$IP $R/scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain data_multimedia \
  --llm $TAG --dst $TB/data_multimedia_evalfull_${TAG}

echo "GUIDED_MM_DONE $(date)"
