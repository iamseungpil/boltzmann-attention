#!/bin/bash
# grounded-copy v1 LIVE: lodo_daily adapter + guided decoding on held-out daily full.
# 대조군 = 기존 unguided data_dailylifeapis_evalfull_tb_lodo_daily (edge 59.6, snap Δ0).
# GPU0 사용. log: /home/woori/scratch/tb_guided_daily.log, sentinel GUIDED_DAILY_DONE.
R=/home/woori/workspace_common/boltzmann-attention-pi
TB=/home/woori/scratch/JARVIS_tb/taskbench
RUNS=$R/reports/facet_rft_2026/phase4_distill/sft_runs
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
IP=/home/woori/scratch/tbeval_venv/bin/python
S=/home/woori/scratch
TAG=tb_lodo_daily_guided
GPU=0; PORT=8000
exec > $S/tb_guided_daily.log 2>&1
set -x

# 0. kill vllm on GPU0 only (trainer 보호)
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid,process_name --format=csv,noheader | grep -i vllm | cut -d, -f1); do kill -9 $p; done
sleep 10

# 1. patch inference.py (idempotent) + schema
$IP $R/scripts/distill/taskbench/tb_guided_patch.py $TB/inference.py || exit 1
$IP $R/scripts/distill/taskbench/tb_guided_schema.py \
  --tool_desc $TB/data_dailylifeapis/tool_desc.json --out $S/tb_guided_daily_schema.json || exit 1

# 2. serve lodo_daily adapter
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
  --port $PORT --served-model-name base_model --enable-lora \
  --lora-modules ${TAG}=$RUNS/qwen7b_tb_lodo_daily \
  --max-model-len 8192 --gpu-memory-utilization 0.85 \
  > $S/vllm_${TAG}.log 2>&1 &
for i in $(seq 1 90); do
  curl -s localhost:$PORT/v1/models | grep -q "\"$TAG\"" && break
  sleep 10
done
curl -s localhost:$PORT/v1/models | grep -q "\"$TAG\"" || { echo "SERVE_FAIL_$TAG"; exit 1; }

# 3. pre-warm grammar compile (1 dummy guided request, 실패해도 진행)
curl -s -m 120 localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "{
  \"model\": \"$TAG\", \"max_tokens\": 32,
  \"messages\": [{\"role\": \"user\", \"content\": \"emit a tiny plan\"}],
  \"structured_outputs\": {\"json\": $(cat $S/tb_guided_daily_schema.json)}}" | head -c 300
echo

# 4. guided inference on held-out daily full + eval
(cd $TB && TB_GUIDED=1 TB_GUIDED_SCHEMA=$S/tb_guided_daily_schema.json \
  $IP inference.py --data_dir data_dailylifeapis --api_addr localhost --api_port $PORT \
  --api_key dummy --llm $TAG --multiworker 8 --dependency_type temporal)
$IP $R/scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain data_dailylifeapis \
  --llm $TAG --dst $TB/data_dailylifeapis_evalfull_${TAG}

echo "GUIDED_DAILY_DONE $(date)"
