#!/bin/bash
# E1: base(무어댑터)+guided, held-out MM full — 2×2 factorial 마지막 셀 (§6.5 ToolDec-대조 완결).
# 사전예측: ≈50.4±0.3 (base+snap 통제와 동급 — 고칠 어휘 1.7%뿐).
# GPU0. log: /home/woori/scratch/tb_guided_base.log, sentinel GUIDED_BASE_DONE.
R=/home/woori/workspace_common/boltzmann-attention-pi
TB=/home/woori/scratch/JARVIS_tb/taskbench
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
IP=/home/woori/scratch/tbeval_venv/bin/python
S=/home/woori/scratch
TAG=qwen7b_guided
GPU=0; PORT=8000
exec > $S/tb_guided_base.log 2>&1
set -x

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid,process_name --format=csv,noheader | grep -i vllm | cut -d, -f1); do kill -9 $p; done
sleep 10

$IP $R/scripts/distill/taskbench/tb_guided_patch.py $TB/inference.py || exit 1
[ -f $S/tb_guided_mm_schema.json ] || $IP $R/scripts/distill/taskbench/tb_guided_schema.py \
  --tool_desc $TB/data_multimedia/tool_desc.json --dep resource --out $S/tb_guided_mm_schema.json

CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
  --port $PORT --served-model-name ${TAG} \
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
  \"structured_outputs\": {\"json\": $(cat $S/tb_guided_mm_schema.json)}}" | head -c 200
echo

(cd $TB && TB_GUIDED=1 TB_GUIDED_SCHEMA=$S/tb_guided_mm_schema.json \
  $IP inference.py --data_dir data_multimedia --api_addr localhost --api_port $PORT \
  --api_key dummy --llm $TAG --multiworker 8 --dependency_type resource)
$IP $R/scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain data_multimedia \
  --llm $TAG --dst $TB/data_multimedia_evalfull_${TAG}

echo "GUIDED_BASE_DONE $(date)"
