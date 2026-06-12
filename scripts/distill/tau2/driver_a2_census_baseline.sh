#!/bin/bash
# P-A2-0b baseline census driver (HANDOFF day4 §0.2-①):
# 7B(GPU0:8000) + 14B(GPU1:8001) serve -> t2_a2_size_census airline -> GPU 해제.
# Run: setsid bash driver_a2_census_baseline.sh </dev/null >/dev/null 2>&1 &
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2
OUT=/home/woori/scratch/a2_census
mkdir -p $OUT
exec > $OUT/census_baseline.log 2>&1
cd $REPO && git pull --ff-only

VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
CUDA_VISIBLE_DEVICES=0 $VLLM serve Qwen/Qwen2.5-7B-Instruct --port 8000 \
  --max-model-len 16384 > $OUT/vllm7b.log 2>&1 &
P7=$!
CUDA_VISIBLE_DEVICES=1 $VLLM serve Qwen/Qwen2.5-14B-Instruct --port 8001 \
  --max-model-len 16384 > $OUT/vllm14b.log 2>&1 &
P14=$!

ok=0
for port in 8000 8001; do
  up=0
  for i in $(seq 1 120); do
    curl -s http://localhost:$port/v1/models | grep -q '"data"' && { up=1; break; }
    sleep 5
  done
  echo "[driver] port $port up=$up"
  ok=$((ok + up))
done

if [ "$ok" -eq 2 ]; then
  /home/woori/venvs/seka_env/bin/python $T2/t2_a2_size_census.py \
    --target airline --ref_dir $T2/specs \
    --policy $T2/specs/airline_policy.md \
    --catalog $T2/specs/airline_tool_catalog.json \
    --model qwen7b:http://localhost:8000/v1:Qwen/Qwen2.5-7B-Instruct \
    --model qwen14b:http://localhost:8001/v1:Qwen/Qwen2.5-14B-Instruct \
    --out $OUT/baseline
  echo CENSUS_BASELINE_DONE
else
  echo CENSUS_BASELINE_SERVE_FAIL
fi

kill $P7 $P14 2>/dev/null
sleep 15
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  kill -9 "$pid" 2>/dev/null
done
echo GPU_RELEASED
