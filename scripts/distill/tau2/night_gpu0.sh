#!/bin/bash
# Overnight GPU0 chain (FREE): after base-8k isolated probe finishes, swap 32B->14B and run
# 14B thinking scale probes (prompted-CoT 900 = Phase A 14B; big 8000 = 14B budget). Isolated·gpt-4.1=0.
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm; S=/home/woori/scratch
GPU=0; PORT=8140; M=Qwen/Qwen2.5-14B-Instruct
LOG=$S/night_gpu0.log
exec > $LOG 2>&1; set -x; date
# 1. wait for base-8k isolated probe (GPU0) to finish (max ~2h)
for i in $(seq 1 240); do [ -f $S/base8k_end ] && break; sleep 30; done
echo "=== base8k done; swap GPU0 -> 14B ==="; date
# 2. free GPU0, serve 14B (no tool parser needed: probe uses plain /chat/completions)
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 5
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve "$M" --port $PORT \
  --max-model-len 32768 --enforce-eager --gpu-memory-utilization 0.92 > $S/vllm_night14b.log 2>&1 &
ok=0; for i in $(seq 1 120); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$M" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo "SERVE_FAIL 14B"; tail -30 $S/vllm_night14b.log; touch $S/night_gpu0_end; exit 1; }
echo "=== 14B SERVE_OK ==="; date
cd $REPO; export PYTHONPATH=src:$T2
# 3a. 14B prompted-CoT (Phase A 14B): cot_max 900, temp 0.0
echo "=== RUN 14B cot900 (Phase A 14B) ==="
$PY -u $T2/testtime_lever_probe.py --agent_base http://localhost:$PORT/v1 --agent_model "$M" \
  --cot_max 900 --temperature 0.0 --req_timeout 240 --save_json $S/ttl_14b_cot900_rows.json
echo "=== 14B cot900 DONE ==="; date
# 3b. 14B big budget: cot_max 8000, temp 0.6
echo "=== RUN 14B big8000 ==="
$PY -u $T2/testtime_lever_probe.py --agent_base http://localhost:$PORT/v1 --agent_model "$M" \
  --cot_max 8000 --temperature 0.6 --req_timeout 600 --save_json $S/ttl_14b_big_rows.json
echo "=== 14B big8000 DONE ==="; date
touch $S/night_gpu0_end
echo "NIGHT_GPU0_DONE"; date
