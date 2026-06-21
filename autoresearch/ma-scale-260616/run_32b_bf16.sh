#!/bin/bash
set -u
MA=/home/v-seungplee/boltzmann-facet/scripts/distill/ma
TAU=/home/v-seungplee/tau2-bench/data/tau2/domains/retail
PY=/home/v-seungplee/miniconda3/envs/metavllm/bin/python
VLLM=/home/v-seungplee/miniconda3/envs/metavllm/bin/vllm
HFCLI=/home/v-seungplee/miniconda3/envs/metavllm/bin/hf
WORK=/home/v-seungplee/boltzmann-facet/autoresearch/ma-scale-260616
M=Qwen/Qwen2.5-32B-Instruct
PORT=8021
TAG=Qwen2_5_32B_Instruct
echo "[$(date +%T)] STAGE1 download $M"
$HFCLI download $M > $WORK/dl_32b.log 2>&1 && echo "download OK" || { echo "DOWNLOAD_FAIL"; tail -5 $WORK/dl_32b.log; exit 1; }
echo "[$(date +%T)] STAGE2 serve 32B-bf16 TP1"
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done; sleep 3
CUDA_VISIBLE_DEVICES=0 setsid nohup $VLLM serve $M --port $PORT --dtype bfloat16 \
  --max-model-len 8192 --gpu-memory-utilization 0.95 > $WORK/vllm_32b.log 2>&1 &
ok=0; for i in $(seq 1 90); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "32B" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo "SERVE_FAIL"; tail -25 $WORK/vllm_32b.log; exit 1; }
echo "[$(date +%T)] serve ready"
echo "[$(date +%T)] STAGE3 eval 29x7"
PYTHONPATH=$MA $PY $MA/ma_eval.py --cases $WORK/ma_eval_cases.jsonl --base http://localhost:$PORT/v1 \
  --model $M --arms A,Bfair,L0,L1,L2a,L2b,L3 --out $WORK/ma_eval_${TAG}_bf16.jsonl > $WORK/eval_32b.log 2>&1
echo "[$(date +%T)] eval done rc=$?"
grep -A20 "=== SUMMARY ===" $WORK/eval_32b.log | tail -12
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done
echo "[$(date +%T)] MA_32B_DONE"
