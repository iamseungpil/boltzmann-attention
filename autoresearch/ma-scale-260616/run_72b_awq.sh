#!/bin/bash
set -u
MA=/home/v-seungplee/boltzmann-facet/scripts/distill/ma
PY=/home/v-seungplee/miniconda3/envs/metavllm/bin/python
VLLM=/home/v-seungplee/miniconda3/envs/metavllm/bin/vllm
HFCLI=/home/v-seungplee/miniconda3/envs/metavllm/bin/hf
WORK=/home/v-seungplee/boltzmann-facet/autoresearch/ma-scale-260616
M=Qwen/Qwen2.5-72B-Instruct-AWQ
PORT=8023; TAG=Qwen2_5_72B_Instruct_awq_int4
echo "[$(date +%T)] STAGE1 download $M"
$HFCLI download $M > $WORK/dl_72b.log 2>&1 && echo "download OK" || { echo "DOWNLOAD_FAIL"; tail -5 $WORK/dl_72b.log; exit 1; }
echo "[$(date +%T)] STAGE2 serve 72B-AWQ TP1"
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done; sleep 3
CUDA_VISIBLE_DEVICES=0 setsid nohup $VLLM serve $M --port $PORT \
  --max-model-len 8192 --gpu-memory-utilization 0.92 > $WORK/vllm_72b.log 2>&1 &
ok=0; for i in $(seq 1 120); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "72B" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo "SERVE_FAIL"; tail -25 $WORK/vllm_72b.log; exit 1; }
echo "[$(date +%T)] serve ready"
echo "[$(date +%T)] STAGE3 eval 29x7"
PYTHONPATH=$MA $PY $MA/ma_eval.py --cases $WORK/ma_eval_cases.jsonl --base http://localhost:$PORT/v1 \
  --model $M --arms A,Bfair,L0,L1,L2a,L2b,L3 --out $WORK/ma_eval_${TAG}.jsonl > $WORK/eval_72b.log 2>&1
echo "[$(date +%T)] eval done rc=$?"
grep -A20 "=== SUMMARY ===" $WORK/eval_72b.log | tail -12
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done
echo "[$(date +%T)] MA_72B_DONE"
