#!/bin/bash
# M-A 3-arm eval: serve base Qwen2.5-7B (NO adapter) on GPU0, run ma_eval.py.
# GPU0 only; never touch GPU1. xgrammar via guided_json (arm B).
set -u
GPU=0; PORT=8013
REPO=/home/woori/workspace_common/boltzmann-attention-pi
MA=$REPO/scripts/distill/ma
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
LOG=$S/ma_eval.log
exec > $LOG 2>&1; set -x; date

cd $REPO && git pull --ff-only
$PY $MA/ma_gold_extract.py --out $S/ma_eval_cases.jsonl

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
  > $S/vllm_ma.log 2>&1 &
ok=0; for i in $(seq 1 60); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q Qwen && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -30 $S/vllm_ma.log; exit 1; }

$PY $MA/ma_eval.py --cases $S/ma_eval_cases.jsonl --base http://localhost:$PORT/v1 \
  --model Qwen/Qwen2.5-7B-Instruct --arms A,B,C --out $S/ma_eval_results.jsonl

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo MA_EVAL_DONE; date
