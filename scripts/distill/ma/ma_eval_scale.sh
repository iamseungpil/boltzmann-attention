#!/bin/bash
# M-A elicitation x scale sweep: for each model, serve on GPU0 and run all 5 arms
# (A/Acot/B/Bcot/C). Rules out the forced-JSON reasoning-suppression confound (CoT arms)
# and gives the first scale-gradient point (7B vs 14B). GPU0 only; never touch GPU1.
set -u
GPU="${4:-0}"; PORT="${5:-8013}"   # 4th/5th args: run on a chosen GPU/port (concurrency)
REPO=/home/woori/workspace_common/boltzmann-attention-pi
MA=$REPO/scripts/distill/ma
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
LOG=$S/ma_eval_scale_p${PORT}.log
exec > $LOG 2>&1; set -x; date

MODELS="${1:-Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct}"
ARMS="${2:-A,Acot,Atwo,B,Bcot,Btwo,C}"
SUFFIX="${3:-}"   # output -> ma_eval_<tag><SUFFIX>.jsonl (avoid clobbering prior runs)
cd $REPO && git pull --ff-only
$PY $MA/ma_gold_extract.py --out $S/ma_eval_cases.jsonl

for M in $MODELS; do
  tag=$(echo $M | sed 's#.*/##; s/[^A-Za-z0-9]/_/g')
  echo "######## MODEL $M (tag=$tag) ########"; date
  for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $M --port $PORT \
    --max-model-len 16384 --gpu-memory-utilization 0.92 > $S/vllm_ma_$tag.log 2>&1 &
  ok=0; for i in $(seq 1 90); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$M" && ok=1 && break; sleep 10; done
  [ $ok = 1 ] || { echo "SERVE_FAIL $M"; tail -30 $S/vllm_ma_$tag.log; continue; }
  $PY $MA/ma_eval.py --cases $S/ma_eval_cases.jsonl --base http://localhost:$PORT/v1 \
    --model $M --arms $ARMS --out $S/ma_eval_${tag}${SUFFIX}.jsonl
  for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 3
done
echo MA_SCALE_DONE; date
