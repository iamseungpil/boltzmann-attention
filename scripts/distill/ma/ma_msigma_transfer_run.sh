#!/bin/bash
# M-D transfer eval (C8): serve base 7B + M-sigma LoRA (tool-call parsing), run the tau2
# exchange transfer eval (per-arg-type: threading vs selection) for base vs M-sigma.
set -u
GPU="${1:-0}"; PORT=8015
S=/home/woori/scratch
MA=/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/ma
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
ADP=$S/sft_runs/qwen7b_msigma
LOG=$S/msigma_transfer.log
exec > $LOG 2>&1; set -x; date
cd /home/woori/workspace_common/boltzmann-attention-pi && git pull --ff-only
$PY $MA/ma_gold_extract.py --out $S/ma_eval_cases.jsonl

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
  --enable-lora --lora-modules msigma=$ADP --max-lora-rank 32 > $S/vllm_msigma_transfer.log 2>&1 &
ok=0; for i in $(seq 1 60); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q msigma && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -30 $S/vllm_msigma_transfer.log; exit 1; }

echo "===== M-D transfer (tau2 exchange): base vs M-sigma, per-arg-type ====="
$PY $MA/m_sigma_transfer_eval.py --base http://localhost:$PORT/v1 --model Qwen/Qwen2.5-7B-Instruct --tag base
$PY $MA/m_sigma_transfer_eval.py --base http://localhost:$PORT/v1 --model msigma --tag msigma
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo MSIGMA_TRANSFER_DONE; date
