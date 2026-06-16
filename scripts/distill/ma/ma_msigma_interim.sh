#!/bin/bash
# Interim M-sigma checkpoint eval (user req: train ~1h -> eval -> training CONTINUES).
# Waits ~55min + resume_adapter present, snapshots it, serves base+LoRA on GPU1:8015,
# runs in-dist $ref-emit eval (base vs M-sigma). Training on GPU0 is NOT touched.
set -u
S=/home/woori/scratch
REPO=/home/woori/workspace_common/boltzmann-attention-pi
MA=$REPO/scripts/distill/ma
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
ADP=$S/sft_runs/qwen7b_msigma/resume_adapter
SNAP=$S/sft_runs/qwen7b_msigma/interim_snap
LOG=$S/msigma_interim.log
exec > $LOG 2>&1; set -x; date

# wait ~55min AND adapter present (cap 80min)
for i in $(seq 1 160); do
  [ -f $ADP/adapter_model.safetensors ] && [ $i -ge 110 ] && break   # ~110*30s=55min
  sleep 30
done
date; echo "snapshot resume_adapter"
rm -rf $SNAP; mkdir -p $SNAP; cp -a $ADP/. $SNAP/ 2>/dev/null; ls $SNAP

# serve base+LoRA on GPU1 (training is GPU0 — do NOT kill GPU0)
for p in $(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=1 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port 8015 \
  --max-model-len 16384 --gpu-memory-utilization 0.92 \
  --enable-lora --lora-modules msigma=$SNAP --max-lora-rank 32 > $S/vllm_msigma_interim.log 2>&1 &
ok=0; for i in $(seq 1 60); do curl -s localhost:8015/v1/models 2>/dev/null | grep -q msigma && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -30 $S/vllm_msigma_interim.log; exit 1; }

echo "===== in-dist \$ref-emit eval: base vs M-sigma ====="
$PY $MA/m_sigma_eval.py --base http://localhost:8015/v1 --model Qwen/Qwen2.5-7B-Instruct --tag base --n 60
$PY $MA/m_sigma_eval.py --base http://localhost:8015/v1 --model msigma --tag msigma --n 60
for p in $(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo MSIGMA_INTERIM_DONE; date
