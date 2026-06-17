#!/bin/bash
# B(L,width) sweep at scale — width-twin of depth_scale_batch. Serve an open-weight model and run
# width_eval (arm A in-head + arm B set-extraction) on controlled-width synthetic substitute, to test
# whether multi-attr delta extraction is a budget WALL that persists at scale (=> width-offload by
# decomposition is necessary) or dissolves with size. FROZEN: gen seed + prompts identical across sizes.
#   Usage: bash width_scale_batch.sh <MODEL> <TAG> <GPUS> [PORT] [EXTRA_VLLM_ARGS]
#     GPUS = comma list (tensor-parallel = count). Env-overridable: REPO PY VLLM SCRATCH.
set -u
MODEL="${1:?hf model}"; TAG="${2:?tag}"; GPUS="${3:-0}"; PORT="${4:-8065}"; EXTRA="${5:-}"
REPO=${REPO:-/home/woori/workspace_common/boltzmann-attention-pi}
PY=${PY:-/home/woori/venvs/seka_env/bin/python}
VLLM=${VLLM:-/home/woori/venvs/tau2_vllm_env/bin/vllm}
SCRATCH=${SCRATCH:-/home/woori/scratch}
MA=$REPO/scripts/distill/ma; OUT=$SCRATCH/depth/c8/width; mkdir -p $OUT
TP=$(echo $GPUS | awk -F, '{print NF}'); G0=$(echo $GPUS | cut -d, -f1)
LOG=$OUT/width_${TAG}.log; exec > $LOG 2>&1; set -x; date; echo "MODEL=$MODEL TAG=$TAG GPUS=$GPUS TP=$TP"
cd $REPO && git pull --ff-only 2>&1 | tail -1

for p in $(nvidia-smi --id=$G0 --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPUS setsid nohup $VLLM serve "$MODEL" --port $PORT --max-model-len 16384 \
  --tensor-parallel-size $TP $EXTRA > $SCRATCH/vllm_width_${TAG}.log 2>&1 &
ok=0; for i in $(seq 1 150); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q '"id"' && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -40 $SCRATCH/vllm_width_${TAG}.log; exit 1; }

# gloss=1 (give the operator definition) so failure = genuine width-binding, not op-recognition.
$PY $MA/width_eval.py --base http://localhost:$PORT/v1 --model "$MODEL" --tag $TAG \
  --widths 1,2,3,4,5 --n 100 --arms A,B --gloss 1 --out $OUT/width_${TAG}.json
for p in $(nvidia-smi --id=$G0 --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "WIDTH_${TAG}_DONE"; date
