#!/bin/bash
# Multi-domain content-routing AT SCALE (inference-only, NO training) — coworker lane.
# Serve a big base model (TP) and run the op-IR routing eval on the COMMITTED retail+airline content
# cases at gloss=0 (floor) and gloss=1 (ceiling, operator vocabulary spoon-fed in-context). Answers:
# does domain-general op-routing (substitute vs comparative vs create on BOTH domains) come from SCALE
# + in-context gloss, or does it need the trained routing LoRA (woori 7B §20)? FROZEN: cases + prompts
# identical across sizes (valid scaling). Reads cases from the repo so no tau2-bench dependency.
#   Usage: bash multidomain_scale_eval.sh <MODEL> <TAG> <GPUS> [PORT] [EXTRA_VLLM_ARGS]
#     GPUS = comma list, e.g. "0,1,2,3" (tensor-parallel = count). Env-overridable: REPO PY VLLM SCRATCH.
set -u
MODEL="${1:?hf model}"; TAG="${2:?tag}"; GPUS="${3:-0}"; PORT="${4:-8055}"; EXTRA="${5:-}"
REPO=${REPO:-/home/woori/workspace_common/boltzmann-attention-pi}
PY=${PY:-/home/woori/venvs/seka_env/bin/python}
VLLM=${VLLM:-/home/woori/venvs/tau2_vllm_env/bin/vllm}
SCRATCH=${SCRATCH:-/home/woori/scratch}
MA=$REPO/scripts/distill/ma; CASES=$MA/cases
OUT=$SCRATCH/depth/c8/multidomain/results; mkdir -p $OUT
TP=$(echo $GPUS | awk -F, '{print NF}'); G0=$(echo $GPUS | cut -d, -f1)
LOG=$SCRATCH/mdscale_${TAG}.log; exec > $LOG 2>&1; set -x; date; echo "MODEL=$MODEL TAG=$TAG GPUS=$GPUS TP=$TP"
cd $REPO && git pull --ff-only 2>&1 | tail -1

for p in $(nvidia-smi --id=$G0 --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPUS setsid nohup $VLLM serve "$MODEL" --port $PORT --max-model-len 16384 \
  --tensor-parallel-size $TP $EXTRA > $SCRATCH/vllm_mdscale_${TAG}.log 2>&1 &
ok=0; for i in $(seq 1 150); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q '"id"' && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -40 $SCRATCH/vllm_mdscale_${TAG}.log; exit 1; }

# S0 (floor, gloss off) + S1 (ceiling, gloss on) on BOTH domains. tau2_op_eval reports overall acc,
# by case_op (substitute/create) breakdown, op-routing recognition, and emitted-op distribution.
for D in retail airline; do
  for G in 0 1; do
    echo "===== $TAG $D gloss=$G ====="
    $PY $MA/tau2_op_eval.py --cases $CASES/tau2_${D}_cases.jsonl --base http://localhost:$PORT/v1 \
      --model "$MODEL" --gloss $G --out $OUT/${TAG}__${D}_g${G}.json 2>&1 | sed "s/^/[$D g$G] /"
  done
done
for p in $(nvidia-smi --id=$G0 --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "MDSCALE_${TAG}_DONE"; date
