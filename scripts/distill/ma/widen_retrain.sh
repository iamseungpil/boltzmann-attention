#!/bin/bash
# Component-B training fix (M_A_RESULTS §20-22): does COVERING the B-width axis in training (substitute
# width 1-4 instead of 1-2) lift the small model's multi-attr set-extraction toward frontier? This is the
# learning-side complement to decomposition-offload. Train ONE 7B routing LoRA on wide-substitute synth,
# then eval (a) retail/airline transfer and (b) width_eval set-extraction curve. Robust; runs on a
# dedicated GPU/port (default GPU0:8069) so it coexists with the local width ladder (GPU1:8068).
#   Usage: bash widen_retrain.sh [GPU] [PORT] [SUBST_WIDTH] [N]
set -u
GPU=${1:-0}; PORT=${2:-8069}; SW=${3:-4}; N=${4:-6000}
REPO=/home/woori/workspace_common/boltzmann-attention-pi
MA=$REPO/scripts/distill/ma; DIST=$REPO/scripts/distill
S=/home/woori/scratch/depth/c8; MD=$S/multidomain; CASES=$MA/cases
PY=/home/woori/venvs/seka_env/bin/python; VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
BASE=Qwen/Qwen2.5-7B-Instruct; TAG=MD_widesubst_w${SW}
OUT=$MD/results; WOUT=$S/width; mkdir -p $OUT $WOUT $S/adapters
LOG=$MD/logs/${TAG}.log; mkdir -p $MD/logs; exec > $LOG 2>&1; set -x; date; echo "TAG=$TAG GPU=$GPU SW=$SW"
cd $REPO && git pull --ff-only 2>&1 | tail -1
kg(){ for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4; }

# 1. wide-substitute routing SFT (CPU)
$PY $MA/c8_build_sft.py --mode train --out $MD/widesubst_sft.jsonl --n $N --diverse 1 --subst_width $SW || { echo BUILD_FAIL; exit 1; }

# 2. train (dedicated GPU)
CUDA_VISIBLE_DEVICES=$GPU $PY $DIST/lora_train_chat_toolcall.py --base-model $BASE \
  --train-jsonl $MD/widesubst_sft.jsonl --out-dir $S/adapters/$TAG --device cuda:0 \
  --epochs 1 --lr 2e-4 --max-seq-len 2048 --lora-r 64 --lora-alpha 128 --grad-accum 16 --max-examples $N \
  || { echo TRAIN_FAIL; exit 1; }

# 3. serve adapter
kg
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT --max-model-len 16384 \
  --enable-lora --max-lora-rank 64 --lora-modules ${TAG}=$S/adapters/$TAG > $MD/logs/serve_${TAG}.log 2>&1 &
ok=0; for i in $(seq 1 100); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q '"id"' && ok=1 && break; sleep 6; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -20 $MD/logs/serve_${TAG}.log; exit 1; }

# 4. eval: (a) tau2 transfer  (b) width_eval set-extraction curve (does wide training lift multi-attr?)
$PY $MA/tau2_op_eval.py --cases $CASES/tau2_retail_cases.jsonl  --base http://localhost:$PORT/v1 --model $TAG --gloss 0 --out $OUT/${TAG}__retail_g0.json  2>&1 | grep -E "overall new_item|recognition"
$PY $MA/tau2_op_eval.py --cases $CASES/tau2_airline_cases.jsonl --base http://localhost:$PORT/v1 --model $TAG --gloss 0 --out $OUT/${TAG}__airline_g0.json 2>&1 | grep -E "overall new_item|recognition"
$PY $MA/width_eval.py --base http://localhost:$PORT/v1 --model $TAG --tag $TAG --widths 1,2,3,4,5 --n 100 --arms A,B --gloss 1 --out $WOUT/width_${TAG}.json
kg
echo "WIDEN_RETRAIN_DONE"; date