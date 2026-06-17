#!/bin/bash
# Multi-domain content-routing transfer (HANDOFF_2026_06_17_PM §2): train ONE 7-op isotropic routing
# LoRA (NL+config -> content op-IR; filter/argmax/argmin/rank/comparative/substitute/create) on synth
# ONLY, then test config-swap transfer to BOTH retail (variant-exchange) AND airline (cabin-update)
# with NO retraining. Closure-as-learning: if the same routing fires on both domains => domain-general
# generator (attack "retail-specific" refuted). tau2-blind: synth expression pool is op-agnostic
# (render_nl_diverse), NEVER reverse-engineered from tau2.
#   Usage: bash multidomain_route.sh <GPU> <PORT> [EPOCHS] [N]
# Run ONLY when the target GPU is free (do NOT preempt the K-sweep). setsid nohup => survives ssh drop.
set -u
GPU=${1:?gpu}; PORT=${2:?port}; EP=${3:-1}; N=${4:-6000}
REPO=/home/woori/workspace_common/boltzmann-attention-pi
MA=$REPO/scripts/distill/ma; DIST=$REPO/scripts/distill
S=/home/woori/scratch/depth/c8; MD=$S/multidomain
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
BASE=Qwen/Qwen2.5-7B-Instruct
TAG=MD_route_ep${EP}
mkdir -p $MD/results $MD/logs $S/adapters
exec > $MD/logs/${TAG}_g${GPU}.log 2>&1; set -x; date; echo "TAG=$TAG GPU=$GPU PORT=$PORT EP=$EP N=$N"

kill_gpu(){ for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 5; }
serve(){ # $1=served-id $2=lora_path(empty=base)
  kill_gpu
  if [ -n "${2:-}" ]; then
    CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT --max-model-len 16384 \
      --enable-lora --max-lora-rank 64 --lora-modules ${1}=${2} > $MD/logs/serve_${1}_g${GPU}.log 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT --max-model-len 16384 \
      > $MD/logs/serve_base_g${GPU}.log 2>&1 &
  fi
  for i in $(seq 1 100); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q '"id"' && return 0; sleep 6; done
  echo "SERVE_FAIL $1"; tail -20 $MD/logs/serve_${1}_g${GPU}.log; return 1
}
# tau2 transfer eval on a domain's cases (gloss-free = pure weight transfer)
evald(){ # $1=served-id $2=domain $3=cases $4=gloss
  $PY $MA/tau2_op_eval.py --cases "$3" --base http://localhost:$PORT/v1 --model "$1" --gloss "$4" \
    --out $MD/results/${1}__$2_g$4.json 2>&1 | sed "s/^/[$2 g$4] /" || true
}

cd $REPO && git pull --ff-only 2>&1 | tail -2 || true

# ---- 0. data (CPU; safe to run while GPU busy elsewhere) ----
$PY $MA/c8_build_sft.py --mode eval  --outdir $S --n_eval 250 --diverse 1                # synth heldout (7-op routing)
$PY $MA/c8_build_sft.py --mode train --out $MD/route_sft.jsonl --n $N --diverse 1         # 7-op isotropic routing SFT
$PY $MA/ma_gold_extract.py --domain retail  --out /home/woori/scratch/ma_eval_cases.jsonl
$PY $MA/ma_gold_extract.py --domain airline --out /home/woori/scratch/ma_eval_cases_airline.jsonl
RETAIL=/home/woori/scratch/ma_eval_cases.jsonl; AIR=/home/woori/scratch/ma_eval_cases_airline.jsonl

# ---- 1. base reference on BOTH domains (floor gloss0 + ceiling gloss1) ----
if serve base ""; then
  evald $BASE retail  $RETAIL 0; evald $BASE retail  $RETAIL 1
  evald $BASE airline $AIR    0; evald $BASE airline $AIR    1
fi
kill_gpu   # ALWAYS free gpu before training

# ---- 2. train ONE routing LoRA (synth only) ----
CUDA_VISIBLE_DEVICES=$GPU $PY $DIST/lora_train_chat_toolcall.py \
  --base-model $BASE --train-jsonl $MD/route_sft.jsonl --out-dir $S/adapters/$TAG --device cuda:0 \
  --epochs $EP --lr 2e-4 --max-seq-len 2048 --lora-r 64 --lora-alpha 128 --grad-accum 16 --max-examples $N \
  || { echo "TRAIN_FAIL"; }

# ---- 3. config-swap transfer: SAME adapter -> retail + airline + synth heldout (all gloss-free) ----
if [ -d $S/adapters/$TAG ] && serve $TAG "$S/adapters/$TAG"; then
  evald $TAG retail  $RETAIL 0
  evald $TAG airline $AIR    0
  $PY $MA/depth_eval.py --data $S/c8_eval_heldout_div.jsonl --base http://localhost:$PORT/v1 \
    --model $TAG --arms B --gloss 0 --out $MD/results/${TAG}__synth_heldout.json 2>&1 | sed 's/^/[synth] /' || true
  kill_gpu
fi
echo "MULTIDOMAIN_ROUTE_DONE"; date
