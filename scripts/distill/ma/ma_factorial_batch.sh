#!/bin/bash
# ===== M-sigma v4 pure-synth 2^3 FACTORIAL — one arm (mechanism leg) ====================
# arm = data gen (pure-abstract P-select, knobs) -> 7B LoRA SFT (FROZEN recipe, identical across
# arms) -> harness v4 (held-out tau2, per-provenance split). (M_SIGMA_V4 §7 / V3 factorial)
#
# Axes ISO x NL x PROV. Cell codes: M0=000 A-iso=100 A-nl=010 A-prov=001 C-in=110 C-ip=101
# C-np=011 FULL=111.  CONFOUND CONTROL: only the 3 knobs differ across arms; n/epochs/lr/seed/
# rank ALL pinned -> DO NOT EDIT per-arm.  No tau2 in training.
# SEED (4th arg) = synth-data replication for variance (tau2 n=29 noisy); seed 0 = no suffix.
# Idempotent: skips if split output already exists (resumable overnight).
#
# Usage: bash ma_factorial_batch.sh <ARM> <GPU> [PORT] [SEED]
set -u
ARM="${1:?arm: M0|A-iso|A-nl|A-prov|C-in|C-ip|C-np|FULL}"; GPU="${2:-1}"; PORT="${3:-8017}"; SEED="${4:-0}"
case "$ARM" in
  M0)     ISO=0; NL=0; PV=0;;  A-iso) ISO=1; NL=0; PV=0;;  A-nl)  ISO=0; NL=1; PV=0;;
  A-prov) ISO=0; NL=0; PV=1;;  C-in)  ISO=1; NL=1; PV=0;;  C-ip)  ISO=1; NL=0; PV=1;;
  C-np)   ISO=0; NL=1; PV=1;;  FULL)  ISO=1; NL=1; PV=1;;
  *) echo "bad arm $ARM"; exit 2;;
esac
[ "$SEED" = "0" ] && SUF="" || SUF="_s${SEED}"
TAG=$(echo "${ARM}${SUF}" | tr '-' '_')          # vllm lora-module-safe name
S=/home/woori/scratch
MA=/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/ma
TR=/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/lora_train_chat_toolcall.py
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
RUN=$S/factorial; mkdir -p $RUN
SPLIT=$RUN/split_${ARM}${SUF}.json
LOG=$S/ma_factorial_${ARM}${SUF}_g${GPU}.log
exec > $LOG 2>&1; set -x; date; echo "ARM=$ARM SEED=$SEED ISO=$ISO NL=$NL PROV=$PV TAG=$TAG"
[ -f "$SPLIT" ] && { echo "SKIP (exists: $SPLIT)"; exit 0; }   # idempotent / resumable
cd /home/woori/workspace_common/boltzmann-attention-pi && git pull --ff-only
$PY $MA/ma_gold_extract.py --out $S/ma_eval_cases.jsonl

# ---- 1. synth data (PINNED: n=2000; only knobs + SEED vary) ----
$PY $MA/synth_selection.py --out $RUN/synth_${ARM}${SUF}.jsonl --n 2000 --seed $SEED --iso $ISO --nl $NL --prov $PV \
  || { echo "SYNTH_FAIL"; exit 1; }
wc -l $RUN/synth_${ARM}${SUF}.jsonl

# ---- 2. SFT (PINNED recipe; cap/skip inert for short synth, kept identical to exp0) ----
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HP="--base-model Qwen/Qwen2.5-7B-Instruct --epochs 2 --lr 1e-4 --lora-r 32 --lora-alpha 64 \
    --max-seq-len 6144 --skip-overlong --seed 42 --device cuda:0"
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 3
CUDA_VISIBLE_DEVICES=$GPU $PY $TR $HP --train-jsonl $RUN/synth_${ARM}${SUF}.jsonl --out-dir $S/sft_runs/fact_${TAG} \
  || { echo "TRAIN_FAIL"; exit 1; }

# ---- 3. serve base + arm LoRA, eval arm with harness v4 (per-provenance split) ----
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
  --enable-lora --lora-modules ${TAG}=$S/sft_runs/fact_${TAG} --max-lora-rank 32 \
  > $S/vllm_fact_${TAG}_g${GPU}.log 2>&1 &
ok=0; for i in $(seq 1 60); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$TAG" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -30 $S/vllm_fact_${TAG}_g${GPU}.log; \
  for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; exit 1; }

echo "===== FACTORIAL arm=$ARM seed=$SEED (per-provenance split) ====="
$PY $MA/m_sigma_transfer_eval_v4.py --base http://localhost:$PORT/v1 --model "$TAG" \
  --tag "fact_${ARM}${SUF}" --out $SPLIT
# base reference = exp0 $S/exp0/split_base.json (same harness, same cases, identical base model)

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "FACTORIAL_${ARM}${SUF}_DONE"; date
