#!/bin/bash
# ===== M-sigma v4 pure-synth 2^3 FACTORIAL — one arm (mechanism leg) ====================
# arm = data gen (pure-abstract P-select, knobs) -> 7B LoRA SFT (FROZEN recipe, identical across
# arms) -> harness v4 (held-out tau2, per-provenance split). (M_SIGMA_V4 §7 / V3 factorial)
#
# Axes ISO x NL x PROV. Cell codes: M0=000 A-iso=100 A-nl=010 A-prov=001 C-in=110 C-ip=101
# C-np=011 FULL=111.  CONFOUND CONTROL: only the 3 knobs differ across arms; n/epochs/lr/seed/
# rank ALL pinned here -> DO NOT EDIT per-arm (factorial validity).  No tau2 in training.
#
# Usage: bash ma_factorial_batch.sh <ARM> <GPU> [PORT]
set -u
ARM="${1:?arm: M0|A-iso|A-nl|A-prov|C-in|C-ip|C-np|FULL}"; GPU="${2:-1}"; PORT="${3:-8017}"
case "$ARM" in
  M0)     ISO=0; NL=0; PV=0;;  A-iso) ISO=1; NL=0; PV=0;;  A-nl)  ISO=0; NL=1; PV=0;;
  A-prov) ISO=0; NL=0; PV=1;;  C-in)  ISO=1; NL=1; PV=0;;  C-ip)  ISO=1; NL=0; PV=1;;
  C-np)   ISO=0; NL=1; PV=1;;  FULL)  ISO=1; NL=1; PV=1;;
  *) echo "bad arm $ARM"; exit 2;;
esac
S=/home/woori/scratch
MA=/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/ma
TR=/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/lora_train_chat_toolcall.py
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
RUN=$S/factorial; mkdir -p $RUN
LOG=$S/ma_factorial_${ARM}_g${GPU}.log
exec > $LOG 2>&1; set -x; date; echo "ARM=$ARM ISO=$ISO NL=$NL PROV=$PV"
cd /home/woori/workspace_common/boltzmann-attention-pi && git pull --ff-only
$PY $MA/ma_gold_extract.py --out $S/ma_eval_cases.jsonl   # held-out tau2 cases (idempotent)

# ---- 1. synth data (PINNED: n=2000, seed=0; only knobs vary) ----
$PY $MA/synth_selection.py --out $RUN/synth_${ARM}.jsonl --n 2000 --seed 0 --iso $ISO --nl $NL --prov $PV \
  || { echo "SYNTH_FAIL"; exit 1; }
wc -l $RUN/synth_${ARM}.jsonl

# ---- 2. SFT (PINNED recipe, identical to every arm + exp0) ----
# synth convs are short (~1-2k tok); cap/skip are inert here but kept IDENTICAL to exp0 for
# cross-experiment comparability + OOM safety.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HP="--base-model Qwen/Qwen2.5-7B-Instruct --epochs 2 --lr 1e-4 --lora-r 32 --lora-alpha 64 \
    --max-seq-len 6144 --skip-overlong --seed 42 --device cuda:0"
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 3
CUDA_VISIBLE_DEVICES=$GPU $PY $TR $HP --train-jsonl $RUN/synth_${ARM}.jsonl --out-dir $S/sft_runs/fact_${ARM} \
  || { echo "TRAIN_FAIL"; exit 1; }

# ---- 3. serve base + arm LoRA, eval arm with harness v4 (per-provenance split) ----
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
  --enable-lora --lora-modules ${ARM}=$S/sft_runs/fact_${ARM} --max-lora-rank 32 \
  > $S/vllm_fact_${ARM}_g${GPU}.log 2>&1 &
ok=0; for i in $(seq 1 60); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$ARM" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -30 $S/vllm_fact_${ARM}_g${GPU}.log; exit 1; }

echo "===== FACTORIAL arm=$ARM (per-provenance split) ====="
$PY $MA/m_sigma_transfer_eval_v4.py --base http://localhost:$PORT/v1 --model "$ARM" \
  --tag "fact_${ARM}" --out $RUN/split_${ARM}.json
# base reference = exp0 $S/exp0/split_base.json (same harness, same cases, identical base model)

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "FACTORIAL_${ARM}_DONE"; date
