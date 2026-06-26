#!/bin/bash
# ===== M-sigma v4 EXPERIMENT 0 (lead / kill-switch) =====================================
# Matched-pair re-extraction diagnostic: does the TARGET LEVEL (typed-spec vs concrete) bind
# transfer? Trains TWO 7B LoRA arms from the SAME cfb source with IDENTICAL hyperparams,
# differing ONLY in gold target-format (typed=$ref / concrete=literal = v4-v7 neg control),
# then runs harness v4 (per-provenance split) on held-out tau2.  (M_SIGMA_V4_UNION_CORPUS_DESIGN §2)
#
# PRE-REGISTERED prediction (SUBTRACT_MAP §2): passive-$ref bucket IMPROVES typed>concrete
# (fabrication down), $select bucket FLAT for both (cfb has no selection episodes -> synth-exclusive).
# Readout = 2-way split, NOT binary.  This proves NEGATIVE+SIZING only (not "synth fixes it").
#
# Usage: bash ma_exp0_batch.sh <GPU> [PORT]      (run under setsid/nohup; watch the LOG)
set -u
GPU="${1:-0}"; PORT="${2:-8016}"
S=/home/woori/scratch
MA=/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/ma
TR=/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/lora_train_chat_toolcall.py
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
CFB=$S/fc_build/cfb.jsonl
RUN=$S/exp0; mkdir -p $RUN
LOG=$S/ma_exp0_g${GPU}.log
exec > $LOG 2>&1; set -x; date
cd /home/woori/workspace_common/boltzmann-attention-pi && git pull --ff-only

# ---- 0. held-out tau2 cases (same as M-D) ----
$PY $MA/ma_gold_extract.py --out $S/ma_eval_cases.jsonl

# ---- 1. matched pair (IDENTICAL filter/iso; only --target differs = clean neg control) ----
$PY $MA/m_sigma_data.py --src $CFB --out $RUN/cfb_typed.jsonl    --iso 1 --target typed
$PY $MA/m_sigma_data.py --src $CFB --out $RUN/cfb_concrete.jsonl --iso 1 --target concrete
wc -l $RUN/cfb_typed.jsonl $RUN/cfb_concrete.jsonl   # MUST be equal (matched)

# ---- 2. SFT two arms, IDENTICAL hyperparams (confound control: only train data differs) ----
# cfb full-catalog convs are long (median ~7.6k tok, max ~65k) -> cap 6144 + skip-overlong
# (drops pathological full-catalog examples; threading binding survives in shorter ones) +
# expandable_segments to avoid fragmentation OOM. Both arms share this -> matched preserved.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HP="--base-model Qwen/Qwen2.5-7B-Instruct --epochs 2 --lr 1e-4 --lora-r 32 --lora-alpha 64 \
    --max-seq-len 6144 --skip-overlong --seed 42 --device cuda:0"
for arm in typed concrete; do
  for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 3
  CUDA_VISIBLE_DEVICES=$GPU $PY $TR $HP --train-jsonl $RUN/cfb_${arm}.jsonl --out-dir $S/sft_runs/exp0_${arm} \
    || { echo "TRAIN_FAIL_${arm}"; exit 1; }
done

# ---- 3. serve base + both LoRA arms, eval each with harness v4 (per-provenance split) ----
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
  --enable-lora --lora-modules typed=$S/sft_runs/exp0_typed concrete=$S/sft_runs/exp0_concrete \
  --max-lora-rank 32 > $S/vllm_exp0_g${GPU}.log 2>&1 &
ok=0; for i in $(seq 1 60); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q typed && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -30 $S/vllm_exp0_g${GPU}.log; exit 1; }

echo "===== EXPERIMENT 0: per-provenance split — base vs concrete-arm vs typed-arm ====="
B=http://localhost:$PORT/v1
$PY $MA/m_sigma_transfer_eval_v4.py --base $B --model Qwen/Qwen2.5-7B-Instruct --tag base     --out $RUN/split_base.json
$PY $MA/m_sigma_transfer_eval_v4.py --base $B --model concrete                --tag concrete --out $RUN/split_concrete.json
$PY $MA/m_sigma_transfer_eval_v4.py --base $B --model typed                   --tag typed    --out $RUN/split_typed.json

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo EXP0_DONE; date
