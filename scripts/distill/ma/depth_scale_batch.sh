#!/bin/bash
# THESIS + scale eval: serve a model, run depth_eval (arms A/B/D) on synth_depth data across N.
# (B_BUDGET_SCALE_DESIGN). FROZEN: data gen seed + arms + prompts identical across model sizes.
# Usage: bash depth_scale_batch.sh <MODEL> <TAG> <GPU> [PORT] [EXTRA_VLLM_ARGS]
set -u
MODEL="${1:?hf model}"; TAG="${2:?tag}"; GPU="${3:-0}"; PORT="${4:-8025}"; EXTRA="${5:-}"
S=/home/woori/scratch; MA=/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/ma
PY=/home/woori/venvs/seka_env/bin/python; VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
RUN=$S/depth; mkdir -p $RUN
LOG=$S/depth_${TAG}_g${GPU}.log; exec > $LOG 2>&1; set -x; date; echo "MODEL=$MODEL TAG=$TAG"
cd /home/woori/workspace_common/boltzmann-attention-pi && git pull --ff-only

# ---- 1. synth data (FROZEN seed; one set per N, shared across all model sizes) ----
for NN in 5 10 20 50; do
  [ -f $RUN/synth_N${NN}.jsonl ] || $PY $MA/synth_depth.py --out $RUN/synth_N${NN}.jsonl --n 250 --N $NN --iso 1 --seed 0
done

# ---- 2. serve ----
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve "$MODEL" --port $PORT --max-model-len 16384 $EXTRA \
  > $S/vllm_depth_${TAG}.log 2>&1 &
ok=0; for i in $(seq 1 120); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q '"id"' && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -30 $S/vllm_depth_${TAG}.log; exit 1; }

# ---- 3. eval arms A (in-head) / B (op-IR+engine) / D (oracle) per N ----
for NN in 5 10 20 50; do
  echo "===== TAG=$TAG N=$NN ====="
  $PY $MA/depth_eval.py --data $RUN/synth_N${NN}.jsonl --base http://localhost:$PORT/v1 \
    --model "$MODEL" --arms A,B,D --out $RUN/depth_${TAG}_N${NN}.json
done
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "DEPTH_${TAG}_DONE"; date
