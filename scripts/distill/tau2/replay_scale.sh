#!/bin/bash
# 선택-formalize scale 곡선 — base 모델 1개 serve + t2_selection_replay(raw-context, 비용 0·로컬 vLLM).
# 컨텍스트 소스 = 기존 실 e2e sim(real_ground_solo_cfb_mid_{retail,airline}). 크기만 바꿔 동일 컨텍스트 재생.
# Usage: replay_scale.sh <gpu> <port> <hf_model> <tag> [extra_serve_args]
set -u
GPU=${1:?gpu}; PORT=${2:?port}; MODEL=${3:?hf_model}; TAG=${4:?tag}; EXTRA=${5:-}
R=/home/woori/workspace_common/boltzmann-attention-pi
T2=$R/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
LOG=$S/replay_scale_${TAG}.log
exec > $LOG 2>&1; set -x; date
echo "MODEL=$MODEL TAG=$TAG"

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $MODEL --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 $EXTRA \
  > $S/vllm_replay_${TAG}.log 2>&1 &
ok=0; for i in $(seq 1 100); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$MODEL" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -30 $S/vllm_replay_${TAG}.log; exit 1; }

cd $S/tau2-bench; export PYTHONPATH=src:$T2
set +x
for d in retail airline; do
  echo "===== $TAG · $d ====="
  $PY $T2/t2_selection_replay.py \
    --sim $S/tau2-bench/data/simulations/real_ground_solo_cfb_mid_${d} \
    --spec $T2/a2/${d}.grounding.json --model "$MODEL" --base http://localhost:$PORT/v1 \
    --out $S/replay_${TAG}_${d}.json 2>&1
done
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "=== REPLAY_SCALE DONE ($TAG) ==="; date; echo REPLAY_SCALE_DONE
