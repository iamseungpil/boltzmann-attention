#!/bin/bash
# 도메인-파라미터화 실 τ² e2e — grounding-spec wiring 검증(A2_GROUNDING_WIRING_DESIGN S0).
# real_e2e_solo.sh(retail 하드코딩)의 일반화: DOMAIN 인자·도메인별 a2/<domain>.grounding.json 자동선택
# (런처가 파일 선택·코드 분기 0). T2_GROUND_LOG에 §7 이벤트 기록 → t2_ground_metrics로 3원인 분해.
# Usage: real_e2e_ground.sh <gpu> <port> <domain> [gate] [num_tasks] [num_trials] [adapter_dir] [tag]
set -u
GPU=${1:?gpu}; PORT=${2:?port}; DOMAIN=${3:?domain}; GATE=${4:-0}; NT=${5:-15}; TR=${6:-1}
R=/home/woori/workspace_common/boltzmann-attention-pi
T2=$R/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
BASE=Qwen/Qwen2.5-7B-Instruct
LORA=${7:-/home/woori/scratch/adapters/qwen7b_solo_cfb_mid}
TAG=${8:-solo_cfb_mid}
S=/home/woori/scratch
SAVE=real_ground_${TAG}_${DOMAIN}
GLOG=${T2_GROUND_LOG:-$S/ground_${TAG}_${DOMAIN}.jsonl}
export T2_GROUND_LOG=$GLOG
rm -f "$GLOG"
LOG=$S/real_e2e_ground_${TAG}_${DOMAIN}.log
exec > $LOG 2>&1; set -x; date
echo "DOMAIN=$DOMAIN GATE=$GATE NT=$NT TR=$TR LORA=$LORA GROUND_LOG=$GLOG"

[ -d "$LORA" ] || { echo "ADAPTER_MISSING $LORA"; exit 1; }
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
  --enable-lora --max-lora-rank 64 --lora-modules ${TAG}=${LORA} > $S/vllm_ground_${TAG}_${DOMAIN}.log 2>&1 &
ok=0; for i in $(seq 1 80); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$TAG" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -30 $S/vllm_ground_${TAG}_${DOMAIN}.log; exit 1; }

set +x; source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
cd $S/tau2-bench; export PYTHONPATH=src:$T2
rm -rf data/simulations/$SAVE
$PY $T2/t2_run_gated.py --gate $GATE --resolve 1 --domain $DOMAIN --num_trials $TR --num_tasks $NT \
  --agent_model $TAG --agent_base http://localhost:$PORT/v1 \
  --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 --save_to $SAVE \
  > $S/t2_ground_${TAG}_${DOMAIN}.log 2>&1 || echo "FAIL $DOMAIN"

set +x
echo "=== RESULT ==="; grep -E "RESULT|pass1|wiring ON|FAIL" $S/t2_ground_${TAG}_${DOMAIN}.log | tail -5
echo "=== §7 conditional metrics ==="
$PY $T2/t2_ground_metrics.py --log "$GLOG" --results "$S/tau2-bench/data/simulations/$SAVE" 2>&1 || echo "metrics NA"
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "=== GROUND E2E DONE ($TAG·$DOMAIN·gate=$GATE) ==="; date; echo GROUND_E2E_DONE
