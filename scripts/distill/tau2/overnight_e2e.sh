#!/bin/bash
# 밤샘 무인 — 한 base 모델 serve로 floor/rules 두 arm e2e(풀 retail·다trial). 노이즈 해소 + 크기×rules.
# 견고성: serve-check·arm마다 metrics+autopsy·끝에 GPU kill·DONE 마커. gpt-4.1 user-sim(COST GUARD).
# Usage: overnight_e2e.sh <gpu> <port> <hf_model> <tag> <num_tasks> <num_trials> [extra_serve]
set -u
GPU=${1:?}; PORT=${2:?}; MODEL=${3:?}; TAG=${4:?}; NT=${5:-0}; TR=${6:-3}; EXTRA=${7:-}
R=/home/woori/workspace_common/boltzmann-attention-pi
T2=$R/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
LOG=$S/overnight_${TAG}.log
exec > $LOG 2>&1; set -x; date; echo "MODEL=$MODEL NT=$NT TR=$TR"

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 12
# --enforce-eager: CUDA-graph capture(엔진 사망 지점) 회피 = 무인 안정성(overnight라 속도 무관)
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $MODEL --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 --enforce-eager $EXTRA \
  > $S/vllm_overnight_${TAG}.log 2>&1 &
ok=0; for i in $(seq 1 120); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$MODEL" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL_MODELS; tail -40 $S/vllm_overnight_${TAG}.log; exit 1; }
# ★실제 생성 health-check: 엔진이 살아 응답하나(죽은 엔진에 arm 돌리는 garbage 방지)
hc=$(curl -s localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3}" 2>/dev/null)
echo "$hc" | grep -q '"content"' || { echo "SERVE_FAIL_HEALTHCHECK: $hc"; tail -40 $S/vllm_overnight_${TAG}.log; \
  for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; exit 1; }
echo "[healthcheck OK] engine alive"

set +x; source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
cd $S/tau2-bench; export PYTHONPATH=src:$T2

run_arm () {  # $1=armtag $2=rulesflag
  local ARM=$1 RULES=$2 SAVE GLOG
  SAVE=on_${TAG}_${ARM}_retail; GLOG=$S/ground_on_${TAG}_${ARM}.jsonl
  export T2_GROUND_LOG=$GLOG; rm -f "$GLOG"
  local RARG=""; [ "$RULES" = "1" ] && RARG="--rules_prompt $T2/a2/RULES_PROMPT.txt"
  rm -rf data/simulations/$SAVE
  echo "######## ARM $ARM (rules=$RULES) ########"; date
  $PY $T2/t2_run_gated.py --gate 0 --resolve 1 --domain retail --num_trials $TR --num_tasks $NT \
    $RARG --agent_model "$MODEL" --agent_base http://localhost:$PORT/v1 \
    --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 --save_to $SAVE \
    > $S/t2_on_${TAG}_${ARM}.log 2>&1 || echo "FAIL $ARM"
  echo "=== $ARM RESULT ==="; grep -E "RESULT|pass1" $S/t2_on_${TAG}_${ARM}.log | tail -1
  echo "=== $ARM §7 ==="; $PY $T2/t2_ground_metrics.py --log "$GLOG" --results "$S/tau2-bench/data/simulations/$SAVE" 2>&1 | grep -E "emitted, task|present|ground_OK \| emit|routed" || echo NA
  echo "=== $ARM autopsy ==="; $PY $T2/t2_ground_autopsy.py --sim "$S/tau2-bench/data/simulations/$SAVE" --spec $T2/a2/retail.grounding.json --show 0 2>&1 | grep -E "resolve 호출|MISMATCH|UNDER_DET|UNIQUE|NO_CATALOG|ANCHOR" || echo NA
}

run_arm floor 0
run_arm rules 1

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "=== OVERNIGHT DONE ($TAG) ==="; date; echo OVERNIGHT_DONE_${TAG}
