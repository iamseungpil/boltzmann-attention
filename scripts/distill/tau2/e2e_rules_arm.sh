#!/bin/bash
# prompt-vs-SFT 실험 — base 모델(LoRA 없음) e2e. rules-prompt on/off arm.
#   arm A(floor)   = RULES=0 : base + resolve wiring(선택규칙은 docstring에 있음)
#   arm B(rules)   = RULES=1 : base + 도메인-일반 rules-prompt 주입(닫힌 기저 명시)
# arm C(SFT)는 real_e2e_ground.sh(어댑터)로 별도. 같은 도메인/태스크/resolve로 비교.
# Usage: e2e_rules_arm.sh <gpu> <port> <domain> <rules:0|1> [num_tasks] [num_trials] [hf_model] [tag]
set -u
GPU=${1:?gpu}; PORT=${2:?port}; DOMAIN=${3:?domain}; RULES=${4:-1}; NT=${5:-40}; TR=${6:-1}
MODEL=${7:-Qwen/Qwen2.5-7B-Instruct}; TAG=${8:-base7b_rules}
R=/home/woori/workspace_common/boltzmann-attention-pi
T2=$R/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
SAVE=e2e_${TAG}_${DOMAIN}
GLOG=$S/ground_${TAG}_${DOMAIN}.jsonl
export T2_GROUND_LOG=$GLOG; rm -f "$GLOG"
LOG=$S/e2e_rules_${TAG}_${DOMAIN}.log
exec > $LOG 2>&1; set -x; date
echo "MODEL=$MODEL RULES=$RULES DOMAIN=$DOMAIN NT=$NT"

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $MODEL --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
  > $S/vllm_rules_${TAG}.log 2>&1 &
ok=0; for i in $(seq 1 100); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$MODEL" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -30 $S/vllm_rules_${TAG}.log; exit 1; }

set +x; source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
cd $S/tau2-bench; export PYTHONPATH=src:$T2
RULESARG=""; [ "$RULES" = "1" ] && RULESARG="--rules_prompt $T2/a2/RULES_PROMPT.txt"
rm -rf data/simulations/$SAVE
$PY $T2/t2_run_gated.py --gate 0 --resolve 1 --domain $DOMAIN --num_trials $TR --num_tasks $NT \
  $RULESARG --agent_model "$MODEL" --agent_base http://localhost:$PORT/v1 \
  --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 --save_to $SAVE \
  > $S/t2_rules_${TAG}_${DOMAIN}.log 2>&1 || echo "FAIL"
set +x
echo "=== RESULT ==="; grep -E "RESULT|pass1|rules-prompt|wiring ON" $S/t2_rules_${TAG}_${DOMAIN}.log | tail -5
echo "=== §7 ==="; $PY $T2/t2_ground_metrics.py --log "$GLOG" --results "$S/tau2-bench/data/simulations/$SAVE" 2>&1 || echo "metrics NA"
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "=== E2E_RULES DONE ($TAG·$DOMAIN·rules=$RULES) ==="; date; echo E2E_RULES_DONE
