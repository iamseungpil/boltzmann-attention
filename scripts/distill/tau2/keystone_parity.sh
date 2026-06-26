#!/bin/bash
# 키스톤 라이브 e2e 패리티 — A2-구동 scaffold의 engine arm(gate+prov+autofetch)이
# 리팩터 전과 동등 작동하는지. ★핵심 신호 = A_notfound census(autofetch-via-A2-producer
# 가 살아있으면 baseline ~9, 깨지면 ~33로 복귀) + pass^1(~0.26 baseline §35b).
# GPU1 전용(C3 sweep=GPU0 무간섭). Usage: keystone_parity.sh <gpu> <port> [num_tasks]
set -u
GPU=${1:?}; PORT=${2:?}; NT=${3:-114}
BASE=Qwen/Qwen2.5-7B-Instruct
R=/home/woori/workspace_common/boltzmann-attention-pi; T2=$R/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python; VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
LOG=$S/keystone_parity.log; exec > $LOG 2>&1; set -x; date

# GPU1만 정리(GPU0 C3 sweep 보호)
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 10
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 --enforce-eager \
  > $S/vllm_parity.log 2>&1 &
ok=0; for i in $(seq 1 120); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$BASE" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -40 $S/vllm_parity.log; exit 1; }
hc=$(curl -s localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$BASE\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3}" 2>/dev/null)
echo "$hc" | grep -q '"content"' || { echo "HC_FAIL: $hc"; exit 1; }; echo "[hc OK]"

set +x; source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
cd $S/tau2-bench; export PYTHONPATH=src:$T2
export T2_PROVENANCE=1 T2_AUTOFETCH=1; unset T2_RETRY_CONTROLLER
SAVE=keystone_parity_engine; rm -rf data/simulations/$SAVE
echo "######## engine (gate1 prov1 af1) A2-driven ########"; date
$PY $T2/t2_run_gated.py --gate 1 --resolve 0 --domain retail --num_trials 1 --num_tasks $NT \
  --agent_model "$BASE" --agent_base http://localhost:$PORT/v1 \
  --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 --save_to $SAVE > $S/parity_engine.log 2>&1 || echo RUN_FAIL
P=$(grep -oE "pass1=[0-9]+/[0-9]+" $S/parity_engine.log | tail -1)
F=$($PY $T2/t2_failcensus_deep.py data/simulations/$SAVE 2>/dev/null | grep -E "A_notfound|B_wrong|too_many" | tr '\n' ' ')
echo "PARITY_ROW engine pass=$P | $F"

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
date; echo KEYSTONE_PARITY_DONE
